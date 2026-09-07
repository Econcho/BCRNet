"""An inference execution adapter with unchanged parameter/state_dict names."""

import copy
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from ..models.detector import BCRNet, RefinementOutput
from ..models.refinement import ContextReader, ContextRefiner, CrossAttentionBlock
from .attention import run_shared_refiner
from .config import ExecutionConfig
from .context import prepare_context
from .plan import AccessPlan, AccessPlanner, read_core, write_cores
from .policy import AdaptivePolicy


@dataclass
class ExecutionRefinementOutput(RefinementOutput):
    access_plan: AccessPlan | None = None


class ExecutionBCRNet(BCRNet):
    def __init__(self, source, execution=None, *, clone=True):
        if type(source) is not BCRNet:
            raise TypeError("Use a BCRNet instance; custom detector subclasses need their own adapter")
        nn.Module.__init__(self)
        original = copy.deepcopy(source) if clone else source
        self.config = copy.deepcopy(source.config)
        for name, module in original.named_children():
            self.add_module(name, module)
        self.training = source.training
        self.execution = execution or ExecutionConfig()
        self.policy = AdaptivePolicy(self.execution)
        self.last_execution = None
        self._native_status = {}
        self.planner = AccessPlanner()

    def scope(self, name):
        return record_function(f"BCRExec::{name}") if self.execution.trace else nullcontext()

    def _compatible(self):
        return (
            type(self.reader) is ContextReader
            and type(self.refiner) is ContextRefiner
            and self.config.refiner_type == "attention"
            and all(type(block) is CrossAttentionBlock for block in self.refiner.blocks)
        )

    def _backend(self, strategy, feature):
        choice = self.execution.attention_backend
        if choice != "auto":
            return choice, None
        if strategy == "shared":
            return "sdpa", None
        if feature.device.type != "cuda" or feature.dtype not in {torch.float16, torch.float32}:
            return "torch_indexed", "portable_indexed_backend"
        if self.config.width // self.config.heads > 256:
            return "torch_indexed", "native_head_dimension_limit"
        from .cuda_backend import BackendUnavailable, warmup_backend

        key = (feature.device, feature.dtype)
        if key not in self._native_status:
            try:
                warmup_backend(feature.device, feature.dtype)
                self._native_status[key] = None
            except BackendUnavailable as exc:
                self._native_status[key] = str(exc)
        error = self._native_status[key]
        return ("cuda_indexed", None) if error is None else ("torch_indexed", error)

    def forward(self, *args, **kwargs):
        self.last_execution = None
        output = super().forward(*args, **kwargs)
        if self.last_execution is None:
            self.last_execution = {"strategy": "skipped", "reason": "mode_or_zero_budget"}
        return output

    def extract_features(self, images, valid_mask=None):
        if not self.execution.reuse_feature_masks:
            return super().extract_features(images, valid_mask)
        # Conventional strong-baseline optimization; same pooling definition, once per H/W.
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("images must be floating-point NCHW RGB")
        if any(n % self.config.input_divisor for n in images.shape[-2:]):
            raise ValueError(f"Image H/W must be multiples of {self.config.input_divisor}")
        if valid_mask is None:
            valid_mask = torch.ones_like(images[:, :1], dtype=torch.bool)
        if valid_mask.shape != images[:, :1].shape or valid_mask.device != images.device:
            raise ValueError("valid_mask must have shape B,1,H,W on the image device")
        features = self.backbone(images)
        pyramid = self.neck(features)
        features.update(pyramid)
        features.update({f"e{i}": self.adapters[f"e{i}"](pyramid[f"p{i}"]) for i in (2, 3, 4)})
        masks, cache = {}, {}
        valid_float = valid_mask.float()
        for key, tensor in features.items():
            size = tensor.shape[-2:]
            if size not in cache:
                cache[size] = F.adaptive_max_pool2d(valid_float, size).bool()
            masks[key] = cache[size]
        return features, masks

    def refine_windows(self, features, masks, base_p2, indices):
        requested = self.execution.strategy
        fallback = None
        if self.training or torch.is_grad_enabled():
            fallback = "training_or_grad_enabled"
        elif not self._compatible():
            fallback = "custom_reader_refiner_or_conv_ablation"
        elif indices.numel() == 0:
            fallback = "empty_selection"
        if requested == "reference" or fallback:
            self.last_execution = {"strategy": "reference", "reason": fallback or "requested"}
            return super().refine_windows(features, masks, base_p2, indices)
        with self.scope("choose"):
            strategy, decision = (
                self.policy.choose(indices, self.config, features)
                if requested == "adaptive"
                else (requested, {"reason": "requested"})
            )
        with self.scope("access_plan"):
            plan = self.planner.build(indices, self.config, features, shared=strategy != "packed")
        bank = prepare_context(self.reader, features, masks, plan, scope=self.scope)
        chunk = self.config.patch_chunk_size
        if strategy == "packed":
            with self.scope("packed_refiner"):
                memory = bank.materialize()
                residual = torch.cat(
                    [
                        self.refiner(
                            bank.query[i : i + chunk],
                            memory[i : i + chunk],
                            bank.valid[i : i + chunk],
                            bank.core[i : i + chunk],
                            bank.core_valid[i : i + chunk],
                        )
                        for i in range(0, plan.slots, chunk)
                    ]
                )
            backend, backend_note = "original_sdpa", None
        else:
            backend, backend_note = self._backend(strategy, features["e2"])
            residual = run_shared_refiner(
                self.refiner,
                bank,
                backend,
                chunk_size=chunk,
                key_tile=self.execution.key_tile,
                scope=self.scope,
            )
        with self.scope("local_prediction"):
            b, k = indices.shape
            c = self.config.core_size
            base = read_core(base_p2, plan).reshape(b, k, -1, c, c)
            cmask = bank.core_valid.reshape(b, k, 1, c, c)
            refined = self.head(bank.core + residual).reshape(b, k, -1, c, c)
            refined = torch.where(cmask, refined, base)
        self.last_execution = {
            "requested": requested,
            "strategy": strategy,
            "backend": backend,
            "backend_note": backend_note,
            "decision": decision,
            "plan": plan.metadata(),
        }
        return ExecutionRefinementOutput(
            indices,
            base,
            refined,
            residual.reshape(b, k, -1, c, c),
            cmask,
            plan,
        )

    def merge_refinement(self, base_p2, refinement):
        if isinstance(refinement, ExecutionRefinementOutput):
            with self.scope("writeback"):
                return write_cores(
                    base_p2, refinement.refined_logits, refinement.access_plan, refinement.base_logits
                )
        return super().merge_refinement(base_p2, refinement)
