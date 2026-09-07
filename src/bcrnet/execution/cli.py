"""python -m bcrnet.execution: benchmark, audit and offline policy calibration."""

import argparse
import json
from dataclasses import replace
from functools import partial
from itertools import pairwise
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .. import BCRNet, ModelConfig
from ..checkpoint import load_model
from ..config import load_yaml
from ..data.datasets import build_dataset, collate_detection
from ..inference import postprocess
from .config import ExecutionConfig
from .experiments import compare_outputs, distribution, measure, profile_model, sample_indices
from .model import ExecutionBCRNet
from .policy import geometry_summary, workload_signature


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def environment(args):
    spatial_shape = [args.size, args.size] if isinstance(args.size, int) else list(args.size)
    return {
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
        "amp": args.amp,
        "reuse_feature_masks": args.reuse_feature_masks,
        "attention_backend": args.attention_backend,
        "input_shape": [args.batch_size, 3, *spatial_shape],
        "seed": args.seed,
        "measurement": "wall-clock ms with synchronization; round-robin order; no preprocessing/H2D/NMS",
    }


def executor(source, name, args, *, trace=False):
    return ExecutionBCRNet(
        source,
        ExecutionConfig(
            strategy=name,
            attention_backend=args.attention_backend,
            overlap_threshold=args.overlap_threshold,
            policy_path=args.policy,
            trace=trace,
            reuse_feature_masks=args.reuse_feature_masks,
        ),
    ).eval()


def benchmark(args):
    document = load_yaml(args.config)
    cfg = ModelConfig.from_dict(document.get("model", {}))
    if args.checkpoint:
        source, _ = load_model(args.checkpoint, args.device)
        cfg = source.config
        if args.halos and args.halos != [cfg.halo]:
            raise ValueError(
                "A checkpoint fixes halo/relative-bias geometry; do not change it for speed comparisons"
            )
    budgets, halos = args.budgets or [cfg.budget], args.halos or [cfg.halo]
    report = {
        "format_version": 1,
        "environment": environment(args),
        "cases": [],
        "scope": "synthetic inputs; random weights"
        if not args.checkpoint
        else "synthetic inputs; checkpoint weights",
        "checkpoint": args.checkpoint,
    }
    with torch.inference_mode(), torch.autocast(torch.device(args.device).type, enabled=args.amp):
        for halo in halos:
            for budget in budgets:
                torch.manual_seed(args.seed)
                if args.checkpoint:
                    source.config = replace(cfg, budget=budget)
                else:
                    source = BCRNet(replace(cfg, halo=halo, budget=budget)).to(args.device)
                source.eval()
                model_cfg = source.config
                if args.size % model_cfg.input_divisor:
                    raise ValueError(f"--size must be a multiple of {model_cfg.input_divisor}")
                models = {
                    name: executor(source, name, args)
                    for name in dict.fromkeys(["reference", *args.executors])
                }
                images = torch.randn(args.batch_size, 3, args.size, args.size, device=args.device)
                features, masks = source.extract_features(images)
                base_p2 = source.head(features["e2"])
                for pattern in args.patterns:
                    kwargs = {"mode": "full"}
                    if pattern != "router":
                        selected = sample_indices(
                            pattern,
                            args.batch_size,
                            budget,
                            args.size // 4 // model_cfg.core_size,
                            args.size // 4 // model_cfg.core_size,
                            device=args.device,
                            seed=args.seed,
                        )
                        kwargs = {"mode": "custom", "window_indices": selected}
                    expected = models["reference"](images, **kwargs)
                    selected = expected.routing.indices
                    case = {
                        "budget": budget,
                        "halo": halo,
                        "pattern": pattern,
                        "signature": workload_signature(model_cfg, features, selected),
                        "geometry": geometry_summary(selected, model_cfg, features["e2"].shape[-2:]),
                        "correctness": {},
                        "execution": {},
                    }
                    for name, model in models.items():
                        actual = model(images, **kwargs)
                        case["correctness"][name] = compare_outputs(
                            expected, actual, atol=args.atol, rtol=args.rtol
                        )
                        case["execution"][name] = model.last_execution
                        del actual
                    functions = {name: partial(model, images, **kwargs) for name, model in models.items()}
                    case["model"] = measure(
                        functions, args.device, warmup=args.warmup, iterations=args.iterations
                    )
                    subgraphs = {
                        name: (
                            lambda m=model, f=features, mask=masks, base=base_p2, ids=selected: (
                                m.merge_refinement(base, m.refine_windows(f, mask, base, ids))
                            )
                        )
                        for name, model in models.items()
                    }
                    case["refinement_and_writeback"] = measure(
                        subgraphs, args.device, warmup=args.warmup, iterations=args.iterations
                    )
                    case["geometry_selector"] = measure(
                        {
                            "geometry": partial(
                                geometry_summary, selected, model_cfg, features["e2"].shape[-2:]
                            )
                        },
                        args.device,
                        warmup=1,
                        iterations=max(3, args.iterations // 3),
                    )["geometry"]
                    if args.profile:
                        case["profiles"] = {}
                        for name in models:
                            traced = executor(source, name, args, trace=True)
                            function = partial(traced, images, **kwargs)
                            function()  # Compile/warm before profiler.
                            trace = (
                                Path(args.output).parent / f"trace_k{budget}_h{halo}_{pattern}_{name}.json"
                            )
                            case["profiles"][name] = profile_model(traced, function, args.device, trace)
                    report["cases"].append(case)
                    write_json(args.output, report)
                    summary = {
                        name: round(row["latency_ms"]["p50"], 3) for name, row in case["model"].items()
                    }
                    print(
                        json.dumps({"K": budget, "halo": halo, "pattern": pattern, "p50_ms": summary}),
                        flush=True,
                    )
                    del expected
    report["passed"] = all(
        check["passed"] for case in report["cases"] for check in case["correctness"].values()
    )
    write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError(f"Numerical audit failed; inspect {args.output}")


def audit(args):
    source, checkpoint = load_model(args.checkpoint, args.device)
    preprocessing = checkpoint.get("preprocessing", {})
    if args.size is None:
        args.size = preprocessing.get("image_size", 640)
    source.eval()
    optimized = executor(source, args.executor, args)
    config_path = Path(args.dataset).resolve()
    config = load_yaml(config_path)
    root = Path(config["root"])
    config["root"] = str(root if root.is_absolute() else config_path.parent / root)
    dataset = build_dataset(
        config, args.split, image_size=args.size, mean=preprocessing.get("mean"), std=preprocessing.get("std")
    )
    if dataset.num_classes != source.config.num_classes:
        raise ValueError("Dataset category count does not match checkpoint")
    if "category_to_label" in checkpoint and dataset.category_to_label != checkpoint["category_to_label"]:
        raise ValueError("Dataset category mapping does not match checkpoint")
    loader = DataLoader(
        Subset(dataset, range(min(len(dataset), args.limit))),
        batch_size=args.batch_size,
        collate_fn=collate_detection,
        shuffle=False,
        num_workers=0,
    )
    rows, checks, detected_equal = [], [], []
    with torch.inference_mode(), torch.autocast(torch.device(args.device).type, enabled=args.amp):
        for images, masks, targets in loader:
            images, masks = images.to(args.device), masks.to(args.device)
            expected, actual = source(images, masks), optimized(images, masks)
            checks.append(compare_outputs(expected, actual, atol=args.atol, rtol=args.rtol))
            meta = [target["meta"] for target in targets]
            before, after = (
                postprocess(expected, meta, source.config.num_classes),
                postprocess(actual, meta, source.config.num_classes),
            )
            for i, target in enumerate(targets):
                left, right = before[i], after[i]
                same = all(
                    left[key].shape == right[key].shape
                    and torch.allclose(left[key].float(), right[key].float(), atol=args.atol, rtol=args.rtol)
                    for key in ("boxes", "scores", "labels")
                )
                detected_equal.append(same)
                stats = geometry_summary(
                    expected.routing.indices[i : i + 1], source.config, expected.masks["e2"].shape[-2:]
                )
                boxes = target["boxes"]
                tiny = bool(
                    len(boxes)
                    and ((boxes[:, 2:] - boxes[:, :2]).amax(1) < source.config.tiny_threshold).any()
                )
                rows.append(
                    {
                        "image_id": meta[i]["image_id"],
                        "sequence_id": meta[i].get("sequence_id", ""),
                        "positive": bool(len(boxes)),
                        "tiny_at_input_scale": tiny,
                        "geometry": stats,
                        "detections_ordered_close": same,
                    }
                )
    report = {
        "format_version": 1,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(config_path),
        "environment": environment(args),
        "rows": rows,
        "tensor_checks": checks,
        "tensor_checks_passed": all(c["passed"] for c in checks),
        "ordered_detection_agreement": sum(detected_equal) / len(detected_equal) if detected_equal else None,
        "detail_ratio": distribution([r["geometry"]["detail_ratio"] for r in rows]),
        "semantic_ratio": distribution([r["geometry"]["semantic_ratio"] for r in rows]),
        "scope": "No training, extraction or timing; GT is used only to label analysis groups",
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"rows", "tensor_checks"}}, ensure_ascii=False
        )
    )
    if not rows:
        raise ValueError("Audit dataset is empty")
    if not report["tensor_checks_passed"]:
        raise RuntimeError("Tensor/route audit failed; inspect saved report")


def calibrate(args):
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cases = [case for case in data["cases"] if case["budget"] == args.budget and case["halo"] == args.halo]
    if not cases:
        raise ValueError("No matching benchmark cases")
    signature = cases[0]["signature"]
    if any(case["signature"] != signature for case in cases):
        raise ValueError("Calibration requires a single workload signature")
    for case in cases:
        if args.shared_strategy not in case["model"] or "packed" not in case["model"]:
            raise ValueError("Calibration needs packed and the requested shared strategy")
        if any(not case["correctness"][name]["passed"] for name in ("packed", args.shared_strategy)):
            raise ValueError("Cannot calibrate a backend that failed numerical validation")
    packed = [c["model"]["packed"]["latency_ms"]["p50"] for c in cases]
    shared = [c["model"][args.shared_strategy]["latency_ms"]["p50"] for c in cases]
    overhead = [c["geometry_selector"]["latency_ms"]["p50"] for c in cases]
    candidates = [("packed", None, sum(packed)), (args.shared_strategy, None, sum(shared))]
    ratios = sorted({c["geometry"]["ratio"] for c in cases})
    for threshold in sorted({1.0, *ratios, *[(a + b) / 2 for a, b in pairwise(ratios)]}):
        total = sum(
            (shared[i] if case["geometry"]["ratio"] >= threshold else packed[i]) + overhead[i]
            for i, case in enumerate(cases)
        )
        candidates.append(("threshold", threshold, total))
    choice, threshold, cost = min(candidates, key=lambda v: v[2])
    policy = {
        "format_version": 1,
        "signature": signature,
        "choice": choice,
        "threshold": threshold,
        "shared_strategy": args.shared_strategy,
        "source_report": str(Path(args.report).resolve()),
        "scope": data["scope"],
        "attention_backend": data["environment"]["attention_backend"],
        "reuse_feature_masks": data["environment"]["reuse_feature_masks"],
        "estimated_mean_ms": cost / len(cases),
        "cases": len(cases),
        "warning": "Offline estimate includes measured selector cost; validate on held-out inputs before deployment",
    }
    write_json(args.output, policy)
    print(json.dumps(policy, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--config", default="configs/bcrnet.yaml")
    bench.add_argument("--checkpoint")
    bench.add_argument("--budgets", nargs="+", type=int)
    bench.add_argument("--halos", nargs="+", type=int)
    bench.add_argument(
        "--executors", nargs="+", default=["reference", "packed", "shared", "indexed", "adaptive"]
    )
    bench.add_argument(
        "--patterns",
        nargs="+",
        choices=["router", "clustered", "dispersed", "random"],
        default=["router", "clustered", "dispersed"],
    )
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument("--profile", action="store_true")
    inspect = commands.add_parser("audit")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--dataset", required=True)
    inspect.add_argument("--split", choices=["train", "val", "test"], default="val")
    inspect.add_argument("--limit", type=int, default=1000)
    inspect.add_argument("--executor", choices=["packed", "shared", "indexed", "adaptive"], default="indexed")
    for child in (bench, inspect):
        child.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        child.add_argument("--size", type=int, default=640)
        child.add_argument("--batch-size", type=int, default=1)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--amp", action="store_true")
        child.add_argument("--attention-backend", default="auto")
        child.add_argument("--overlap-threshold", type=float, default=1.4)
        child.add_argument("--policy")
        child.add_argument("--reuse-feature-masks", action="store_true")
        child.add_argument("--atol", type=float, default=3e-3)
        child.add_argument("--rtol", type=float, default=3e-3)
        child.add_argument("--threads", type=int, default=4)
        child.add_argument("--output", required=True)
    calibration = commands.add_parser("calibrate")
    inspect.set_defaults(size=None)
    calibration.add_argument("--report", required=True)
    calibration.add_argument("--budget", type=int, default=16)
    calibration.add_argument("--halo", type=int, default=4)
    calibration.add_argument("--shared-strategy", choices=["shared", "indexed"], default="indexed")
    calibration.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command != "calibrate":
        if (args.size is not None and args.size < 1) or args.batch_size < 1 or args.threads < 1:
            parser.error("size, batch-size and threads must be positive")
        if args.command == "benchmark" and (args.iterations < 1 or args.warmup < 0):
            parser.error("iterations must be positive; warmup must be nonnegative")
        if args.command == "audit" and args.limit < 1:
            parser.error("limit must be positive")
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
    {"benchmark": benchmark, "audit": audit, "calibrate": calibrate}[args.command](args)
