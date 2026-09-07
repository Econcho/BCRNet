from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from bcrnet import BCRNet, ForwardMode
from bcrnet.execution import ExecutionBCRNet, ExecutionConfig, build_access_plan
from bcrnet.execution.attention import gathered_sdpa, tiled_indexed
from bcrnet.execution.context import prepare_context
from bcrnet.execution.plan import write_cores
from bcrnet.execution.policy import geometry_summary, rectangle_union
from bcrnet.models.windows import scatter_cores


def stress_weights(model):
    # Nonzero affine/type/bias terms expose invalid-token bugs hidden by default initialization.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith(("reader.", "refiner.")):
                if "norm" in name and name.endswith("weight"):
                    param.uniform_(0.7, 1.3)
                else:
                    param.normal_(0, 0.15)
        model.refiner.residual_scale.fill_(0.7)


@pytest.mark.parametrize("halo", [0, 2, 4, 8])
@pytest.mark.parametrize("detail,context", [(True, True), (False, True), (True, False)])
def test_context_bank_matches_original_including_pool_phase(cfg, halo, detail, context):
    cfg = replace(cfg, halo=halo, use_detail=detail, use_context=context)
    model = BCRNet(cfg).eval()
    stress_weights(model)
    images = torch.randn(2, 3, 64, 96)
    mask = torch.rand(2, 1, 64, 96) > 0.4
    mask[1] = False
    indices = torch.tensor([[0, 1, 2, 5, -1], [3, 0, -1, 4, 2]])
    with torch.inference_mode():
        features, masks = model.extract_features(images, mask)
        expected = model.reader(features, masks, indices)
        for shared in (False, True):
            plan = build_access_plan(indices, cfg, features, shared=shared)
            bank = prepare_context(model.reader, features, masks, plan, scope=lambda _: nullcontext())
            actual = (bank.query, bank.materialize(), bank.valid, bank.core, bank.core_valid)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("strategy", ["packed", "shared", "indexed", "adaptive"])
@pytest.mark.parametrize("mode", [m.value for m in ForwardMode])
def test_all_model_modes_preserve_semantics(cfg, strategy, mode):
    torch.manual_seed(913)
    model = BCRNet(cfg).eval()
    stress_weights(model)
    optimized = ExecutionBCRNet(model, ExecutionConfig(strategy=strategy, key_tile=17))
    images = torch.randn(2, 3, 64, 96)
    mask = torch.ones(2, 1, 64, 96, dtype=torch.bool)
    mask[1, :, 49:] = False
    kwargs = {"window_indices": torch.tensor([[0, 5], [1, -1]])} if mode == "custom" else {}
    with torch.inference_mode():
        a = model(images, mask, mode=mode, generator=torch.Generator().manual_seed(5), **kwargs)
        b = optimized(images, mask, mode=mode, generator=torch.Generator().manual_seed(5), **kwargs)
    assert set(model.state_dict()) == set(optimized.state_dict())
    for key in a.predictions:
        torch.testing.assert_close(a.predictions[key], b.predictions[key], atol=2e-5, rtol=2e-5)
    if a.routing:
        assert torch.equal(a.routing.indices, b.routing.indices)
    if a.refinement:
        torch.testing.assert_close(a.refinement.residual, b.refinement.residual, atol=2e-5, rtol=2e-5)
        selected = torch.zeros_like(a.base_predictions["p2"], dtype=torch.bool)
        selected = scatter_cores(
            selected.float(),
            a.refinement.valid_mask.float().expand_as(a.refinement.refined_logits),
            a.refinement.indices,
            cfg.core_size,
        ).bool()
        assert torch.equal(b.predictions["p2"][~selected], a.predictions["p2"][~selected])
        assert torch.equal(a.predictions["p3"], b.predictions["p3"])
        assert torch.equal(a.predictions["p4"], b.predictions["p4"])


def test_dummy_zero_budget_fallback_and_state(cfg):
    base = BCRNet(cfg).eval()
    optimized = ExecutionBCRNet(base, ExecutionConfig(strategy="shared"))
    x = torch.randn(2, 3, 64, 96)
    with torch.inference_mode():
        out = optimized(x, torch.zeros(2, 1, 64, 96, dtype=torch.bool))
        assert all(torch.isfinite(p).all() for p in out.predictions.values())
        assert (out.routing.indices == -1).all()
        torch.testing.assert_close(out.predictions["p2"], out.base_predictions["p2"], rtol=0, atol=0)
        out = optimized(x, budget=0)
        assert out.refinement is None
        assert optimized.last_execution["strategy"] == "skipped"
    optimized.train()
    optimized(x).predictions["p2"].sum().backward()
    assert optimized.last_execution["reason"] == "training_or_grad_enabled"
    assert optimized.reader.detail_projection.weight.grad is not None
    assert base.reader.detail_projection.weight.grad is None
    conv = ExecutionBCRNet(BCRNet(replace(cfg, refiner_type="conv")).eval())
    with torch.inference_mode():
        conv(x)
    assert conv.last_execution["reason"] == "custom_reader_refiner_or_conv_ablation"
    with pytest.raises(ValueError, match="unique"), torch.inference_mode():
        optimized.eval()(x, mode="custom", window_indices=torch.tensor([[0, 0], [1, 1]]))


def test_plan_scatter_reordered_and_sentinel(cfg):
    features = {"e2": torch.randn(2, cfg.width, 16, 24), "e3": torch.randn(2, cfg.width, 8, 12)}
    indices = torch.tensor([[5, -1, 0], [-1, 4, 2]])
    plan = build_access_plan(indices, cfg, features, shared=True)
    base = torch.randn(2, 5, 16, 24)
    refined = torch.randn(2, 3, 5, 8, 8)
    assert torch.equal(write_cores(base, refined, plan), scatter_cores(base, refined, indices, 8))


def test_static_cache_does_not_reuse_frame_values_and_mask_baseline(cfg):
    base = BCRNet(replace(cfg, core_size=4, halo=2)).eval()
    optimized = ExecutionBCRNet(base, ExecutionConfig(strategy="shared", reuse_feature_masks=True))
    with torch.inference_mode():
        for width in (64, 96, 64):
            image = torch.randn(2, 3, 64, width)
            mask = torch.rand(2, 1, 64, width) > 0.5
            expected, actual = base(image, mask), optimized(image, mask)
            torch.testing.assert_close(expected.predictions["p2"], actual.predictions["p2"])
            assert all(torch.equal(expected.masks[key], actual.masks[key]) for key in expected.masks)
    assert len(optimized.planner.templates) == 1  # Only static B/K/core/halo/device, no frame content.
    copied = BCRNet(base.config)
    copied.load_state_dict(optimized.state_dict(), strict=True)


def test_geometry_phase_matches_enumerated_supports(cfg):
    for halo in (0, 2, 4, 8):
        local = replace(cfg, halo=halo)
        ids = torch.tensor([[0, 1, 5, -1], [0, 2, 5, 4]])
        expected = geometry_summary(ids, local, (16, 24))
        features = {"e2": torch.randn(2, cfg.width, 16, 24), "e3": torch.randn(2, cfg.width, 8, 12)}
        plan = build_access_plan(ids, local, features, shared=True)
        assert expected["detail_unique"] == int(plan.detail.active.sum())
        assert expected["semantic_unique"] == int(plan.semantic.active.sum())


def test_geometric_overlap_and_policy(cfg):
    cfg = replace(cfg, budget=16)
    indices = torch.tensor([[r * 20 + c for r in range(5, 9) for c in range(5, 9)]])
    stats = geometry_summary(indices, cfg, (160, 160))
    assert stats["detail_references"] == 4096
    assert stats["detail_unique"] == 1600
    assert stats["detail_ratio"] == 2.56
    assert stats["semantic_unique"] == 100
    assert rectangle_union([(0, 0, 2, 2), (1, 1, 3, 3)]) == 7
    model = ExecutionBCRNet(
        BCRNet(cfg).eval(),
        ExecutionConfig(
            strategy="adaptive", overlap_threshold=1, minimum_references=0, attention_backend="sdpa"
        ),
    )
    with torch.inference_mode():
        model(torch.randn(1, 3, 64, 96))
    assert model.last_execution["strategy"] == "indexed"
    assert model.last_execution["backend"] == "sdpa"


@pytest.mark.parametrize("dim", [4, 16, 24, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_cuda_attention_equivalence(dim, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from bcrnet.execution.cuda_backend import indexed_attention

    torch.manual_seed(24)
    n, heads, nq, nm, unique = 3, 2, 7, 41, 53
    q = torch.randn(n, heads, nq, dim, device="cuda", dtype=dtype)
    kv = torch.randn(unique, 2, heads, dim, device="cuda", dtype=dtype)
    inverse = torch.randint(unique, (n, nm), device="cuda")
    valid = torch.rand(n, nm, device="cuda") > 0.4
    valid[0, :25] = False  # All-masked early tiles must not poison online softmax.
    valid[:, -1] = True
    bias = torch.randn(heads, nq, nm, device="cuda", dtype=dtype) * 0.2
    with torch.inference_mode():
        expected = gathered_sdpa(q, kv, inverse, valid, bias)
        actual = indexed_attention(q, kv, inverse, valid, bias)
        tiled = tiled_indexed(q, kv, inverse, valid, bias, key_tile=11)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(tiled, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("strategy", ["packed", "shared", "indexed"])
def test_cuda_full_amp_and_alternative_stream(cfg, strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = BCRNet(cfg).cuda().eval()
    stress_weights(model)
    optimized = ExecutionBCRNet(model, ExecutionConfig(strategy=strategy))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode(), torch.autocast("cuda"):
        x = torch.randn(2, 3, 64, 96, device="cuda")
        mask = torch.ones(2, 1, 64, 96, dtype=torch.bool, device="cuda")
        mask[1] = False
        a, b = model(x, mask), optimized(x, mask)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(a.predictions["p2"], b.predictions["p2"], atol=3e-3, rtol=3e-3)
    assert torch.equal(a.routing.indices, b.routing.indices)
    assert torch.isfinite(b.predictions["p2"]).all()
