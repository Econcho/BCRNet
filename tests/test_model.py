from dataclasses import replace

import pytest
import torch

from bcrnet import BCRNet, ForwardMode
from bcrnet.losses import BCRCriterion
from bcrnet.models.windows import gather_regions, scatter_cores
from bcrnet.profiling import count_macs
from bcrnet.training import configure_stage, extra_utility_probes, stage_schedule, training_forward


@pytest.mark.parametrize("mode", [m.value for m in ForwardMode])
def test_forward_modes(cfg, mode):
    model = BCRNet(cfg).eval()
    kwargs = {"window_indices": torch.tensor([[0, 5], [1, -1]])} if mode == "custom" else {}
    with torch.no_grad():
        result = model(torch.randn(2, 3, 64, 96), mode=mode, **kwargs)
    if mode == "features":
        assert result.predictions == {} and "d1" in result.features
        return
    assert result.predictions["p2"].shape == (2, 5, 16, 24)
    assert all(torch.isfinite(t).all() for t in result.predictions.values())
    if result.routing:
        assert result.routing.indices.shape[1] == (6 if mode == "dense" else 2)
        for row in result.routing.indices:
            used = row[row >= 0]
            assert len(used) == len(used.unique())
        assert torch.equal(result.predictions["p3"], result.base_predictions["p3"])


def test_base_really_skips_modules(cfg):
    model = BCRNet(cfg).eval()
    calls = []
    hooks = [
        getattr(model, key).register_forward_hook(lambda *a, k=key: calls.append(k))
        for key in ("router", "refiner", "tiny_head")
    ]
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 96), mode="base")
    assert calls == []
    for h in hooks:
        h.remove()


def test_zero_budget_and_all_padding(cfg):
    model = BCRNet(cfg).eval()
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        zero = model(x, budget=0)
        empty = model(x, torch.zeros(2, 1, 64, 96, dtype=torch.bool))
    assert zero.refinement is None
    assert torch.equal(zero.predictions["p2"], zero.base_predictions["p2"])
    assert (empty.routing.indices == -1).all()
    assert torch.isfinite(empty.predictions["p2"]).all()
    assert torch.equal(empty.predictions["p2"], empty.base_predictions["p2"])


def test_custom_validation_and_random_reproducibility(cfg):
    model = BCRNet(cfg).eval()
    x = torch.randn(1, 3, 64, 96)
    with pytest.raises(ValueError, match="unique"):
        model(x, mode="custom", window_indices=torch.tensor([[1, 1]]))
    with pytest.raises(ValueError, match="bounds"):
        model(x, mode="custom", window_indices=torch.tensor([[6]]))
    with pytest.raises(ValueError, match="multiples"):
        model(torch.randn(1, 3, 63, 96))
    with torch.no_grad():
        a = model(x, mode="random", generator=torch.Generator().manual_seed(4))
        b = model(x, mode="random", generator=torch.Generator().manual_seed(4))
    assert torch.equal(a.routing.indices, b.routing.indices)


def test_window_geometry_and_dummy_scatter():
    x = torch.arange(64.0).reshape(1, 1, 8, 8)
    ids = torch.tensor([[0, 3, -1]])
    patches = gather_regions(x, ids, 2, 4, 2)
    assert patches.shape == (1, 3, 1, 8, 8)
    assert torch.equal(patches[0, 0, 0, 2:6, 2:6], x[0, 0, :4, :4])
    assert (patches[0, 0, 0, :2] == 0).all()
    assert (patches[0, 2] == 0).all()
    cores = gather_regions(x, ids, 2, 4) + 100
    out = scatter_cores(x, cores, ids, 4)
    assert torch.equal(out[:, :, :4, :4], x[:, :, :4, :4] + 100)
    assert torch.equal(out[:, :, :4, 4:], x[:, :, :4, 4:])
    assert torch.equal(out[:, :, 4:, 4:], x[:, :, 4:, 4:] + 100)


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("A", {"backbone", "head", "tiny_head"}),
        ("B", {"reader", "refiner"}),
        ("C", {"router"}),
        ("D", {"backbone", "head", "refiner", "router"}),
    ],
)
def test_training_stage_gradients(cfg, targets, stage, expected):
    model = BCRNet(cfg)
    configure_stage(model, stage)
    criterion = BCRCriterion(cfg)
    output = training_forward(
        model, torch.randn(2, 3, 64, 96), torch.ones(2, 1, 64, 96, dtype=torch.bool), targets, stage
    )
    probes = extra_utility_probes(model, output, count=2) if stage == "D" else None
    terms = criterion(output, targets, stage=stage, probes=probes)
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    gradients = {
        name.split(".")[0]
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert expected <= gradients
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, name
    if stage == "C":
        assert model.router.scene_projection.weight.grad.abs().sum() > 0
        assert gradients == {"router"}
    if stage in {"B", "C"}:
        assert not model.backbone.training


def test_chunking_parity_and_ablation_components(cfg):
    torch.manual_seed(3)
    first = BCRNet(cfg).eval()
    second = BCRNet(replace(cfg, patch_chunk_size=1)).eval()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        assert torch.allclose(first(x).predictions["p2"], second(x).predictions["p2"], atol=1e-6)
        for options in (
            {"use_detail": False},
            {"use_context": False},
            {"use_tiny": False},
            {"refiner_type": "conv"},
        ):
            result = BCRNet(replace(cfg, **options)).eval()(x)
            assert torch.isfinite(result.predictions["p2"]).all()


def test_schedule():
    for epochs in (4, 5, 7, 10, 100):
        result = stage_schedule(epochs)
        assert len(result) == epochs and set(result) == set("ABCD")
    assert stage_schedule(2, "A") == ["A", "A"]


def test_utility_gradient_does_not_leak_in_joint_stage(cfg, targets):
    model = BCRNet(cfg)
    configure_stage(model, "D")
    criterion = BCRCriterion(cfg)
    output = model(torch.randn(2, 3, 64, 96))
    built = criterion.builder(output, targets)
    utility, _ = criterion.utility_loss(output, built)
    utility.backward()
    active = {name.split(".")[0] for name, p in model.named_parameters() if p.grad is not None}
    assert active == {"router"}


def test_executed_macs_and_frozen_mode_preservation(cfg):
    model = BCRNet(cfg)
    configure_stage(model, "B")
    states = [module.training for module in model.modules()]
    x = torch.randn(1, 3, 64, 96)
    base = count_macs(model, x, "base")
    full = count_macs(model, x, "full")
    assert full["GMAC_per_image"] > base["GMAC_per_image"]
    assert base["per_batch"]["attention_matmul"] == 0
    assert full["per_batch"]["attention_matmul"] > 0
    assert states == [module.training for module in model.modules()]


def test_multiclass_tiny_target_and_negative_loss(cfg):
    cfg.num_classes = 3
    model = BCRNet(cfg).eval()
    output = model(torch.randn(2, 3, 64, 96), mode="base_tiny")
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 11.0, 11.0]]), "labels": torch.tensor([2])},
        {
            "boxes": torch.empty(0, 4),
            "labels": torch.empty(0, dtype=torch.long),
            "ignore_boxes": torch.tensor([[0.0, 0.0, 32.0, 32.0]]),
        },
    ]
    criterion = BCRCriterion(cfg)
    built = criterion.builder(output, targets)
    assert built["p2"].heatmap[0, 2, 2, 2] == 1
    assert not built["p2"].valid[1, 0, :8, :8].any()
    loss = criterion(output, targets, stage="A")["loss"]
    loss.backward()
    assert torch.isfinite(loss)
