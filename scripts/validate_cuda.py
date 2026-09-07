"""Full-size synthetic CUDA forward/backward verification. No real dataset access."""

import json
from pathlib import Path

import torch

from bcrnet import BCRNet, ModelConfig
from bcrnet.losses import BCRCriterion
from bcrnet.training import configure_stage, training_forward


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires CUDA")
    cfg = ModelConfig()
    model = BCRNet(cfg).cuda()
    configure_stage(model, "D")
    criterion = BCRCriterion(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=16)
    images = torch.randn(1, 3, 640, 640, device="cuda")
    masks = torch.ones(1, 1, 640, 640, device="cuda", dtype=torch.bool)
    targets = [
        {
            "boxes": torch.tensor([[310.0, 290.0, 317.0, 295.0], [60.0, 70.0, 130.0, 150.0]]),
            "labels": torch.tensor([0, 0]),
        }
    ]
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda"):
        output = training_forward(model, images, masks, targets, "D")
        terms = criterion(output, targets, stage="D")
    assert torch.isfinite(terms["loss"])
    scaler.scale(terms["loss"]).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    report = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "input": list(images.shape),
        "mode": "full",
        "stage": "D",
        "amp": True,
        "finite_loss": float(terms["loss"].detach()),
        "gradient_norm_before_clip": float(norm),
        "optimizer_step": True,
        "selected_windows": output.routing.indices.cpu().tolist(),
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    path = Path("runs/validation/cuda_training.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
