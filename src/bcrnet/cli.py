"""Console entry points; paths resolve explicitly and inference never needs ground truth."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .checkpoint import load_model
from .config import ModelConfig, load_yaml
from .data.transforms import Letterbox
from .engine import dataset_config, make_loader, select_device, train, write_json
from .evaluation import evaluate
from .inference import postprocess
from .models.detector import BCRNet, ForwardMode
from .profiling import count_macs

DEPLOY_MODES = [m.value for m in ForwardMode if m.value not in {"features", "custom"}]


def parser():
    root = argparse.ArgumentParser(description="BCRNet RGB single-frame detection research toolkit")
    sub = root.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--stage", choices=["all", "A", "B", "C", "D"], default="all")
    fit.add_argument("--resume")
    fit.add_argument("--weights")
    fit.add_argument("--stop-after-epochs", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--split", choices=["train", "val", "test"], default="test")
    ev.add_argument("--batch-size", type=int, default=1)
    ev.add_argument("--output", required=True)
    for p in (fit, ev):
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--max-batches", type=int, help="Diagnostic partial run; not a reportable benchmark")
    pred = sub.add_parser("predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--image", required=True, help="One RGB image")
    pred.add_argument("--output", required=True, help="New or empty result directory")
    pred.add_argument("--score", type=float, default=0.25)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", required=True)
    bench.add_argument("--checkpoint")
    bench.add_argument("--output", required=True)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--amp", action="store_true")
    for p in (fit, ev, pred, bench):
        p.add_argument("--device", default="auto")
        p.add_argument("--mode", choices=DEPLOY_MODES, default="full")
        p.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    return root


def new_output_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def run_evaluate(args):
    device = select_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    dataset, loader = make_loader(
        dataset_config(args.data),
        args.split,
        ckpt["preprocessing"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if dataset.category_to_label != ckpt["category_to_label"]:
        raise ValueError("Checkpoint/dataset category mapping mismatch")
    output = new_output_directory(args.output)
    metrics, predictions = evaluate(model, loader, dataset, device, args.mode, args.max_batches)
    write_json(output / "metrics.json", metrics)
    write_json(output / "detections.json", predictions)
    print(json.dumps(metrics, indent=2))


@torch.no_grad()
def predict(args):
    device = select_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    model.eval()
    prep = dict(ckpt["preprocessing"])
    prep["size"] = prep.pop("image_size", 640)
    with Image.open(args.image) as handle:
        original = handle.convert("RGB")
    image, mask, _, meta = Letterbox(**prep)(original)
    result = postprocess(
        model(image[None].to(device), mask[None].to(device), mode=args.mode),
        [meta],
        model.config.num_classes,
        score_threshold=args.score,
    )[0]
    output = new_output_directory(args.output)
    rows = {key: value.cpu().tolist() for key, value in result.items()}
    rows["categories"] = ckpt["categories"]
    rows["image"] = str(Path(args.image).resolve())
    write_json(output / "prediction.json", rows)
    draw = ImageDraw.Draw(original)
    for box, score, label in zip(rows["boxes"], rows["scores"], rows["labels"]):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{ckpt['categories'][label]['name']} {score:.3f}", fill="red")
    original.save(output / "prediction.jpg")
    print(str(output))


@torch.no_grad()
def benchmark(args):
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations > 0 and warmup >= 0 required")
    device = select_device(args.device)
    config = load_yaml(args.config)
    if args.checkpoint:
        model, ckpt = load_model(args.checkpoint, device)
        prep = ckpt["preprocessing"]
    else:
        model = BCRNet(ModelConfig.from_dict(config.get("model", {}))).to(device)
        prep = config.get("preprocessing", {})
    size = prep.get("image_size", 640)
    shape = (size, size) if isinstance(size, int) else size
    images = torch.randn(args.batch_size, 3, *shape, device=device)
    model.eval()
    amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.amp.autocast(device.type, enabled=amp):
        for index in range(args.warmup + args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(images, mode=args.mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= args.warmup:
                timings.append((time.perf_counter() - start) * 1000)
    report = {
        "mode": args.mode,
        "shape": list(images.shape),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__),
        "amp": amp,
        "weights": str(args.checkpoint) if args.checkpoint else "random initialization",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batch_latency_ms_mean": float(np.mean(timings)),
        "batch_latency_ms_p50": float(np.percentile(timings, 50)),
        "batch_latency_ms_p95": float(np.percentile(timings, 95)),
        "images_per_second": 1000 * args.batch_size / float(np.mean(timings)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
        "includes": "model only; excludes preprocessing, H2D transfer, decode and NMS",
        "model_config": model.config.to_dict(),
    }
    report["compute"] = count_macs(model, images, args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    args = parser().parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if getattr(args, "max_batches", None) is not None and args.max_batches <= 0:
        raise ValueError("max-batches must be positive")
    torch.set_num_threads(args.threads)
    {"train": train, "evaluate": run_evaluate, "predict": predict, "benchmark": benchmark}[args.command](args)


if __name__ == "__main__":
    main()
