"""Single-device reference trainer. All experiment state is explicit and checkpointed."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import atomic_save, load_checkpoint, load_compatible_weights, restore_rng, rng_state
from .config import ModelConfig, load_yaml
from .data.datasets import build_dataset, collate_detection
from .evaluation import evaluate
from .losses import BCRCriterion
from .models.detector import BCRNet
from .training import configure_stage, extra_utility_probes, stage_schedule, training_forward


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def select_device(value="auto"):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def dataset_config(path):
    path = Path(path).resolve()
    cfg = load_yaml(path)
    root = Path(cfg["root"])
    cfg["root"] = str(root if root.is_absolute() else (path.parent / root).resolve())
    return cfg


def make_loader(data, split, preprocessing, *, batch_size=1, workers=0, training=False, generator=None):
    dataset = build_dataset(data, split, training=training, **preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return dataset, loader


def dataset_fingerprint(*datasets):
    # Hash annotation content, not just the path, so split changes invalidate resume.
    return {
        str(d.annotation_path.resolve()): hashlib.sha256(d.annotation_path.read_bytes()).hexdigest()
        for d in datasets
    }


def train(args):
    config = load_yaml(args.config)
    settings = config.get("training", {})
    preprocessing = config.get("preprocessing", {"image_size": 640})
    model_cfg = ModelConfig.from_dict(config.get("model", {}))
    size = preprocessing.get("image_size", 640)
    shape = (size, size) if isinstance(size, int) else size
    if any(s % model_cfg.input_divisor for s in shape):
        raise ValueError("Preprocessing size must divide model.input_divisor")
    seed = int(settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(args.device)
    epochs = args.epochs or settings.get("epochs", 100)
    schedule = stage_schedule(epochs, args.stage)
    run_dir = Path(args.output).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError("Run output is nonempty; use --resume or choose a new directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = dataset_config(args.data)
    loader_rng, route_rng = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)
    batch_size = args.batch_size or settings.get("batch_size", 2)
    train_set, train_loader = make_loader(
        data,
        "train",
        preprocessing,
        batch_size=batch_size,
        workers=args.workers,
        training=True,
        generator=loader_rng,
    )
    val_set, val_loader = make_loader(data, "val", preprocessing, batch_size=batch_size, workers=args.workers)
    if (
        train_set.category_to_label != val_set.category_to_label
        or train_set.num_classes != model_cfg.num_classes
    ):
        raise ValueError("Model/train/val category mapping mismatch")
    fingerprint = dataset_fingerprint(train_set, val_set)
    model = BCRNet(model_cfg).to(device)
    criterion = BCRCriterion(model_cfg, **config.get("loss", {})).to(device)
    amp = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp, init_scale=float(settings.get("amp_init_scale", 16))
    )
    optimizer = scheduler = None
    start_epoch, best_ap, previous_stage = 0, -float("inf"), None
    saved = None
    if args.resume:
        if args.weights:
            raise ValueError("--resume and --weights are mutually exclusive")
        saved = load_checkpoint(args.resume)
        checks = {
            "model_config": model_cfg.to_dict(),
            "schedule": schedule,
            "preprocessing": preprocessing,
            "dataset_fingerprint": fingerprint,
            "category_to_label": train_set.category_to_label,
            "training_settings": settings,
            "loss_settings": config.get("loss", {}),
            "workers": args.workers,
            "amp": amp,
            "batch_size": batch_size,
            "mode": args.mode,
            "max_batches": args.max_batches,
        }
        for key, expected in checks.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch: {key}; use --weights for a new experiment")
        model.load_state_dict(saved["model"])
        criterion.load_state_dict(saved["criterion"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, best_ap = saved["epoch"] + 1, saved["best_ap"]
        log_path = run_dir / "metrics.jsonl"
        if log_path.exists():
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            if records and records[-1]["epoch"] != start_epoch:
                raise ValueError("Run log/checkpoint epoch mismatch; resume into a new output directory")
        previous_stage = saved["stage"]
        loader_rng.set_state(saved["loader_rng"])
        route_rng.set_state(saved["route_rng"])
        restore_rng(saved["rng"])
    elif args.weights:
        write_json(run_dir / "warm_start.json", load_compatible_weights(model, args.weights))
    write_json(
        run_dir / "config.json",
        {
            "experiment": config,
            "data": data,
            "schedule": schedule,
            "mode": args.mode,
            "device": str(device),
            "batch_size": batch_size,
            "torch": str(torch.__version__),
            "dataset_fingerprint": fingerprint,
        },
    )
    for epoch in range(start_epoch, len(schedule)):
        stage = schedule[epoch]
        parameters = configure_stage(model, stage)
        criterion.train()
        if optimizer is None or stage != previous_stage:
            lr = float(settings.get("lr", 0.001)) * float(
                settings.get("stage_lr_multipliers", {}).get(stage, 1.0)
            )
            optimizer = torch.optim.AdamW(
                parameters, lr=lr, weight_decay=float(settings.get("weight_decay", 0.01))
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=schedule.count(stage), eta_min=lr * 0.05
            )
            if saved is not None and stage == saved["stage"]:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
        saved = None
        previous_stage = stage
        sums, batches, skipped_updates, consecutive_skips = {}, 0, 0, 0
        started = time.perf_counter()
        for batch_index, (images, masks, targets) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                output = training_forward(
                    model, images, masks, targets, stage, mode=args.mode, generator=route_rng
                )
                probes = None
                if stage == "D" and random.random() < float(settings.get("probe_probability", 0.1)):
                    probes = extra_utility_probes(model, output, route_rng)
                terms = criterion(output, targets, stage=stage, probes=probes)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(settings.get("grad_clip", 10.0)), error_if_nonfinite=not amp
            )
            scaler.step(optimizer)
            scaler.update()
            if not torch.isfinite(norm):
                skipped_updates += 1
                consecutive_skips += 1
                if consecutive_skips >= 16:
                    raise FloatingPointError("16 consecutive AMP gradient overflows; inspect the model/data")
            else:
                consecutive_skips = 0
            batches += 1
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
        if not batches:
            raise ValueError("Empty training loader")
        if skipped_updates == batches:
            raise FloatingPointError("No optimizer update succeeded this epoch; lower AMP scale or use FP32")
        scheduler.step()
        # Every phase is evaluated with a deployable mode, never training-only GT routing.
        metrics, _ = evaluate(
            model, val_loader, val_set, device, mode=args.mode, max_batches=args.max_batches
        )
        improved = metrics["AP50_95"] > best_ap
        best_ap = max(best_ap, metrics["AP50_95"])
        record = {
            "epoch": epoch + 1,
            "stage": stage,
            "amp_skipped_updates": skipped_updates,
            "amp_scale": scaler.get_scale(),
            "train": {k: v / batches for k, v in sums.items()},
            "val": metrics,
            "seconds": time.perf_counter() - started,
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "stage": stage,
            "schedule": schedule,
            "model_config": model_cfg.to_dict(),
            "preprocessing": preprocessing,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_ap": best_ap,
            "rng": rng_state(),
            "loader_rng": loader_rng.get_state(),
            "route_rng": route_rng.get_state(),
            "dataset_fingerprint": fingerprint,
            "category_to_label": train_set.category_to_label,
            "categories": train_set.categories,
            "training_settings": settings,
            "loss_settings": config.get("loss", {}),
            "workers": args.workers,
            "amp": amp,
            "batch_size": batch_size,
            "mode": args.mode,
            "max_batches": args.max_batches,
        }
        atomic_save(run_dir / "last.pt", payload)
        if improved:
            atomic_save(run_dir / "best.pt", payload)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.stop_after_epochs is not None and epoch - start_epoch + 1 >= args.stop_after_epochs:
            break
    return run_dir
