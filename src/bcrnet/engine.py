"""Single-device reference trainer. All experiment state is explicit and checkpointed."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import atomic_save, load_checkpoint, load_compatible_weights, restore_rng, rng_state
from .config import ModelConfig, load_yaml
from .data.datasets import build_dataset, collate_detection
from .evaluation import evaluate
from .losses import BCRCriterion
from .models.detector import BCRNet
from .training import configure_stage, extra_utility_probes, stage_schedule, training_forward

MONITOR_METRICS = ("AP50_95", "AP50", "AP75", "AP_small_original_COCO", "AR100")


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


def _resolve_training_controls(args, settings):
    """Resolve CLI-over-YAML controls and validate the training-loop contract."""

    def value(name, default):
        override = getattr(args, name, None)
        return default if override is None else override

    configured_early_stop_stage = settings.get("early_stop_stage")
    if getattr(args, "early_stop_stage", None) is not None:
        early_stop_stage = args.early_stop_stage
    elif args.stage != "all":
        # A single-stage CLI override should not inherit the full A/B/C/D default.
        early_stop_stage = args.stage
    else:
        early_stop_stage = configured_early_stop_stage or "D"
    controls = {
        "val_interval": int(value("val_interval", settings.get("val_interval", 1))),
        "patience": int(value("patience", settings.get("patience", 0))),
        "min_delta": float(value("min_delta", settings.get("min_delta", 0.0))),
        "monitor": str(value("monitor", settings.get("monitor", "AP50_95"))),
        "early_stop_stage": str(early_stop_stage),
        "progress": bool(value("progress", settings.get("progress", True))),
    }
    if controls["val_interval"] <= 0:
        raise ValueError("training.val_interval must be a positive integer")
    if controls["patience"] < 0:
        raise ValueError("training.patience must be nonnegative; zero disables early stopping")
    if controls["min_delta"] < 0:
        raise ValueError("training.min_delta must be nonnegative")
    if controls["monitor"] not in MONITOR_METRICS:
        raise ValueError(f"training.monitor must be one of {MONITOR_METRICS}")
    if controls["early_stop_stage"] not in {"A", "B", "C", "D"}:
        raise ValueError("training.early_stop_stage must be one of A/B/C/D")
    return controls


def _should_validate(epoch_number, total_epochs, val_interval, stage_changed):
    """Validate on the configured cadence plus stage transitions and the final epoch."""

    return epoch_number % val_interval == 0 or epoch_number == total_epochs or stage_changed


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


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
    controls = _resolve_training_controls(args, settings)
    if controls["early_stop_stage"] not in set(schedule):
        raise ValueError(
            f"training.early_stop_stage={controls['early_stop_stage']!r} is not present in the requested stage schedule"
        )
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
    start_epoch = 0
    best_ap = -float("inf")
    best_metric = -float("inf")
    best_epoch = None
    bad_validations = 0
    last_validation_epoch = None
    previous_stage = None
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
            "val_interval": controls["val_interval"],
            "patience": controls["patience"],
            "min_delta": controls["min_delta"],
            "monitor": controls["monitor"],
            "early_stop_stage": controls["early_stop_stage"],
        }
        legacy_defaults = {
            "val_interval": 1,
            "patience": 0,
            "min_delta": 0.0,
            "monitor": "AP50_95",
            "early_stop_stage": controls["early_stop_stage"],
        }
        for key, expected in checks.items():
            actual = saved.get(key, legacy_defaults.get(key))
            if actual != expected:
                raise ValueError(f"Resume mismatch: {key}; use --weights for a new experiment")
        model.load_state_dict(saved["model"])
        criterion.load_state_dict(saved["criterion"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"] + 1
        best_ap = saved.get("best_ap", -float("inf"))
        saved_best_metric = saved.get("best_metric", best_ap)
        best_metric = -float("inf") if saved_best_metric is None else float(saved_best_metric)
        best_epoch = saved.get("best_epoch")
        bad_validations = saved.get("bad_validations", 0)
        last_validation_epoch = saved.get("last_validation_epoch")
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
            "training_controls": controls,
        },
    )
    for epoch in range(start_epoch, len(schedule)):
        stage = schedule[epoch]
        stage_changed = previous_stage is not None and stage != previous_stage
        entered_early_stop_stage = stage_changed and stage == controls["early_stop_stage"]
        if entered_early_stop_stage:
            # Warm-up stages are not comparable to the final deployable stage.
            best_ap = -float("inf")
            best_metric = -float("inf")
            best_epoch = None
            bad_validations = 0
            last_validation_epoch = None
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
        batch_iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{len(schedule)} stage {stage}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not controls["progress"],
        )
        try:
            for batch_index, (images, masks, targets) in enumerate(batch_iterator):
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
                        raise FloatingPointError(
                            "16 consecutive AMP gradient overflows; inspect the model/data"
                        )
                else:
                    consecutive_skips = 0
                batches += 1
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                if controls["progress"]:
                    batch_iterator.set_postfix(
                        loss=f"{sums['loss'] / batches:.4g}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
        finally:
            batch_iterator.close()
        if not batches:
            raise ValueError("Empty training loader")
        if skipped_updates == batches:
            raise FloatingPointError("No optimizer update succeeded this epoch; lower AMP scale or use FP32")
        scheduler.step()
        epoch_number = epoch + 1
        validate = _should_validate(epoch_number, len(schedule), controls["val_interval"], stage_changed)
        early_stop_eligible = stage == controls["early_stop_stage"]
        metrics = None
        monitor_value = None
        improved = False
        if validate:
            # Every validation uses a deployable mode, never training-only GT routing.
            if controls["progress"]:
                tqdm.write(f"Validating epoch {epoch_number}/{len(schedule)} (monitor={controls['monitor']})")
            metrics, _ = evaluate(
                model, val_loader, val_set, device, mode=args.mode, max_batches=args.max_batches
            )
            if controls["monitor"] not in metrics:
                raise ValueError(
                    f"Validation did not return monitor metric {controls['monitor']!r}; "
                    f"available metrics: {sorted(metrics)}"
                )
            monitor_value = float(metrics[controls["monitor"]])
            if not np.isfinite(monitor_value):
                raise FloatingPointError(
                    f"Validation monitor {controls['monitor']} is not finite at epoch {epoch_number}"
                )
            if early_stop_eligible:
                ap_value = float(metrics["AP50_95"])
                best_ap = max(best_ap, ap_value)
                improved = monitor_value > best_metric + controls["min_delta"]
                if improved:
                    best_metric = monitor_value
                    best_epoch = epoch_number
                    bad_validations = 0
                else:
                    bad_validations += 1
                last_validation_epoch = epoch_number
            else:
                bad_validations = 0
        early_stop = bool(
            validate
            and early_stop_eligible
            and controls["patience"] > 0
            and bad_validations >= controls["patience"]
        )
        record = {
            "epoch": epoch_number,
            "stage": stage,
            "amp_skipped_updates": skipped_updates,
            "amp_scale": scaler.get_scale(),
            "train": {k: v / batches for k, v in sums.items()},
            "val": metrics,
            "validated": validate,
            "monitor": controls["monitor"],
            "early_stop_stage": controls["early_stop_stage"],
            "early_stop_active": early_stop_eligible,
            "monitor_value": monitor_value,
            "best_metric": _finite_or_none(best_metric),
            "best_epoch": best_epoch,
            "bad_validations": bad_validations,
            "patience": controls["patience"],
            "val_interval": controls["val_interval"],
            "early_stopping": early_stop,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
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
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_validations": bad_validations,
            "last_validation_epoch": last_validation_epoch,
            "val_interval": controls["val_interval"],
            "patience": controls["patience"],
            "min_delta": controls["min_delta"],
            "monitor": controls["monitor"],
            "early_stop_stage": controls["early_stop_stage"],
            "early_stopped": early_stop,
            "training_controls": controls,
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
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if controls["progress"]:
            tqdm.write(line)
        else:
            print(line, flush=True)
        if early_stop:
            if controls["progress"]:
                tqdm.write(
                    f"Early stopping at epoch {epoch_number}: "
                    f"{controls['monitor']} did not improve for {bad_validations} validation events."
                )
            break
        if args.stop_after_epochs is not None and epoch - start_epoch + 1 >= args.stop_after_epochs:
            break
    return run_dir
