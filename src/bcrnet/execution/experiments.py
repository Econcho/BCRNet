"""Reproducible measurements. Profiling is separate from uninstrumented latency."""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(data.mean()),
        "p25": float(np.percentile(data, 25)),
        "p50": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def measure(functions, device, *, warmup, iterations):
    """Round-robin ordering limits monotonic temperature/order bias. Includes host dispatch."""
    cold = {}
    for name, fn in functions.items():
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        cold[name] = (time.perf_counter() - started) * 1000
        del result
        for _ in range(warmup):
            result = fn()
            del result
    synchronize(device)
    values = {name: [] for name in functions}
    names = list(functions)
    for iteration in range(iterations):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            synchronize(device)
            started = time.perf_counter()
            result = functions[name]()
            synchronize(device)
            values[name].append((time.perf_counter() - started) * 1000)
            del result
    report = {}
    for name, fn in functions.items():
        report[name] = {
            "latency_ms": distribution(values[name]),
            "first_call_ms": cold[name],
            "first_call_scope": "First call inside measure; earlier correctness checks may have compiled/warmed it",
        }
        if torch.device(device).type == "cuda":
            synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = fn()
            synchronize(device)
            report[name]["additional_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device) - baseline
            )
            del result
    return report


def tensor_error(reference, actual):
    if reference.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "finite": True}
    delta = (reference.float() - actual.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "p99_abs": torch.quantile(delta.flatten(), 0.99).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def compare_outputs(reference, actual, *, atol, rtol):
    errors = {
        key: tensor_error(value, actual.predictions[key]) for key, value in reference.predictions.items()
    }
    close = all(
        torch.allclose(value, actual.predictions[key], atol=atol, rtol=rtol)
        for key, value in reference.predictions.items()
    )
    routing_equal = (
        reference.routing is None
        and actual.routing is None
        or reference.routing is not None
        and actual.routing is not None
        and torch.equal(reference.routing.indices, actual.routing.indices)
    )
    if reference.refinement is not None and actual.refinement is not None:
        errors["residual"] = tensor_error(reference.refinement.residual, actual.refinement.residual)
        close = close and torch.allclose(
            reference.refinement.residual, actual.refinement.residual, atol=atol, rtol=rtol
        )
    return {
        "passed": bool(close and routing_equal),
        "routing_equal": routing_equal,
        "errors": errors,
        "atol": atol,
        "rtol": rtol,
    }


@contextmanager
def module_ranges(model):
    handles, stacks = [], {}
    names = ("backbone", "neck", "router", "tiny_head", "reader", "refiner", "head")
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        stacks[name] = []

        def before(module, args, key=name):
            event = record_function(f"BCRModule::{key}")
            event.__enter__()
            stacks[key].append(event)

        def after(module, args, output, key=name):
            stacks[key].pop().__exit__(None, None, None)

        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for stack in stacks.values():
            while stack:
                stack.pop().__exit__(None, None, None)


def profile_model(model, function, device, path, *, iterations=3):
    activities = [ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    synchronize(device)
    with (
        module_ranges(model),
        profile(activities=activities, record_shapes=True, profile_memory=True) as profiler,
    ):
        for _ in range(iterations):
            result = function()
            del result
        synchronize(device)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    rows = []
    for event in profiler.key_averages():
        # GPU annotations can duplicate CPU ranges; never add both or infer percentages.
        if str(event.device_type).endswith("CPU") and (
            event.key.startswith(("BCRExec::", "BCRModule::"))
            or event.key in {"cudaLaunchKernel", "cuLaunchKernel", "aten::clone", "aten::copy_"}
        ):
            rows.append(
                {
                    "name": event.key,
                    "count_per_forward": event.count / iterations,
                    "cpu_inclusive_us_per_forward": event.cpu_time_total / iterations,
                    "cpu_self_us_per_forward": event.self_cpu_time_total / iterations,
                }
            )
    return {
        "trace": str(path.resolve()),
        "events": rows,
        "scope": "CPU annotations/counts; profiling perturbs latency; ranges are not additive",
    }


def sample_indices(pattern, batch, budget, grid_height, grid_width, *, device, seed):
    count = min(budget, grid_height * grid_width)
    if pattern == "clustered":
        cy, cx = (grid_height - 1) / 2, (grid_width - 1) / 2
        candidates = sorted(
            range(grid_height * grid_width),
            key=lambda j: (
                max(abs(j // grid_width - cy), abs(j % grid_width - cx)),
                abs(j // grid_width - cy) + abs(j % grid_width - cx),
                j,
            ),
        )
    elif pattern == "dispersed":
        # Farthest-point selection avoids assuming that any fixed step has enough slots.
        remaining, candidates = set(range(grid_height * grid_width)), []
        while remaining and len(candidates) < count:
            chosen = (
                min(remaining)
                if not candidates
                else max(
                    remaining,
                    key=lambda j: (
                        min(
                            (j // grid_width - v // grid_width) ** 2 + (j % grid_width - v % grid_width) ** 2
                            for v in candidates
                        ),
                        -j,
                    ),
                )
            )
            candidates.append(chosen)
            remaining.remove(chosen)
    elif pattern == "random":
        candidates = torch.randperm(
            grid_height * grid_width, generator=torch.Generator().manual_seed(seed)
        ).tolist()
    else:
        raise ValueError(f"Unsupported synthetic window pattern: {pattern}")
    return torch.tensor(candidates[:count], device=device, dtype=torch.long)[None].expand(batch, -1).clone()
