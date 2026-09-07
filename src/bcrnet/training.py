"""Training-only scheduling and probes; no oracle behavior is hidden in model.eval()."""

import torch

from .models.router import ranked_indices
from .models.windows import scatter_cores


def stage_schedule(epochs, stage="all"):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if stage != "all":
        if stage not in "ABCD" or len(stage) != 1:
            raise ValueError("stage must be all/A/B/C/D")
        return [stage] * epochs
    if epochs < 4:
        raise ValueError("all-stage training needs at least 4 epochs")
    counts = [max(1, round(epochs * r)) for r in (0.4, 0.25, 0.1)]
    while sum(counts) >= epochs:
        largest = max(range(3), key=lambda i: counts[i])
        counts[largest] -= 1
    counts.append(epochs - sum(counts))
    return [s for s, count in zip("ABCD", counts) for _ in range(count)]


def configure_stage(model, stage):
    dense = {"backbone", "neck", "adapters", "head", "tiny_head"}
    active = {
        "A": dense,
        "B": {"reader", "refiner"},
        "C": {"router"},
        "D": dense | {"reader", "refiner", "router"},
    }[stage]
    model.train()
    for name, module in model.named_children():
        enabled = name in active
        module.requires_grad_(enabled)
        module.train(enabled)  # Frozen BN statistics must stay frozen too.
    return [p for p in model.parameters() if p.requires_grad]


def training_windows(routing, targets, core, grid_width, budget, coverage_count, generator=None):
    count = min(budget, routing.valid_windows.shape[1])
    selected = torch.full((len(targets), count), -1, dtype=torch.long, device=routing.coverage.device)
    coverage = ranked_indices(routing.coverage, routing.valid_windows, min(coverage_count, count))
    for b, target in enumerate(targets):
        chosen = [int(i) for i in coverage[b].tolist() if i >= 0]
        valid = set(routing.valid_windows[b].nonzero().flatten().tolist())
        boxes = target["boxes"].detach().cpu()
        sizes = (boxes[:, 2:] - boxes[:, :2]).prod(-1)
        inserted = 0
        for box in boxes[sizes.argsort()]:
            x, y = ((box[:2] + box[2:]) / 2).tolist()
            index = int(y // (4 * core)) * grid_width + int(x // (4 * core))
            if index in valid and index not in chosen and inserted < 4 and len(chosen) < count:
                chosen.append(index)
                inserted += 1
        remaining = sorted(valid - set(chosen))
        if remaining:
            order = torch.randperm(len(remaining), generator=generator).tolist()
            chosen.extend(remaining[i] for i in order[: count - len(chosen)])
        if chosen:
            selected[b, : len(chosen)] = torch.tensor(chosen, device=selected.device)
    return selected


def training_forward(model, images, masks, targets, stage, *, mode="full", generator=None):
    if stage == "A":
        return model(images, masks, mode="base_tiny")
    if stage == "D":
        if mode in {"features", "custom"}:
            raise ValueError("Final training stage needs a deployable prediction mode")
        return model(images, masks, mode=mode, generator=generator, return_features=True)
    if model.config.budget == 0:
        raise ValueError(
            "Refinement/utility warm-up requires a nonzero budget; use stage A for base ablations"
        )
    output = model(images, masks, mode="base_tiny", return_features=True)
    output.routing = model.router(
        output.features,
        output.masks,
        output.base_predictions,
        output.tiny_logits,
        compute_utility=(stage == "C"),
    )
    indices = training_windows(
        output.routing,
        targets,
        model.config.core_size,
        output.predictions["p2"].shape[-1] // model.config.core_size,
        model.config.budget,
        8 if stage == "B" else 4,
        generator,
    )
    output.routing.indices = indices
    output.refinement = model.refine_windows(
        output.features, output.masks, output.base_predictions["p2"], indices
    )
    output.predictions["p2"] = scatter_cores(
        output.base_predictions["p2"], output.refinement.refined_logits, indices, model.config.core_size
    )
    output.mode = f"train_{stage}"
    return output


@torch.no_grad()
def extra_utility_probes(model, output, generator=None, count=8):
    if output.routing is None or output.routing.utility is None or output.features is None:
        return None
    valid = output.routing.valid_windows.clone()
    ids = output.routing.indices
    used = torch.zeros_like(valid, dtype=torch.long)
    used.scatter_add_(1, ids.clamp_min(0), (ids >= 0).long())
    valid &= used == 0
    scores = torch.rand(valid.shape, generator=generator).to(valid.device)
    indices = ranked_indices(scores, valid, count)
    return model.refine_windows(output.features, output.masks, output.base_predictions["p2"], indices)
