import numpy as np
import torch
from PIL import Image


class Letterbox:
    """RGB preprocessing and reversible, integer-rounding-aware geometry."""

    def __init__(self, size=(640, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("size must be positive H,W")
        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)
        if (self.std <= 0).any():
            raise ValueError("std must be positive")

    def __call__(self, image, boxes=None):
        image = image.convert("RGB")
        width, height = image.size
        out_h, out_w = self.size
        scale = min(out_w / width, out_h / height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        px, py = (out_w - rw) // 2, (out_h - rh) // 2
        canvas = Image.new("RGB", (out_w, out_h), (114, 114, 114))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (px, py))
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
        tensor = (tensor - self.mean) / self.std
        mask = torch.zeros(1, out_h, out_w, dtype=torch.bool)
        mask[:, py : py + rh, px : px + rw] = True
        meta = {
            "scale": (rw / width, rh / height),
            "pad": (px, py),
            "original_size": (height, width),
            "input_size": self.size,
            "valid_bounds": (px, py, px + rw, py + rh),
        }
        if boxes is None:
            boxes = torch.empty(0, 4)
        transformed = boxes.clone().float()
        transformed[:, [0, 2]] = transformed[:, [0, 2]] * (rw / width) + px
        transformed[:, [1, 3]] = transformed[:, [1, 3]] * (rh / height) + py
        return tensor, mask, transformed, meta


def flip_sample(image, mask, target):
    width = image.shape[-1]
    result = dict(target)
    for name in ("boxes", "ignore_boxes"):
        boxes = target.get(name, torch.empty(0, 4)).clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        result[name] = boxes
    # Geometry metadata is for evaluation only; flipped training samples are never inverse-decoded.
    return image.flip(-1), mask.flip(-1), result
