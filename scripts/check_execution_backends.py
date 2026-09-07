"""Read-only dependency diagnostics plus a small compiled-CUDA capability probe."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from bcrnet.execution.cuda_backend import warmup_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
        "triton_installed": importlib.util.find_spec("triton") is not None,
    }
    if report["cuda"]:
        report["gpu"] = torch.cuda.get_device_name()
        try:
            report["cuda_indexed"] = warmup_backend()
        except (OSError, RuntimeError) as exc:
            report["cuda_indexed"] = {"error": str(exc)}
        try:
            compiled = torch.compile(lambda x: x.sin() + x.cos(), fullgraph=True)
            with torch.inference_mode():
                compiled(torch.randn(512, device="cuda"))
            torch.cuda.synchronize()
            report["compile_cuda_smoke"] = {"passed": True, "scope": "elementwise only, not full BCRNet"}
        except (RuntimeError, ImportError, OSError) as exc:
            report["compile_cuda_smoke"] = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2500],
            }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
