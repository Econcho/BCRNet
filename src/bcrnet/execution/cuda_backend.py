"""Lazy NVRTC + CUDA Driver API bridge; no compiler subprocess or global install.

The module reads packaged CUDA source and compiles it in memory. Kernel handles are
cached per CUDA context/device/dtype; feature values and index plans are never cached.
"""

import ctypes as ct
import ctypes.util
import math
import os
import threading
from pathlib import Path

import torch

_RUNTIME = None
_LOCK = threading.RLock()
_KERNELS = {}


class BackendUnavailable(RuntimeError):
    pass


class _Runtime:
    def __init__(self):
        torch_lib = Path(torch.__file__).parent / "lib"
        self.dll_dirs = []
        if os.name == "nt":
            self.dll_dirs.append(os.add_dll_directory(str(torch_lib)))
            candidates = list(torch_lib.glob("nvrtc64_*.dll"))
            candidates = [p for p in candidates if ".alt." not in p.name]
            if os.environ.get("CUDA_PATH"):
                cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
                if cuda_bin.exists():
                    self.dll_dirs.append(os.add_dll_directory(str(cuda_bin)))
                    candidates += list(cuda_bin.glob("nvrtc64_*.dll"))
            driver = "nvcuda.dll"
        else:
            candidates = list(torch_lib.glob("libnvrtc.so*"))
            site = Path(torch.__file__).parent.parent
            candidates += list((site / "nvidia" / "cuda_nvrtc" / "lib").glob("libnvrtc.so*"))
            found = ctypes.util.find_library("nvrtc")
            if found:
                candidates.append(found)
            driver = ctypes.util.find_library("cuda") or "libcuda.so.1"
        if not candidates:
            raise BackendUnavailable("NVRTC was not found; use sdpa/torch_indexed or install a CUDA runtime")
        try:
            self.nvrtc = ct.CDLL(str(candidates[0]))
            self.driver = ct.CDLL(driver)
        except OSError as exc:
            raise BackendUnavailable(f"Cannot load NVRTC/CUDA driver: {exc}") from exc
        self.library = str(candidates[0])
        signatures = {
            "nvrtcCreateProgram": [
                ct.POINTER(ct.c_void_p),
                ct.c_char_p,
                ct.c_char_p,
                ct.c_int,
                ct.c_void_p,
                ct.c_void_p,
            ],
            "nvrtcCompileProgram": [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)],
            "nvrtcGetProgramLogSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "nvrtcGetProgramLog": [ct.c_void_p, ct.c_void_p],
            "nvrtcGetPTXSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "nvrtcGetPTX": [ct.c_void_p, ct.c_void_p],
            "nvrtcDestroyProgram": [ct.POINTER(ct.c_void_p)],
        }
        for name, args in signatures.items():
            fn = getattr(self.nvrtc, name)
            fn.argtypes, fn.restype = args, ct.c_int
        signatures = {
            "cuInit": [ct.c_uint],
            "cuCtxGetCurrent": [ct.POINTER(ct.c_void_p)],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
            "cuLaunchKernel": [
                ct.c_void_p,
                ct.c_uint,
                ct.c_uint,
                ct.c_uint,
                ct.c_uint,
                ct.c_uint,
                ct.c_uint,
                ct.c_uint,
                ct.c_void_p,
                ct.POINTER(ct.c_void_p),
                ct.c_void_p,
            ],
        }
        for name, args in signatures.items():
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.check_driver(self.driver.cuInit(0))

    def check_driver(self, status):
        if status:
            message = ct.c_char_p()
            self.driver.cuGetErrorString(status, ct.byref(message))
            raise RuntimeError(f"CUDA driver error {status}: {message.value!r}")

    @staticmethod
    def check_nvrtc(status):
        if status:
            raise BackendUnavailable(f"NVRTC error {status}")

    def compile(self, dtype, device):
        source = (Path(__file__).parent / "kernels" / "indexed_attention.cu").read_bytes()
        program = ct.c_void_p()
        self.check_nvrtc(
            self.nvrtc.nvrtcCreateProgram(
                ct.byref(program), source, b"bcrnet_indexed_attention.cu", 0, None, None
            )
        )
        major, minor = torch.cuda.get_device_capability(device)
        options = [
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            b"--std=c++14",
            f"-DBCR_HALF={int(dtype == torch.float16)}".encode(),
            b"--fmad=false",
        ]
        try:
            status = self.nvrtc.nvrtcCompileProgram(
                program, len(options), (ct.c_char_p * len(options))(*options)
            )
            size = ct.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            self.nvrtc.nvrtcGetProgramLog(program, log)
            if status:
                raise BackendUnavailable(f"NVRTC compilation failed ({status}): {log.value.decode()}")
            self.check_nvrtc(self.nvrtc.nvrtcGetPTXSize(program, ct.byref(size)))
            ptx = ct.create_string_buffer(size.value)
            self.check_nvrtc(self.nvrtc.nvrtcGetPTX(program, ptx))
        finally:
            self.nvrtc.nvrtcDestroyProgram(ct.byref(program))
        module, function = ct.c_void_p(), ct.c_void_p()
        self.check_driver(self.driver.cuModuleLoadData(ct.byref(module), ptx))
        self.check_driver(self.driver.cuModuleGetFunction(ct.byref(function), module, b"indexed_attention"))
        return module, function


def _kernel(dtype, device):
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = _Runtime()
        context = ct.c_void_p()
        _RUNTIME.check_driver(_RUNTIME.driver.cuCtxGetCurrent(ct.byref(context)))
        key = (context.value, device.index, dtype)
        if key not in _KERNELS:
            _KERNELS[key] = _RUNTIME.compile(dtype, device)
        return _RUNTIME, _KERNELS[key][1]


def warmup_backend(device="cuda", dtype=torch.float16):
    """Compile before timed runs; raises rather than pretending another backend was used."""
    device = torch.device(device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise BackendUnavailable("cuda_indexed requires a CUDA device")
    with torch.cuda.device(device):
        torch.empty(1, device=device)  # Initialize the selected primary context.
        runtime, _ = _kernel(dtype, torch.device("cuda", torch.cuda.current_device()))
        return {"backend": "cuda_indexed", "nvrtc": runtime.library}


def indexed_attention(query, kv_bank, inverse, valid, bias):
    if torch.is_grad_enabled():
        raise RuntimeError("cuda_indexed is inference-only; use torch.no_grad()/inference_mode()")
    if query.device.type != "cuda" or query.dtype not in {torch.float16, torch.float32}:
        raise BackendUnavailable("cuda_indexed supports CUDA float16/float32")
    n, heads, nq, dim = query.shape
    nm = inverse.shape[1]
    if not 1 <= dim <= 256 or nm < 1:
        raise BackendUnavailable("Supported head_dim is 1..256; at least one memory token is required")
    if inverse.shape[0] != n or valid.shape != inverse.shape or bias.shape != (heads, nq, nm):
        raise ValueError("Indexed attention shape mismatch")
    if kv_bank.ndim != 4 or kv_bank.shape[1:] != (2, heads, dim):
        raise ValueError("KV bank must have shape [U,2,heads,head_dim]")
    if any(t.device != query.device for t in (kv_bank, inverse, valid, bias)):
        raise ValueError("All attention tensors must be on the same CUDA device")
    if kv_bank.dtype != query.dtype or bias.dtype != query.dtype:
        raise ValueError("Q, KV and bias must share dtype")
    if inverse.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("inverse must be int64 and valid must be bool")
    tensors = [t.contiguous() for t in (query, kv_bank, inverse, valid, bias)]
    output = torch.empty_like(tensors[0])
    if n == 0:
        return output
    shared_bytes = (nm + 256 + dim) * 4
    limit = torch.cuda.get_device_properties(query.device).shared_memory_per_block
    if shared_bytes > limit:
        raise BackendUnavailable(f"Attention requires {shared_bytes} shared bytes; device limit is {limit}")
    with torch.cuda.device(query.device):
        runtime, function = _kernel(query.dtype, query.device)
        args = [ct.c_void_p(t.data_ptr()) for t in (*tensors, output)]
        args += [ct.c_int(v) for v in (heads, nq, nm, dim, kv_bank.shape[0])]
        args += [ct.c_float(1 / math.sqrt(dim))]
        pointers = (ct.c_void_p * len(args))(*[ct.cast(ct.pointer(a), ct.c_void_p) for a in args])
        runtime.check_driver(
            runtime.driver.cuLaunchKernel(
                function,
                n * heads * nq,
                1,
                1,
                256,
                1,
                1,
                shared_bytes,
                ct.c_void_p(torch.cuda.current_stream(query.device).cuda_stream),
                pointers,
                None,
            )
        )
    return output
