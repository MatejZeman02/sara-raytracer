"""route oidn: load native cuda from utils/lib when possible, else pip cpu.

place these next to each other in utils/lib (from the same intel oidn release):
  libOpenImageDenoise.so.2.x.y (c api; symlinks optional)
  libOpenImageDenoise_core.so.2.x.y
  libOpenImageDenoise_device_cuda.so.2.x.y
"""

from __future__ import annotations

import ctypes
import glob
import os
import platform
import warnings

import numpy as np

_utils_dir = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_utils_dir, "lib")

HAS_NATIVE_CUDA_OIDN = False
HAS_PIP_OIDN = False
_NATIVE_LIB_PATH: str | None = None
oidn_native = None  # CDLL handle: c api in libOpenImageDenoise.so*


def _find_api_lib() -> str | None:
    cands = glob.glob(os.path.join(_LIB_DIR, "libOpenImageDenoise.so.*.*"))
    if cands:
        return sorted(cands)[-1]
    cands2 = glob.glob(os.path.join(_LIB_DIR, "libOpenImageDenoise.so"))
    return cands2[0] if cands2 else None


def _find_cuda_device_lib() -> str | None:
    cands = glob.glob(os.path.join(_LIB_DIR, "libOpenImageDenoise_device_cuda.so*"))
    if not cands:
        return None
    return sorted(cands, key=lambda p: (len(os.path.basename(p)), p))[-1]


def _has_core_lib() -> bool:
    return bool(glob.glob(os.path.join(_LIB_DIR, "libOpenImageDenoise_core.so*")))


def _bind_native_symbols(lib: ctypes.CDLL) -> None:
    # cuda shared pointers require a cuda device from oidnNewCUDADevice (not oidnNewDevice)
    lib.oidnNewCUDADevice.restype = ctypes.c_void_p
    lib.oidnNewCUDADevice.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    lib.oidnNewFilter.restype = ctypes.c_void_p
    lib.oidnSetSharedFilterImage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.oidnSetFilterBool.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_bool,
    ]
    lib.oidnCommitFilter.argtypes = [ctypes.c_void_p]
    lib.oidnExecuteFilter.argtypes = [ctypes.c_void_p]
    lib.oidnReleaseFilter.argtypes = [ctypes.c_void_p]
    lib.oidnReleaseDevice.argtypes = [ctypes.c_void_p]


_api_path = _find_api_lib()
_cuda_device_path = _find_cuda_device_lib()

if _api_path and _cuda_device_path and _has_core_lib():
    _lib_dir = os.path.dirname(os.path.abspath(_api_path))
    _saved_ld_path = os.environ.get("LD_LIBRARY_PATH")
    try:
        if platform.system() != "Windows":
            _prev = os.environ.get("LD_LIBRARY_PATH", "")
            _parts = [p for p in _prev.split(os.pathsep) if p]
            if _lib_dir not in _parts:
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([_lib_dir] + _parts)
        # c entry points (oidnNewDevice, etc.) live in libOpenImageDenoise.so*, not in device_*.so
        _api = ctypes.CDLL(_api_path)
        # device module only registers the cuda backend; must be loaded into the process
        ctypes.CDLL(_cuda_device_path, mode=ctypes.RTLD_GLOBAL)
        _bind_native_symbols(_api)
        oidn_native = _api
        HAS_NATIVE_CUDA_OIDN = True
        _NATIVE_LIB_PATH = _api_path
    except OSError as exc:
        warnings.warn(
            f"OIDN native cuda setup failed (api {_api_path}, device {_cuda_device_path}): {exc}",
            RuntimeWarning,
        )
        oidn_native = None
    finally:
        if platform.system() != "Windows":
            if _saved_ld_path is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = _saved_ld_path

try:
    import oidn as _oidn_pip

    if hasattr(_oidn_pip, "NewDevice") and hasattr(_oidn_pip, "DEVICE_TYPE_CPU"):
        HAS_PIP_OIDN = True
    else:
        _oidn_pip = None
except ImportError:
    _oidn_pip = None

HAS_OIDN = HAS_NATIVE_CUDA_OIDN or HAS_PIP_OIDN


def denoise_cuda_hdr_inplace(fb_device, width: int, height: int) -> None:
    """in-place RT denoise on hdr linear float3 using cuda oidn shared buffer."""
    from numba import cuda

    if not HAS_NATIVE_CUDA_OIDN or oidn_native is None:
        raise RuntimeError("native cuda oidn is not available")

    # oidn.h: OIDN_FORMAT_FLOAT=1, FLOAT2=2, FLOAT3=3 (do not use 1 for rgb buffers)
    OIDN_FORMAT_FLOAT3 = 3

    cuda.synchronize()
    dev_id = int(cuda.get_current_device().id)
    _ids = (ctypes.c_int * 1)(dev_id)
    # null stream pointer -> default stream per oidn.h
    device = oidn_native.oidnNewCUDADevice(_ids, None, 1)
    oidn_native.oidnCommitDevice(device)
    filt = None
    try:
        filt = oidn_native.oidnNewFilter(device, b"RT")
        ptr = fb_device.device_ctypes_pointer.value
        oidn_native.oidnSetSharedFilterImage(
            filt, b"color", ptr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0
        )
        oidn_native.oidnSetSharedFilterImage(
            filt, b"output", ptr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0
        )
        oidn_native.oidnSetFilterBool(filt, b"hdr", True)
        oidn_native.oidnCommitFilter(filt)
        oidn_native.oidnExecuteFilter(filt)
    finally:
        if filt is not None:
            oidn_native.oidnReleaseFilter(filt)
        oidn_native.oidnReleaseDevice(device)
    cuda.synchronize()


def denoise_pip_ldr_inplace(fb_ldr: np.ndarray, width: int, height: int) -> None:
    """in-place RT denoise on ldr float3 via pip cpu oidn (no hdr flag)."""
    if not HAS_PIP_OIDN or _oidn_pip is None:
        warnings.warn(
            "pip oidn is not installed; skipping denoising.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    assert fb_ldr.dtype == np.float32
    assert fb_ldr.shape == (height, width, 3)

    if not fb_ldr.flags.c_contiguous:
        fb_ldr[:] = np.ascontiguousarray(fb_ldr)

    device = _oidn_pip.NewDevice(_oidn_pip.DEVICE_TYPE_CPU)
    _oidn_pip.CommitDevice(device)
    filt = None
    try:
        filt = _oidn_pip.NewFilter(device, "RT")
        _oidn_pip.SetSharedFilterImage(
            filt, "color", fb_ldr, _oidn_pip.FORMAT_FLOAT3, width, height
        )
        _oidn_pip.SetSharedFilterImage(
            filt, "output", fb_ldr, _oidn_pip.FORMAT_FLOAT3, width, height
        )
        _oidn_pip.CommitFilter(filt)
        _oidn_pip.ExecuteFilter(filt)
    finally:
        if filt is not None:
            _oidn_pip.ReleaseFilter(filt)
        _oidn_pip.ReleaseDevice(device)
