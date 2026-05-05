"""Post-processing pipeline: denoising and tonemapping."""

import os
import numpy as np
from .settings import settings
from utils.smart_denoiser import HAS_NATIVE_CUDA_OIDN, HAS_PIP_OIDN
from utils.smart_denoiser import denoise_cuda_hdr_inplace, denoise_pip_ldr_inplace
from .framebuffer import tonemap_hdr_to_sdr

if settings.DEVICE == "gpu":
    from numba import cuda


def denoise_gpu_hdr(fb_device, width: int, height: int) -> None:
    """GPU-side HDR denoising (requires native CUDA OIDN library)."""
    if not HAS_NATIVE_CUDA_OIDN:
        return
    denoise_cuda_hdr_inplace(fb_device, width, height)


def denoise_cpu_ldr(fb_ldr: np.ndarray, width: int, height: int) -> None:
    """CPU-side LDR denoising (pip OIDN library)."""
    if not HAS_PIP_OIDN:
        return
    denoise_pip_ldr_inplace(fb_ldr, width, height)


def get_denoise_path(use_gpu_lut: bool) -> tuple:
    """Determine which denoising backends are available.

    Returns:
        (can_denoise_on_gpu, can_denoise_cpu)
    """
    can_denoise_gpu = HAS_NATIVE_CUDA_OIDN and settings.DEVICE == "gpu"
    can_denoise_cpu = HAS_PIP_OIDN
    return can_denoise_gpu, can_denoise_cpu
