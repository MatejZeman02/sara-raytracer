"""Intel Open Image Denoise routing: native cuda from utils/lib or pip cpu."""

from utils.smart_denoiser import (
    HAS_NATIVE_CUDA_OIDN,
    HAS_OIDN,
    HAS_PIP_OIDN,
    denoise_cuda_hdr_inplace,
    denoise_pip_ldr_inplace,
)

__all__ = [
    "HAS_NATIVE_CUDA_OIDN",
    "HAS_OIDN",
    "HAS_PIP_OIDN",
    "denoise_cuda_hdr",
    "denoise_pip_ldr",
]


def denoise_cuda_hdr(fb_device, width: int, height: int) -> None:
    """hdr denoise on gpu framebuffer (requires utils/lib cuda oidn)."""
    denoise_cuda_hdr_inplace(fb_device, width, height)


def denoise_pip_ldr(fb_ldr, width: int, height: int) -> None:
    """ldr denoise on host buffer (pip cpu oidn)."""
    denoise_pip_ldr_inplace(fb_ldr, width, height)
