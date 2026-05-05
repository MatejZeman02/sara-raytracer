"""Deprecated: OIDN routing moved to utils.smart_denoiser.
This module kept for backward compatibility only.
"""

from utils.smart_denoiser import (
    HAS_NATIVE_CUDA_OIDN,
    HAS_OIDN,
    HAS_PIP_OIDN,
    denoise_cuda_hdr_inplace as denoise_cuda_hdr,
    denoise_pip_ldr_inplace as denoise_pip_ldr,
)

__all__ = [
    "HAS_NATIVE_CUDA_OIDN",
    "HAS_OIDN",
    "HAS_PIP_OIDN",
    "denoise_cuda_hdr",
    "denoise_pip_ldr",
]
