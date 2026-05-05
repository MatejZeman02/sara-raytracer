"""DEPRECATED: This module is no longer used.
All OIDN routing is now handled by utils.smart_denoiser.
This file is kept for reference only; delete if no backward compatibility needed.
"""

import warnings

warnings.warn(
    "load_oidn_lib is deprecated. Use utils.smart_denoiser instead.",
    DeprecationWarning,
    stacklevel=2,
)

from utils.smart_denoiser import (
    HAS_NATIVE_CUDA_OIDN,
    denoise_cuda_hdr_inplace as denoise_gpu,
)
