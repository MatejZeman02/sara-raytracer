"""legacy entrypoint; prefer utils.smart_denoiser."""

from utils.smart_denoiser import (
    HAS_NATIVE_CUDA_OIDN,
    denoise_cuda_hdr_inplace as denoise_gpu,
)
