"""Framebuffer and statistics buffer management."""

import numpy as np
from .settings import settings

if settings.DEVICE == "gpu":
    from numba import cuda


def allocate_buffers(width: int, height: int) -> tuple:
    """Allocate HDR framebuffer and statistics arrays.

    Returns:
        (fb_hdr, out_stats)
    """
    assert width > 0 and height > 0
    STATS_TUPLE = (height, width, 9)
    FB_HDR_SHAPE = (height, width, 3)  # float32 HDR framebuffer

    if settings.DEVICE == "gpu":
        fb_hdr = cuda.device_array(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = cuda.device_array(STATS_TUPLE, dtype=np.int32)
    else:
        fb_hdr = np.zeros(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = np.zeros(STATS_TUPLE, dtype=np.int32)

    return fb_hdr, out_stats
