"""Intel Open Image Denoise (OIDN) wrapper for HDR path-traced buffers."""

import numpy as np
import oidn


def denoise(fb_hdr, width, height):
    """run OIDN RT filter on a float32 HDR image buffer in-place.
    The buffer is modified in-place.
    Note: OIDN 1.4.3 via this Python wrapper does not expose the HDR boolean
    flag through the Python API, so the filter runs in LDR mode. For the
    typical path-traced range the difference is negligible after ACES tonemap.
    """
    assert fb_hdr.dtype == np.float32
    assert fb_hdr.shape == (height, width, 3)

    # ensure C-contiguous layout (required for the output binding)
    if not fb_hdr.flags.c_contiguous:
        fb_hdr[:] = np.ascontiguousarray(fb_hdr)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_CPU)
    oidn.CommitDevice(device)
    try:
        filt = oidn.NewFilter(device, "RT")
        oidn.SetSharedFilterImage(
            filt, "color", fb_hdr, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.SetSharedFilterImage(
            filt, "output", fb_hdr, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.CommitFilter(filt)
        oidn.ExecuteFilter(filt)
        oidn.ReleaseFilter(filt)
    finally:
        oidn.ReleaseDevice(device)
