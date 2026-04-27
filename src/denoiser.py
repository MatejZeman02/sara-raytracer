"""Intel Open Image Denoise (OIDN) wrapper for SDR path-traced buffers."""

import warnings

import numpy as np

try:
    import oidn
    if not (hasattr(oidn, "NewDevice") and hasattr(oidn, "DEVICE_TYPE_CPU")):
        raise ImportError("OIDN module lacks expected API")
except (ImportError, AttributeError):
    oidn = None
    HAS_OIDN = False
else:
    HAS_OIDN = True


def denoise(fb_sdr, width, height):
    """run OIDN RT filter on a float32 SDR image buffer in-place.
    The buffer is modified in-place.
    Note: OIDN 1.4.3 via this Python wrapper does not expose the HDR boolean
    flag through the Python API, so the filter runs in SDR mode. For the
    typical path-traced range the difference is negligible after ACES tonemap.
    """
    assert fb_sdr.dtype == np.float32
    assert fb_sdr.shape == (height, width, 3)

    if not HAS_OIDN:
        warnings.warn(
            "OIDN is not installed; skipping denoising.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # C-contiguous layout (required for the output binding)
    if not fb_sdr.flags.c_contiguous:
        fb_sdr[:] = np.ascontiguousarray(fb_sdr)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_CPU)
    oidn.CommitDevice(device)
    try:
        filt = oidn.NewFilter(device, "RT")
        oidn.SetSharedFilterImage(
            filt, "color", fb_sdr, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.SetSharedFilterImage(
            filt, "output", fb_sdr, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.CommitFilter(filt)
        oidn.ExecuteFilter(filt)
        oidn.ReleaseFilter(filt)
    finally:
        oidn.ReleaseDevice(device)
