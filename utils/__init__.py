"""Utilities for rendering and device management."""

from src.settings import settings  # type: ignore
from numba import njit
from numba import cuda


def device_jit(*args, **kwargs):
    """JIT compilation that routes to CPU or GPU based on settings."""
    if settings.DEVICE == "cpu":
        # Translate boolean inline/device to string for cpu njit
        if "inline" in kwargs:
            is_inline = kwargs.pop("inline")
            kwargs["inline"] = "always" if is_inline else "never"
        if "device" in kwargs:
            is_device = kwargs.pop("device")
            kwargs["device"] = "always" if is_device else "never"

        kwargs.setdefault("fastmath", True)
        return njit(*args, **kwargs)
    else:
        kwargs.setdefault("device", True)
        kwargs.setdefault("fastmath", False)
        kwargs.setdefault("inline", True)
        return cuda.jit(*args, **kwargs)


__all__ = ["device_jit"]
