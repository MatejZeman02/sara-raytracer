"""Utilities for rendering and device management."""

from numba import njit, cuda


def _get_settings():
    """Deferred import of settings to avoid circular imports."""
    from src.settings import settings

    return settings


def device_jit(*args, **kwargs):
    """JIT compilation that routes to CPU or GPU based on settings."""
    settings = _get_settings()
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
