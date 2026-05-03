from src.settings import settings  # type: ignore
from numba import njit
from numba import cuda


def device_jit(*args, **kwargs):
    if settings.DEVICE == "cpu":

        # translate boolean inline/device to string for cpu njit
        if "inline" in kwargs:
            is_inline = kwargs.pop("inline")
            kwargs["inline"] = "always" if is_inline else "never"
        if "device" in kwargs:
            is_device = kwargs.pop("device")
            kwargs["device"] = "always" if is_device else "never"

        kwargs.setdefault("fastmath", True)
        # kwargs.setdefault("cache", True)  # not sure if it does anything

        return njit(*args, **kwargs)
    else:
        # kwargs.setdefault("lineinfo", True) # debug info for gpu kernels
        kwargs.setdefault("device", True)
        kwargs.setdefault("fastmath", False)
        # defaults to inline=True for gpu if not explicitly provided
        kwargs.setdefault("inline", True)

        return cuda.jit(*args, **kwargs)
