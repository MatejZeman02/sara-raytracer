from settings import DEVICE  # type: ignore


def device_jit(*args, **kwargs):
    if DEVICE == "cpu":
        from numba import njit

        # translate boolean inline to string for cpu njit
        if "inline" in kwargs:
            is_inline = kwargs.pop("inline")
            kwargs["inline"] = "always" if is_inline else "never"

        kwargs.setdefault("fastmath", True)
        kwargs.setdefault("cache", True)  # not sure if it does anything

        return njit(*args, **kwargs)
    else:
        from numba import cuda

        kwargs.setdefault("device", True)
        # defaults to inline=True for gpu if not explicitly provided
        kwargs.setdefault("inline", True)

        return cuda.jit(*args, **kwargs)
