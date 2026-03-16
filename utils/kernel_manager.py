import inspect
import numpy as np
import time

from settings import DEVICE  # type: ignore

if DEVICE == "gpu":
    from numba import cuda


class KernelManager:
    def __init__(self, kernel_func):
        assert kernel_func is not None
        self.kernel = kernel_func
        self.arg_names = list(inspect.signature(kernel_func).parameters.keys())
        assert len(self.arg_names) > 0
        self.dimensions = None

    def _resolve_args(self, local_vars):
        """maps local variables and handles automatic transfers and precision enforcement."""
        args = []
        for name in self.arg_names:
            # find the variable name, handles 'd_' prefix logic
            search_name = (
                name[2:] if name.startswith("d_") and name not in local_vars else name
            )

            assert search_name in local_vars, f"missing variable: {search_name}"
            val = local_vars[search_name]

            if DEVICE == "gpu":
                # gpu path: transfer numpy arrays to device memory
                if hasattr(val, "copy_to_host"):
                    args.append(val)
                elif isinstance(val, np.ndarray):
                    if val.dtype == np.float64:
                        val = val.astype(np.float32)
                    gpu_val = cuda.to_device(val)
                    # update local scope to avoid re-uploading next time
                    local_vars[search_name] = gpu_val
                    args.append(gpu_val)
                elif isinstance(val, (float, np.float64)):
                    args.append(np.float32(val))
                else:
                    args.append(val)
            else:
                # cpu path: keep as numpy arrays, enforce float32 precision
                if isinstance(val, np.ndarray):
                    if val.dtype == np.float64:
                        val = val.astype(np.float32)
                        local_vars[search_name] = val
                    args.append(val)
                elif isinstance(val, (float, np.float64)):
                    args.append(np.float32(val))
                else:
                    args.append(val)

        assert len(args) == len(self.arg_names)
        return args

    def precompile_run(self, local_vars):
        """runs kernel to trigger jit compilation."""
        if DEVICE == "cpu":
            # direct call triggers njit compilation
            # change resolution to 1x1
            self.dimensions = (local_vars["width"], local_vars["height"])
            local_vars["width"] = 1
            local_vars["height"] = 1

        args = self._resolve_args(local_vars)
        assert len(args) == len(self.arg_names), "argument count mismatch during warmup"

        if DEVICE == "cpu":
            self.kernel(*args)
        else:
            # execute single thread with real memory references
            self.kernel[(1, 1), (1, 1)](*args)
            cuda.synchronize()

    def run(self, grid, block, data_context, measure_time=True):
        """
        calls the kernel using the correct argument order.

        note: if args previously transferred to gpu, this timing will only include kernel execution.
        """
        if DEVICE == "cpu":
            # TODO: needed?
            # ensure dimensions are set for cpu path
            assert self.dimensions is not None, "precompile_run must be called before run on CPU"
            data_context["width"], data_context["height"] = self.dimensions
        args = self._resolve_args(data_context)

        t0 = time.perf_counter()
        if DEVICE == "cpu":            
            # cpu path: direct call, grid and block are ignored
            self.kernel(*args)
        else:
            self.kernel[grid, block](*args)
            cuda.synchronize()
        if measure_time:
            return t0
