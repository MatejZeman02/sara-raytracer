import inspect
import numpy as np
from numba import cuda
import time


class KernelManager:
    def __init__(self, kernel_func):
        self.kernel = kernel_func
        self.arg_names = list(inspect.signature(kernel_func).parameters.keys())

    def _resolve_args(self, local_vars):
        """maps local variables and handles automatic gpu transfers and precision enforcement."""
        gpu_args = []
        for name in self.arg_names:
            # find the variable name (handles 'd_' prefix logic)
            search_name = (
                name[2:] if name.startswith("d_") and name not in local_vars else name
            )

            assert search_name in local_vars, f"missing variable: {search_name}"
            val = local_vars[search_name]

            # check if it's already a numba device array
            if isinstance(val, cuda.devicearray.DeviceNDArray):
                gpu_args.append(val)

            # handle numpy arrays (convert float64 to float32)
            elif isinstance(val, np.ndarray):
                if val.dtype == np.float64:
                    val = val.astype(np.float32)
                gpu_val = cuda.to_device(val)
                # update local scope to avoid re-uploading next time
                local_vars[search_name] = gpu_val
                gpu_args.append(gpu_val)

            # handle scalars (convert python floats to float32)
            elif isinstance(val, (float, np.float64)):
                gpu_val = np.float32(val)
                # don't need to update local_vars for scalars but for consistency
                gpu_args.append(gpu_val)
            else:  # no change
                gpu_args.append(val)

        return gpu_args

    def precompile_run(self, local_vars):
        """runs kernel on 1x1 grid using real data types to trigger compilation."""
        args = self._resolve_args(local_vars)
        assert len(args) == len(self.arg_names), "argument count mismatch during warmup"

        # execute single thread with real memory references
        self.kernel[(1, 1), (1, 1)](*args)
        cuda.synchronize()

    def run(self, grid, block, data_context, measure_time=True):
        """
        calls the kernel using the correct argument order.

        **note:** if args previously transferred to gpu, this timing will only include kernel execution!
        """
        # build the argument list based on the kernel signature
        args = self._resolve_args(data_context)

        t0 = time.perf_counter()
        self.kernel[grid, block](*args)
        cuda.synchronize()
        if measure_time:
            return t0
