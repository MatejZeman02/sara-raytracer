"""Script wrapper that runs package main (to be compatible with older branches)."""

import argparse
import os
import sys


def _apply_runtime_overrides(argv):
    """Apply CLI runtime overrides via environment variables before package import."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=("cpu-sequential", "cpu-parallel", "gpu"),
        help="Execution mode override.",
    )
    parser.add_argument(
        "--block-x",
        type=int,
        help="GPU block X dimension override.",
    )
    parser.add_argument(
        "--block-y",
        type=int,
        help="GPU block Y dimension override.",
    )
    args, _unknown = parser.parse_known_args(argv)

    if args.mode:
        os.environ["RT_EXECUTION_MODE"] = args.mode
    if args.block_x is not None:
        if args.block_x <= 0:
            raise ValueError(f"--block-x must be positive, got {args.block_x}")
        os.environ["RT_GPU_BLOCK_X"] = str(args.block_x)
    if args.block_y is not None:
        if args.block_y <= 0:
            raise ValueError(f"--block-y must be positive, got {args.block_y}")
        os.environ["RT_GPU_BLOCK_Y"] = str(args.block_y)


def _resolve_main():
    """Resolves the main function from the package, handling both direct and package execution contexts."""
    if __package__ in (None, ""):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src import main as package_main

        return package_main

    from . import main as package_main

    return package_main


if __name__ == "__main__":
    _apply_runtime_overrides(sys.argv[1:])
    _resolve_main()()
