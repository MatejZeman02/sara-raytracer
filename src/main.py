"""Script wrapper that runs package main (to be compatible with older branches)."""

import os
import sys


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
    _resolve_main()()
