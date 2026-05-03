"""Settings for the raytracer."""

import argparse

_DEFAULTS = {
    "DEVICE": "gpu",
    "CPU_DIMENSION": 500,
    "GPU_DIMENSION": 1024,
    "SCENE_NAME": "bunny",
    "SAMPLES": 16,
    "MAX_BOUNCES": 16,
    "DENOISE": True,
    "TONEMAPPER": "khronos",
    "IMG_FORMAT": "jpg",
    "USE_BVH_CACHE": True,
    "PRINT_STATS": False,
    "RENDER_NON_BVH_STATS": False,
    "COLLECT_BVH_STATS": False,
    "USE_SAH": True,
    "USE_BINNING": True,
}


def _get(args, name, default=False):
    """Safely get an argparse namespace attribute with a default."""
    val = getattr(args, name, None)
    if val is None:
        return default
    return val


class Settings:
    """Central settings object. CLI args override defaults on first access."""

    _instance = None
    _parsed = False

    def __init__(self):
        self._values = dict(_DEFAULTS)

    def _parse(self):
        if self._parsed:
            return
        self._parsed = True

        parser = argparse.ArgumentParser(description="CUDA Raytracer Settings")
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["gpu", "cpu"],
            help="Render device (gpu / cpu)",
        )
        parser.add_argument(
            "--dimension",
            type=int,
            default=None,
            help="Resolution dimension (overrides CPU_DIMENSION / GPU_DIMENSION)",
        )
        parser.add_argument(
            "--scene",
            type=str,
            default=None,
            help="Scene name (directory under scenes/)",
        )
        parser.add_argument(
            "--samples", type=int, default=None, help="Samples per pixel"
        )
        parser.add_argument("--bounces", type=int, default=None, help="Max ray bounces")
        parser.add_argument(
            "--denoise", action="store_true", default=None, help="Enable OIDN denoising"
        )
        parser.add_argument(
            "--no-denoise",
            action="store_true",
            default=None,
            help="Disable OIDN denoising",
        )
        parser.add_argument(
            "--tonemapper",
            type=str,
            default=None,
            choices=["khronos", "magenta", "none"],
            help="SDR tonemapping operator",
        )
        parser.add_argument(
            "--format",
            type=str,
            default=None,
            choices=["jpg", "png"],
            help="Output image format",
        )
        parser.add_argument(
            "--no-bvh-cache",
            action="store_true",
            default=None,
            help="Disable BVH cache loading",
        )
        parser.add_argument(
            "--print-stats",
            action="store_true",
            default=None,
            help="Print rendering statistics",
        )
        parser.add_argument(
            "--no-print-stats",
            action="store_true",
            default=None,
            help="Suppress rendering statistics",
        )
        parser.add_argument(
            "--render-non-bvh-stats",
            action="store_true",
            default=None,
            help="Run render stats without BVH",
        )
        parser.add_argument(
            "--collect-bvh-stats",
            action="store_true",
            default=None,
            help="Collect BVH traversal metrics",
        )
        parser.add_argument(
            "--use-sah",
            action="store_true",
            default=None,
            help="Use SAH for BVH construction",
        )
        parser.add_argument(
            "--no-sah",
            action="store_true",
            default=None,
            help="Disable SAH for BVH construction",
        )
        parser.add_argument(
            "--use-binning",
            action="store_true",
            default=None,
            help="Use binning for BVH construction",
        )
        parser.add_argument(
            "--no-binning",
            action="store_true",
            default=None,
            help="Disable binning for BVH construction",
        )

        try:
            args = parser.parse_args()
        except SystemExit:
            args = argparse.Namespace()

        if getattr(args, "device", None) is not None:
            self._values["DEVICE"] = args.device
        if getattr(args, "dimension", None) is not None:
            self._values["CPU_DIMENSION"] = args.dimension
            self._values["GPU_DIMENSION"] = args.dimension
        if getattr(args, "scene", None) is not None:
            self._values["SCENE_NAME"] = args.scene
        if getattr(args, "samples", None) is not None:
            self._values["SAMPLES"] = args.samples
        if getattr(args, "bounces", None) is not None:
            self._values["MAX_BOUNCES"] = args.bounces
        if _get(args, "no_denoise"):
            self._values["DENOISE"] = False
        elif _get(args, "denoise"):
            self._values["DENOISE"] = True
        if getattr(args, "tonemapper", None) is not None:
            self._values["TONEMAPPER"] = args.tonemapper
        if getattr(args, "format", None) is not None:
            self._values["IMG_FORMAT"] = args.format
        if _get(args, "no_bvh_cache"):
            self._values["USE_BVH_CACHE"] = False
        if _get(args, "print_stats"):
            self._values["PRINT_STATS"] = True
        elif _get(args, "no_print_stats"):
            self._values["PRINT_STATS"] = False
        if _get(args, "render_non_bvh_stats"):
            self._values["RENDER_NON_BVH_STATS"] = True
        if _get(args, "collect_bvh_stats"):
            self._values["COLLECT_BVH_STATS"] = True
        if _get(args, "use_sah"):
            self._values["USE_SAH"] = True
        elif _get(args, "no_sah"):
            self._values["USE_SAH"] = False
        if _get(args, "use_binning"):
            self._values["USE_BINNING"] = True
        elif _get(args, "no_binning"):
            self._values["USE_BINNING"] = False

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        self._parse()
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._parse()
            self._values[name] = value


_settings = Settings()
_settings._parse()
settings = _settings
