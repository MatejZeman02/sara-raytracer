"""Settings for the raytracer."""

import sys

_DEFAULTS = {
    "DEVICE": "gpu",
    "CPU_DIMENSION": 500,
    "GPU_DIMENSION": 1024,
    "SCENE_NAME": "box-advanced",
    "SAMPLES": 16,
    "MAX_BOUNCES": 16,
    "DENOISE": True,
    # Camera exposure in standard EV stops. Multiplied as linear factor
    # (2**EV) onto raw HDR values before film-stock tonemapping.
    "EXPOSURE_COMPENSATION": 3,
    # valid modes:
    # - "custom": ACEScg -> sRGB baked 3D LUT (gpu path)
    # - "khronos": Khronos PBR neutral curve
    # - "narkowicz": per-channel Narkowicz ACES fit (CPU path)
    # - "hill": Stephen Hill ACES approximation (CPU path)
    # - "none": bypass tonemap and clip to [0, 1]
    # - "magenta": debug overshoot visualizer
    "TONEMAPPER": "custom",
    "IMG_FORMAT": "jpg",
    "USE_BVH_CACHE": False,
    "PRINT_STATS": False,
    "RENDER_NON_BVH_STATS": False,
    "COLLECT_BVH_STATS": False,
    "USE_SAH": True,
    "USE_BINNING": True,
}


class Settings:
    _instance = None
    _parsed = False

    def __init__(self):
        self._values = dict(_DEFAULTS)
        self._apply_cli_args()

    def _apply_cli_args(self):
        """Parse --key value pairs from sys.argv and override settings.

        Aliases: --scene -> SCENE_NAME, --format -> IMG_FORMAT,
        --exposure-compensation -> EXPOSURE_COMPENSATION.

        Usage: python -m src.main --scene bunny --samples 32 --tonemapper khronos
               python -m src.main --help
        """
        _ALIASES = {
            "SCENE": "SCENE_NAME",
            "FORMAT": "IMG_FORMAT",
            "RESOLUTION": "GPU_DIMENSION",
        }
        _BOOL_KEYS = {
            "USE_SAH", "USE_BINNING", "USE_BVH_CACHE", "DENOISE",
            "PRINT_STATS", "RENDER_NON_BVH_STATS", "COLLECT_BVH_STATS",
        }
        args = list(sys.argv[1:])
        for arg in args:
            if arg in ("--help", "-h", "help"):
                print("Sara ray tracer — usage:")
                print()
                print("  python -m src.main [options]")
                print("  python -m src.main --scene <name> [--samples N] [--tonemapper <name>]")
                print()
                print("Options:")
                print("  --scene <name>               Scene directory (default: box-advanced)")
                print("  --samples <N>                Samples per pixel (default: 16)")
                print("  --max-bounces <N>            Max ray bounces (default: 16)")
                print("  --resolution <N>             Output width×height (default: 1024)")
                print("  --exposure-compensation <N>  Exposure EV stops (default: 3)")
                print("  --tonemapper <name>          custom|khronos|narkowicz|hill|none|magenta")
                print("  --format <ext>               Output format: jpg|png|ppm (default: jpg)")
                print("  --device <gpu|cpu>           Render device (default: gpu)")
                print("  --denoise <true|false>       Enable OIDN denoiser (default: true)")
                print("  --help                       Show this message")
                print()
                print("Examples:")
                print("  python -m src.main --scene bunny")
                print("  python -m src.main --scene bunny --samples 64 --tonemapper khronos")
                print("  python -m src.main --scene dragon --samples 8 --denoise false")
                sys.exit(0)
        i = 0
        while i < len(args):
            if args[i].startswith("--") and i + 1 < len(args):
                key = args[i][2:].upper().replace("-", "_")
                key = _ALIASES.get(key, key)
                val = args[i + 1]
                if key in ("SAMPLES", "CPU_DIMENSION", "GPU_DIMENSION", "MAX_BOUNCES"):
                    val = int(val)
                elif key in _BOOL_KEYS:
                    val = val.lower() not in ("false", "0", "no")
                elif key == "EXPOSURE_COMPENSATION":
                    val = float(val)
                # print(f"[settings] {key} = {val}", flush=True)
                self._values[key] = val
                i += 2
            else:
                i += 1

    def _parse(self):
        self._parsed = True

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
_settings._parsed = True
settings = _settings
