"""Settings for the raytracer."""

import sys

_DEFAULTS = {
    "DEVICE": "gpu",
    "CPU_DIMENSION": 500,
    "GPU_DIMENSION": 1024,
    "SCENE_NAME": "box-scaled",
    "SAMPLES": 16,
    "MAX_BOUNCES": 16,
    "DENOISE": True,
    # Camera exposure in standard EV stops. Multiplied as linear factor
    # (2**EV) onto raw HDR values before film-stock tonemapping.
    "EXPOSURE_COMPENSATION": -2.5,
    # valid modes:
    # - "custom-aces": ACEScg -> sRGB baked 3D LUT (gpu path)
    # - "khronos": Khronos PBR neutral curve
    # - "narkowicz": per-channel Narkowicz ACES fit (CPU path)
    # - "hill": Stephen Hill ACES approximation (CPU path)
    # - "none": bypass tonemap and clip to [0, 1]
    # - "magenta": debug overshoot visualizer
    "TONEMAPPER": "custom-aces",
    "IMG_FORMAT": "jpg",
    "USE_BVH_CACHE": True,
    "PRINT_STATS": False,
    "RENDER_NON_BVH_STATS": False,
    "COLLECT_BVH_STATS": False,
    "USE_SAH": True,
    "USE_BINNING": False,
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
        """
        _ALIASES = {"SCENE": "SCENE_NAME", "FORMAT": "IMG_FORMAT"}
        args = list(sys.argv[1:])
        i = 0
        while i < len(args):
            if args[i].startswith("--") and i + 1 < len(args):
                key = args[i][2:].upper().replace("-", "_")
                key = _ALIASES.get(key, key)
                val = args[i + 1]
                if key in ("SAMPLES", "CPU_DIMENSION", "GPU_DIMENSION", "MAX_BOUNCES"):
                    val = int(val)
                elif key in (
                    "USE_SAH",
                    "USE_BINNING",
                    "USE_BVH_CACHE",
                    "DENOISE",
                    "PRINT_STATS",
                    "RENDER_NON_BVH_STATS",
                    "COLLECT_BVH_STATS",
                ):
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
