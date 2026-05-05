"""Settings for the raytracer."""

_DEFAULTS = {
    "DEVICE": "gpu",
    "CPU_DIMENSION": 500,
    "GPU_DIMENSION": 1024,
    "SCENE_NAME": "bunny",
    "SAMPLES": 16,
    "MAX_BOUNCES": 16,
    "DENOISE": True,
    # valid modes:
    # - "custom-aces": ACEScg -> sRGB baked 3D LUT (gpu path)
    # - "khronos": Khronos PBR neutral curve
    # - "aces": per-channel Narkowicz ACES fit
    # - "none": bypass tonemap and clip to [0, 1]
    # - "magenta": debug overshoot visualizer
    "TONEMAPPER": "custom-aces",
    # "TONEMAPPER": "none",
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
        global _DEFAULTS
        self._values = dict(_DEFAULTS)
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
_settings._parse()
settings = _settings
