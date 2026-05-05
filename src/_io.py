"""Image I/O utilities."""

from PIL import Image
from .settings import settings


def save_image(fb, output_path: str) -> None:
    """Save uint8 host framebuffer to disk."""
    assert fb is not None
    host_fb = fb.copy_to_host() if hasattr(fb, "copy_to_host") else fb
    img = Image.fromarray(host_fb)
    img.save(f"{output_path}.{settings.IMG_FORMAT}")
    print(f"Click to see the result onto: {output_path}.{settings.IMG_FORMAT}")
