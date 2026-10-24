"""Image utilities: decode bytes (with EXIF rotation), encode JPEG, crop and letterbox."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidImage

Image.MAX_IMAGE_PIXELS = 80_000_000  # guard against decompression bombs (~80 MP)
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "MPO"}


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> RGB PIL image, rotated according to EXIF (phones store orientation there)."""
    if not data:
        raise InvalidImage("The uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImage("The file could not be read as an image. Upload a JPEG, PNG or WebP photo.",
                           {"reason": str(e)}) from e
    if fmt and fmt not in SUPPORTED_FORMATS:
        raise InvalidImage(f"Unsupported image format '{fmt}'. Upload a JPEG, PNG or WebP photo.")
    if min(img.size) < 32:
        raise InvalidImage(f"The image is too small ({img.width}x{img.height}).")
    return img


def encode_jpeg(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_bgr_array(img: Image.Image) -> np.ndarray:
    """RGB PIL -> BGR numpy (the convention ultralytics uses for numpy input)."""
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def crop_boxes(img: Image.Image, boxes: np.ndarray) -> list[Image.Image]:
    """Crop every box from the ORIGINAL full-resolution image (at least 2x2 px)."""
    W, H = img.size
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        l, t = max(0, int(x1)), max(0, int(y1))
        r, b = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
        r, b = max(r, min(W, l + 2)), max(b, min(H, t + 2))
        crops.append(img.crop((l, t, r, b)))
    return crops


def letterbox(im: Image.Image, size: int, fill=(128, 128, 128)) -> Image.Image:
    """Resize keeping aspect ratio and pad to a square: no stretching, no centre-crop (keeps logos)."""
    s = size / max(im.size)
    nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
    out = Image.new("RGB", (size, size), fill)
    out.paste(im.resize((nw, nh), Image.BICUBIC), ((size - nw) // 2, (size - nh) // 2))
    return out
