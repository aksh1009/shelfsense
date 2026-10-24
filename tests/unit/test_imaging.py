import io

import numpy as np
import pytest
from PIL import Image

from shelfai.errors import InvalidImage
from shelfai.imaging import crop_boxes, decode_image, letterbox


def test_decode_rejects_non_images():
    with pytest.raises(InvalidImage):
        decode_image(b"definitely not an image")
    with pytest.raises(InvalidImage):
        decode_image(b"")


def test_decode_converts_to_rgb(jpeg_bytes):
    buf = io.BytesIO()
    Image.new("RGBA", (100, 80), (10, 20, 30, 128)).save(buf, format="PNG")
    img = decode_image(buf.getvalue())
    assert img.mode == "RGB" and img.size == (100, 80)


def test_decode_applies_exif_rotation():
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = img.getexif(); exif[274] = 6  # orientation: rotate 90 degrees
    buf = io.BytesIO(); img.save(buf, format="JPEG", exif=exif)
    assert decode_image(buf.getvalue()).size == (100, 200)


def test_crop_boxes_clamps_to_image():
    img = Image.new("RGB", (100, 100))
    crops = crop_boxes(img, np.array([[-10, -10, 50, 50], [90, 90, 150, 150], [10, 10, 10.2, 10.2]]))
    assert [c.size for c in crops] == [(50, 50), (10, 10), (2, 2)]


def test_letterbox_keeps_aspect():
    lb = np.asarray(letterbox(Image.new("RGB", (40, 200), (255, 0, 0)), 224))
    assert lb.shape == (224, 224, 3)
    assert lb[112, 112].tolist() == [255, 0, 0] and lb[112, 2].tolist() == [128, 128, 128]
