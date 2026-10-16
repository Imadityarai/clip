"""
preprocess.py
Robust image ingestion, synthetic generation, typographic overlay rendering,
and visual inpainting/masking defense.
"""

import os
from io import BytesIO
from typing import Optional, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_image(image_input, max_size: int = 768) -> Optional[Image.Image]:
    """
    Loads and normalizes an image from:
      - PIL.Image.Image
      - Streamlit UploadedFile / BytesIO / file-like buffer
      - raw bytes
      - local file path string
      - remote HTTP/HTTPS URL
    """
    if image_input is None:
        return None

    img: Optional[Image.Image] = None

    # Case 1: Already a PIL Image
    if isinstance(image_input, Image.Image):
        img = image_input

    # Case 2: Streamlit UploadedFile or file-like buffer
    elif hasattr(image_input, "read"):
        try:
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            img = Image.open(image_input)
        except Exception as e:
            print(f"[ERROR] Failed to load image from file buffer: {e}")
            return None

    # Case 3: Raw bytes
    elif isinstance(image_input, bytes):
        try:
            img = Image.open(BytesIO(image_input))
        except Exception as e:
            print(f"[ERROR] Failed to load image from raw bytes: {e}")
            return None

    # Case 4: File path or URL string
    elif isinstance(image_input, str):
        if image_input.startswith(("http://", "https://")):
            try:
                resp = requests.get(image_input, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
            except Exception as e:
                print(f"[ERROR] Failed to fetch image from URL: {e}")
                return None
        elif os.path.exists(image_input):
            try:
                img = Image.open(image_input)
            except Exception as e:
                print(f"[ERROR] Failed to open local file path: {e}")
                return None
        else:
            print(f"[WARN] File path does not exist: {image_input}")
            return None

    if img is None:
        return None

    # Correct phone/camera orientation using EXIF tags
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Convert to standard RGB (handles RGBA, grayscale, CMYK)
    img = img.convert("RGB")

    # Resize large images to preserve memory and keep font scaling proportional
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return img


def generate_synthetic_apple() -> Image.Image:
    """Generates a clean synthetic apple image fallback."""
    img = Image.new("RGB", (384, 384), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)

    # Stem and leaf
    draw.rectangle([186, 60, 198, 120], fill=(101, 67, 33))
    draw.polygon([(198, 90), (240, 70), (220, 105)], fill=(46, 139, 87))

    # Apple body lobes
    draw.ellipse([80, 100, 240, 310], fill=(220, 30, 40))
    draw.ellipse([144, 100, 304, 310], fill=(210, 25, 35))

    # Specular reflection
    draw.ellipse([115, 130, 150, 190], fill=(250, 120, 120))
    return img


def get_font(size: int):
    """Loads a TrueType font cross-platform with fallback to default bitmap font."""
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "arial.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def overlay_text_on_image(
    image: Optional[Image.Image],
    text: str,
    font_size: int,
    position: Tuple[int, int] = (40, 40),
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Renders typographic text onto an image and returns the altered image
    along with its bounding box coordinates (x0, y0, x1, y1).
    """
    if image is None:
        image = generate_synthetic_apple()

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    font = get_font(font_size)

    bbox = draw.textbbox(position, text, font=font)
    draw.text(position, text, fill=color, font=font)
    return img_copy, bbox


def mask_typographic_region(
    image: Optional[Image.Image],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 8,
    fill_color: Tuple[int, int, int] = (230, 230, 230),
) -> Image.Image:
    """
    Visual Defense 1: Masks the detected typographic region with neutral fill.
    """
    if image is None:
        image = generate_synthetic_apple()

    defended_img = image.copy()
    draw = ImageDraw.Draw(defended_img)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(defended_img.width, x1 + padding)
        y1 = min(defended_img.height, y1 + padding)
        draw.rectangle([x0, y0, x1, y1], fill=fill_color)
    else:
        # Fallback upper-quadrant masking
        draw.rectangle([20, 20, 280, 120], fill=fill_color)

    return defended_img