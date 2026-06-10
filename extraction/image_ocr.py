"""
extraction/image_ocr.py
OCR pipeline using Tesseract (via pytesseract) with an optional pre-processing
step (deskew + denoise) powered by OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from loguru import logger
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# ── public API ────────────────────────────────────────────────────────────────

def ocr_image(
    image: Image.Image | str | Path,
    *,
    lang: str = "eng+hin",
    preprocess: bool = True,
) -> str:
    """
    Run OCR on a single image.

    Args:
        image:      PIL Image, file path, or Path object.
        lang:       Tesseract language code(s), e.g. ``"eng"`` or ``"eng+fra"``.
        preprocess: Apply deskew + denoise before OCR (recommended for scans).

    Returns:
        Extracted text string (may be empty if nothing readable found).
    """
    pil_img = _load_image(image)

    if preprocess:
        pil_img = _preprocess(pil_img)

    custom_config = r'--oem 3 --psm 6'
    try:
        text: str = pytesseract.image_to_string(pil_img, lang=lang, config=custom_config)
        text = text.strip()
        logger.debug(f"OCR returned {len(text)} characters")
        return text
    
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not found in PATH")
        return ""
    
    except pytesseract.TesseractError as exc:
        logger.error(f"Tesseract error: {exc}")
        return ""


def ocr_images(
    images: list[Image.Image | str | Path],
    **kwargs,
) -> list[str]:
    """Batch OCR — returns a list of text strings in the same order as *images*."""
    results = []
    for i, img in enumerate(images):
        logger.debug(f"OCR image {i + 1}/{len(images)}")
        results.append(ocr_image(img, **kwargs))
    return results


def ocr_image_file(path: str | Path, **kwargs) -> str:
    """Convenience wrapper — reads an image from disk and runs OCR."""
    return ocr_image(Path(path), **kwargs)


# ── internals ─────────────────────────────────────────────────────────────────

def _load_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _preprocess(pil_img: Image.Image) -> Image.Image:
    """Denoise and deskew a PIL image using OpenCV."""
    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

     # ── upscale if image is small ──
    h, w = gray.shape
    if w < 1000:
        scale = 2.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # ── denoise ──
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # ── CLAHE contrast enhancement ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)


    # ── adaptive threshold (handles uneven lighting) ──
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )

    # ── deskew ──
    binary = _deskew(binary)

    # Convert back to PIL RGB (pytesseract works fine with grayscale too)
    return Image.fromarray(binary).convert("RGB")


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the image to correct skew detected via Hough lines."""
    coords = np.column_stack(np.where(gray < 128))  # foreground pixels
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); normalise to [-45, 45)
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:          # skip tiny corrections
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug(f"Deskew applied: {angle:.2f}°")
    return rotated
