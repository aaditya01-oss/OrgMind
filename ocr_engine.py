"""
Real OCR for scanned documents (Tesseract, English + Nepali).

Needs on the machine (all free):
  1. the Tesseract program, with the 'eng' and 'nep' language files
  2. pip install pytesseract pymupdf pillow
Nothing here stores anything. It only turns a file into text.
"""
import logging
import os
import shutil

from PIL import Image, ImageOps
import pytesseract

try:
    import pymupdf            # newer name
except ImportError:           # older installs
    import fitz as pymupdf

# Where is the Tesseract program? An explicit TESSERACT_CMD setting wins; otherwise, if it is not on the
# PATH, try the normal Windows install folder.
_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
elif not shutil.which("tesseract") and os.path.exists(_WINDOWS_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_DEFAULT

WANTED_LANGS = ("eng", "nep")
MAX_PAGES = 10                 # a scan job must never run for minutes
RENDER_DPI = 200               # good accuracy without huge images
MAX_PIXELS = 40_000_000        # refuse "decompression bombs" (tiny file, enormous picture)
PAGE_TIMEOUT_SECONDS = 60
UPSCALE_BELOW = 1500           # photos smaller than this (longest side) are enlarged 2x: small print reads better
MIN_TEXT_LAYER_CHARS = 20      # fewer characters than this = the page is really a picture

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class OcrError(ValueError):
    """A problem whose message is safe to show to the user."""


def _languages() -> str:
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # Tesseract program missing or broken
        logging.error("Tesseract is not usable: %s", exc)
        raise OcrError("Scanning is not available on this server right now.")
    langs = [l for l in WANTED_LANGS if l in installed]
    if not langs:
        raise OcrError("Scanning is not available on this server right now.")
    if "nep" not in langs:
        logging.warning("Tesseract Nepali data ('nep') is not installed: Nepali text will be read poorly.")
    return "+".join(langs)


def _ocr_picture(img: Image.Image, langs: str) -> str:
    img = ImageOps.exif_transpose(img)          # phone photos are often rotated
    if img.width * img.height > MAX_PIXELS:
        raise OcrError("This image is too large to scan. Please use a smaller photo or scan.")
    img = ImageOps.autocontrast(img.convert("L"))
    if max(img.size) < UPSCALE_BELOW and img.width * img.height * 4 <= MAX_PIXELS:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    try:
        return pytesseract.image_to_string(img, lang=langs, timeout=PAGE_TIMEOUT_SECONDS).strip()
    except RuntimeError:                         # pytesseract raises RuntimeError on timeout
        raise OcrError("Scanning this page took too long. Please try a smaller or clearer file.")


def _scan_pdf(path: str, langs: str) -> dict:
    try:
        doc = pymupdf.open(path)
    except Exception:
        raise OcrError("This PDF could not be opened. It may be damaged.")
    with doc:
        if doc.needs_pass:
            raise OcrError("This PDF is password protected. Please remove the password and try again.")
        total = doc.page_count
        texts, ocr_pages = [], 0
        for index in range(min(total, MAX_PAGES)):
            page = doc[index]
            layer = page.get_text().strip()
            if len(layer) >= MIN_TEXT_LAYER_CHARS:      # a real text page: no OCR needed
                texts.append(layer)
                continue
            zoom = RENDER_DPI / 72
            while page.rect.width * page.rect.height * zoom * zoom > MAX_PIXELS and zoom > 0.5:
                zoom *= 0.75                             # huge page: render smaller
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            texts.append(_ocr_picture(img, langs))
            ocr_pages += 1
    return {
        "text": "\n\n".join(t for t in texts if t),
        "pages_total": total,
        "pages_read": min(total, MAX_PAGES),
        "pages_ocr": ocr_pages,
    }


def ocr_file(path: str) -> dict:
    """Read a PDF or an image. Returns text plus facts about how it was read."""
    langs = _languages()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        result = _scan_pdf(path, langs)
    else:
        try:
            with Image.open(path) as img:
                img.load()
                text = _ocr_picture(img, langs)
        except OcrError:
            raise
        except Exception:
            raise OcrError("This image could not be read. Please use a PNG or JPG file.")
        result = {"text": text, "pages_total": 1, "pages_read": 1, "pages_ocr": 1}
    result["truncated"] = result["pages_total"] > result["pages_read"]
    result["languages"] = langs
    return result