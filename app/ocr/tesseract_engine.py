# ============================================================
# Tesseract OCR Engine
# ============================================================

import os
import shutil
from typing import List
from PIL import Image
import pytesseract
import fitz

from app.ocr.base import OCREngine, PageResult


class TesseractOCREngine(OCREngine):
    """Tesseract OCR engine implementation."""

    def __init__(self):
        self.is_available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        """Check if tesseract binary is installed and on system PATH."""
        # Check standard PATH
        if shutil.which("tesseract") is not None:
            return True
        # Check common Windows install paths
        win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in win_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                return True
        return False

    def extract_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Run Tesseract OCR on a PIL image."""
        if not self.is_available:
            raise RuntimeError("Tesseract is not installed or not found on PATH.")

        # Extract text
        text = pytesseract.image_to_string(image).strip()

        # Compute average confidence from word-level data
        confidence = 0.85  # default estimate
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if int(c) >= 0]
            if confs:
                confidence = sum(confs) / (len(confs) * 100.0)
        except Exception:
            pass

        return PageResult(
            page_num=page_num,
            text=text,
            confidence=round(min(max(confidence, 0.0), 1.0), 3),
            tables=[],
            layout_blocks=[],
            method="tesseract",
        )

    def extract_file(self, file_path: str) -> List[PageResult]:
        """Extract text from an image or rasterized PDF pages using Tesseract."""
        if not self.is_available:
            raise RuntimeError("Tesseract is not installed or not found on PATH.")

        ext = os.path.splitext(file_path)[1].lower()
        results: List[PageResult] = []

        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page_idx, page in enumerate(doc):
                # Render PDF page to image pixmap (200 DPI for OCR clarity)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_res = self.extract_image(img, page_num=page_idx + 1)
                results.append(page_res)
            doc.close()
        else:
            with Image.open(file_path) as img:
                results.append(self.extract_image(img.convert("RGB"), page_num=1))

        return results
