# ============================================================
# PaddleOCR Engine (Primary OCR)
# ============================================================

import os
from typing import List
from PIL import Image
import fitz

from app.ocr.base import OCREngine, PageResult


class PaddleOCREngine(OCREngine):
    """PaddleOCR Engine wrapper."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None
        self._initialized = False

    def _get_ocr_instance(self):
        """Lazy load PaddleOCR model on first use."""
        if not self._initialized:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang, show_log=False)
                self._initialized = True
            except ImportError:
                raise RuntimeError("PaddleOCR package is not installed.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
        return self._ocr

    def extract_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Run PaddleOCR on a PIL image."""
        ocr = self._get_ocr_instance()

        import numpy as np
        img_np = np.array(image.convert("RGB"))

        result = ocr.ocr(img_np, cls=self.use_angle_cls)

        lines: List[str] = []
        confidences: List[float] = []
        blocks: List[dict] = []

        if result and len(result) > 0 and result[0] is not None:
            for line_info in result[0]:
                box, (text, conf) = line_info
                lines.append(text)
                confidences.append(float(conf))
                blocks.append({
                    "bbox": box,
                    "text": text,
                    "confidence": float(conf),
                })

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.90
        full_text = "\n".join(lines)

        return PageResult(
            page_num=page_num,
            text=full_text,
            confidence=round(min(max(avg_conf, 0.0), 1.0), 3),
            tables=[],
            layout_blocks=blocks,
            method="paddleocr",
        )

    def extract_file(self, file_path: str) -> List[PageResult]:
        """Extract text from all pages of an image or PDF file."""
        ext = os.path.splitext(file_path)[1].lower()
        results: List[PageResult] = []

        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page_idx, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_res = self.extract_image(img, page_num=page_idx + 1)
                results.append(page_res)
            doc.close()
        else:
            with Image.open(file_path) as img:
                results.append(self.extract_image(img.convert("RGB"), page_num=1))

        return results
