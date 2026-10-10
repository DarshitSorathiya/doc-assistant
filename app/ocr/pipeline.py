# ============================================================
# Unified OCR & Text Extraction Pipeline
# ============================================================
# Coordinates born-digital PDF extraction, primary OCR (PaddleOCR),
# and fallback OCR (Tesseract) with table and layout parsing.
# ============================================================

import os
import logging
from typing import List, Tuple, Optional
from PIL import Image

from app.config.settings import get_settings
from app.ocr.base import PageResult, OCREngine
from app.ocr.digital_extractor import DigitalPDFExtractor
from app.ocr.paddleocr_engine import PaddleOCREngine
from app.ocr.tesseract_engine import TesseractOCREngine

logger = logging.getLogger(__name__)


class OCRPipeline:
    """Unified document text and table extraction pipeline."""

    def __init__(self, preferred_engine: Optional[str] = None):
        settings = get_settings()
        self.preferred_engine_name = preferred_engine or settings.OCR_ENGINE
        self._paddle_engine = PaddleOCREngine()
        self._tesseract_engine = TesseractOCREngine()

    def _get_ocr_engine_order(self) -> List[Tuple[str, OCREngine]]:
        """Return engines in priority order based on configuration."""
        if self.preferred_engine_name.lower() == "tesseract":
            return [("tesseract", self._tesseract_engine), ("paddleocr", self._paddle_engine)]
        return [("paddleocr", self._paddle_engine), ("tesseract", self._tesseract_engine)]

    def extract_document(self, file_path: str) -> List[PageResult]:
        """Extract text and tables from a PDF or image file.

        1. If PDF is born-digital, directly extracts digital text & tables (fast & accurate).
        2. If scanned PDF or image, runs configured OCR engine with automatic fallback.

        Returns:
            List of PageResult items (one per page).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Step 1: Check if PDF has native digital text layer
        if ext == ".pdf":
            if DigitalPDFExtractor.is_born_digital(file_path):
                logger.info(f"Extracting native digital text layer for {file_path}")
                return DigitalPDFExtractor.extract(file_path)
            logger.info(f"Scanned PDF detected without digital text layer: {file_path}. Running OCR.")

        # Step 2: Run OCR with automatic fallback
        engines = self._get_ocr_engine_order()
        errors = []

        for name, engine in engines:
            try:
                logger.info(f"Attempting OCR extraction using {name} on {file_path}")
                results = engine.extract_file(file_path)
                if results and any(r.text.strip() for r in results):
                    return results
            except Exception as e:
                logger.warning(f"OCR engine '{name}' failed on {file_path}: {e}")
                errors.append(f"{name}: {str(e)}")

        # Step 3: Ultimate fallback if OCR engines failed or are missing
        logger.warning(f"All OCR engines failed ({errors}). Attempting basic PyMuPDF text/metadata scrape.")
        if ext == ".pdf":
            try:
                return DigitalPDFExtractor.extract(file_path)
            except Exception:
                pass

        # Return minimal fallback placeholder page if everything failed
        return [
            PageResult(
                page_num=1,
                text="",
                confidence=0.0,
                tables=[],
                layout_blocks=[],
                method="failed",
            )
        ]

    def process_file(self, file_path: str) -> Tuple[str, int, float, List[PageResult]]:
        """Convenience method returning aggregated text, page count, and confidence.

        Returns:
            Tuple of (full_text, page_count, avg_confidence, page_results).
        """
        page_results = self.extract_document(file_path)
        page_count = len(page_results)

        texts = [f"--- Page {p.page_num} ---\n{p.text}" for p in page_results if p.text.strip()]
        full_text = "\n\n".join(texts)

        confs = [p.confidence for p in page_results if p.confidence is not None]
        avg_confidence = round(sum(confs) / len(confs), 3) if confs else 0.0

        return full_text, page_count, avg_confidence, page_results


# Global pipeline instance
ocr_pipeline = OCRPipeline()
