# ============================================================
# OCR Package Exports
# ============================================================

from app.ocr.base import PageResult, OCREngine
from app.ocr.digital_extractor import DigitalPDFExtractor
from app.ocr.paddleocr_engine import PaddleOCREngine
from app.ocr.tesseract_engine import TesseractOCREngine
from app.ocr.pipeline import OCRPipeline, ocr_pipeline

__all__ = [
    "PageResult",
    "OCREngine",
    "DigitalPDFExtractor",
    "PaddleOCREngine",
    "TesseractOCREngine",
    "OCRPipeline",
    "ocr_pipeline",
]
