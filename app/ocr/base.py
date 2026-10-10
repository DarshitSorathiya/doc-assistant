# ============================================================
# OCR Base Engine & Data Structures
# ============================================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from PIL import Image


class PageResult(BaseModel):
    """Result of text and table extraction for a single page."""

    page_num: int = Field(..., description="1-indexed page number")
    text: str = Field(default="", description="Extracted text on this page")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    tables: List[List[List[Optional[str]]]] = Field(default_factory=list, description="Extracted 2D tables")
    layout_blocks: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted text blocks/bounding boxes")
    method: str = Field(default="digital", description="Extraction method used: digital | paddleocr | tesseract")


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Extract text and confidence from a PIL Image."""
        pass

    @abstractmethod
    def extract_file(self, file_path: str) -> List[PageResult]:
        """Extract text and confidence across all pages of a document or image file."""
        pass
