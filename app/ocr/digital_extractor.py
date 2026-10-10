# ============================================================
# Digital PDF & Table Extractor (born-digital PDFs)
# ============================================================
# Skips OCR when a digital text layer is present in PDFs, extracting
# text and structured tables with 100% fidelity and fast speed.
# ============================================================

import os
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
import pdfplumber

from app.ocr.base import PageResult


class DigitalPDFExtractor:
    """Extracts text and tables directly from born-digital PDFs."""

    @staticmethod
    def is_born_digital(file_path: str, min_chars_per_page: int = 30) -> bool:
        """Check if a PDF file has an embedded digital text layer.

        Returns True if the average character count per page exceeds ``min_chars_per_page``.
        """
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            if page_count == 0:
                doc.close()
                return False

            total_chars = 0
            for page in doc:
                text = page.get_text().strip()
                total_chars += len(text)
            doc.close()

            avg_chars = total_chars / max(page_count, 1)
            return avg_chars >= min_chars_per_page
        except Exception:
            return False

    @staticmethod
    def extract(file_path: str) -> List[PageResult]:
        """Extract text, tables, and block layouts from a born-digital PDF."""
        results: List[PageResult] = []

        # 1. Extract text and layout blocks using PyMuPDF (fast)
        doc = fitz.open(file_path)
        page_texts: List[str] = []
        page_blocks: List[List[dict]] = []

        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            page_texts.append(text)

            blocks = []
            for b in page.get_text("blocks"):
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 5 and b[4].strip():
                    blocks.append({
                        "bbox": [b[0], b[1], b[2], b[3]],
                        "text": b[4].strip(),
                        "type": b[6] if len(b) > 6 else 0,
                    })
            page_blocks.append(blocks)
        doc.close()

        # 2. Extract structured tables using pdfplumber
        page_tables: List[List[List[Optional[str]]]] = [[] for _ in range(len(page_texts))]
        try:
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    if idx < len(page_tables):
                        extracted_tables = page.extract_tables()
                        if extracted_tables:
                            page_tables[idx] = extracted_tables
        except Exception:
            pass  # Fall back to text if table extraction encounters an issue

        # 3. Assemble PageResults
        for idx in range(len(page_texts)):
            results.append(
                PageResult(
                    page_num=idx + 1,
                    text=page_texts[idx],
                    confidence=1.0,
                    tables=page_tables[idx],
                    layout_blocks=page_blocks[idx],
                    method="digital",
                )
            )

        return results
