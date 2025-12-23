"""
Local LLM PDF OCR - Package for OCR processing using local vision models.

This package provides tools for converting scanned PDFs into searchable documents
using local LLM vision models for text extraction and Surya for layout detection.
"""

__version__ = "1.0.0"

# Convenience imports for public API
from src.pdf_ocr.core.pdf import PDFHandler
from src.pdf_ocr.core.ocr import OCRProcessor
from src.pdf_ocr.core.aligner import HybridAligner

__all__ = [
    "PDFHandler",
    "OCRProcessor", 
    "HybridAligner",
    "__version__",
]
