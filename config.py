"""
Global configuration.
"""
import logging
import os

# Logging
LOG_LEVEL = os.environ.get("IDCARD_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# OCR
OCR_LANG = os.environ.get("IDCARD_OCR_LANG", "ch")
OCR_ENGINE_TYPE = os.environ.get("IDCARD_OCR_ENGINE", "onnxruntime")
OCR_MODEL_TYPE = os.environ.get("IDCARD_OCR_MODEL_TYPE", "mobile")
OCR_VERSION = os.environ.get("IDCARD_OCR_VERSION", "PP-OCRv4")
OCR_USE_ANGLE_CLS = True
OCR_DET_DB_THRESH = 0.3
OCR_DET_DB_BOX_THRESH = 0.5

# Image preprocessing
IDCARD_STANDARD_WIDTH = 1084
IDCARD_STANDARD_HEIGHT = 669

SCAN_CLAHE_CLIP = 2.0
SCAN_CLAHE_GRID = (8, 8)
SCAN_DENOISE_H = 6
SCAN_DENOISE_H_COLOR = 6
SCAN_SATURATION_FACTOR = 0.65
SCAN_CONTRAST_ALPHA = 1.05
SCAN_CONTRAST_BETA = 5

# Output
OUTPUT_DIR = os.environ.get("IDCARD_OUTPUT_DIR", "output")

# PDF
PDF_FONT_NAME = "STSong-Light"
PDF_TITLE_FONT_SIZE = 16
PDF_BODY_FONT_SIZE = 12
PDF_MARGIN = 50


def setup_logging(level: str | None = None) -> None:
    """Initialize global logging."""
    effective_level = level or LOG_LEVEL
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
