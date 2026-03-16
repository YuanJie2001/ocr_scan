"""
PDF generator for ID card output.

Generates a PDF with the cropped card image and optional OCR text layer.
"""
import logging
import os

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

import config
from parser import IDCardInfo

logger = logging.getLogger(__name__)

_FONT_REGISTERED = False


def _ensure_font():
    global _FONT_REGISTERED
    if not _FONT_REGISTERED:
        pdfmetrics.registerFont(UnicodeCIDFont(config.PDF_FONT_NAME))
        _FONT_REGISTERED = True


class PDFGenerator:
    """ID card PDF generator."""

    @staticmethod
    def generate(
        pdf_path: str,
        card_image_path: str | None = None,
        id_info: IDCardInfo | None = None,
        ocr_results: list | None = None,
        page_size: tuple = A4,
    ) -> str:
        """
        Generate ID card PDF.

        Args:
            pdf_path: output PDF path
            card_image_path: cropped card image path
            id_info: parsed ID card info
            ocr_results: OCR results for transparent text layer
            page_size: PDF page size

        Returns:
            output PDF path
        """
        _ensure_font()

        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

        c = canvas.Canvas(pdf_path, pagesize=page_size)
        page_w, page_h = page_size

        # Page 1: card image + transparent OCR text layer
        if card_image_path and os.path.exists(card_image_path):
            PDFGenerator._draw_card_page(c, card_image_path, ocr_results, page_w, page_h)
        elif card_image_path:
            logger.warning("Card image not found: %s, skipping image page", card_image_path)

        # Page 2: structured summary
        if id_info:
            PDFGenerator._draw_info_page(c, id_info, page_w, page_h)

        c.save()
        logger.info("PDF generated: %s", pdf_path)
        return pdf_path

    @staticmethod
    def generate_batch(
        pdf_path: str,
        items: list[dict],
        page_size: tuple = A4,
    ) -> str:
        """
        Generate a single PDF for multiple ID cards.

        Each item should contain:
            - card_image_path: str
            - id_info: IDCardInfo | None
            - ocr_results: list | None
        """
        _ensure_font()

        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

        c = canvas.Canvas(pdf_path, pagesize=page_size)
        page_w, page_h = page_size

        for item in items:
            card_image_path = item.get("card_image_path")
            id_info = item.get("id_info")
            ocr_results = item.get("ocr_results")

            if card_image_path and os.path.exists(card_image_path):
                PDFGenerator._draw_card_page(
                    c, card_image_path, ocr_results, page_w, page_h
                )
            elif card_image_path:
                logger.warning("Card image not found: %s, skipping image page", card_image_path)

            if id_info:
                PDFGenerator._draw_info_page(c, id_info, page_w, page_h)

        c.save()
        logger.info("PDF generated: %s", pdf_path)
        return pdf_path

    @staticmethod
    def _draw_card_page(
        c: canvas.Canvas,
        image_path: str,
        ocr_results: list | None,
        page_w: float,
        page_h: float,
    ) -> None:
        """Draw card image page."""
        img = ImageReader(image_path)
        img_w, img_h = img.getSize()

        # Fit to page with margins
        scale = min(page_w / img_w, page_h / img_h) * 0.9
        draw_w = img_w * scale
        draw_h = img_h * scale
        x0 = (page_w - draw_w) / 2
        y0 = (page_h - draw_h) / 2

        c.drawImage(img, x0, y0, width=draw_w, height=draw_h)

        # Transparent OCR text layer
        if ocr_results:
            c.setFont(config.PDF_FONT_NAME, 10)
            c.setFillColor(Color(0, 0, 0, alpha=0))

            for result in ocr_results:
                if not result.position or len(result.position) < 2:
                    continue

                x_min = min(p[0] for p in result.position)
                y_min = min(p[1] for p in result.position)

                pdf_x = x0 + x_min * scale
                pdf_y = y0 + (img_h - y_min) * scale

                c.drawString(pdf_x, pdf_y, result.text)

        c.showPage()

    @staticmethod
    def _draw_info_page(
        c: canvas.Canvas,
        id_info: IDCardInfo,
        page_w: float,
        page_h: float,
    ) -> None:
        """Draw structured info page."""
        margin = config.PDF_MARGIN

        c.setFont(config.PDF_FONT_NAME, config.PDF_TITLE_FONT_SIZE)
        c.setFillColor(HexColor("#1a1a2e"))
        title = "身份证电子化信息"
        title_width = c.stringWidth(title, config.PDF_FONT_NAME, config.PDF_TITLE_FONT_SIZE)
        c.drawString((page_w - title_width) / 2, page_h - margin, title)

        y = page_h - margin - 20
        c.setStrokeColor(HexColor("#e0e0e0"))
        c.setLineWidth(1)
        c.line(margin, y, page_w - margin, y)

        y -= 40
        fields = id_info.to_dict()
        c.setFont(config.PDF_FONT_NAME, config.PDF_BODY_FONT_SIZE)

        for label, value in fields.items():
            c.setFillColor(HexColor("#666666"))
            c.drawString(margin, y, f"{label}：")

            c.setFillColor(HexColor("#1a1a2e"))
            c.drawString(margin + 120, y, value)
            y -= 30

        y -= 20
        c.setStrokeColor(HexColor("#e0e0e0"))
        c.line(margin, y, page_w - margin, y)
        y -= 30

        if id_info.id_number:
            from validator import IDCardValidator

            is_valid, error_msg = IDCardValidator.validate(id_info.id_number)
            c.setFont(config.PDF_FONT_NAME, config.PDF_BODY_FONT_SIZE)
            c.setFillColor(HexColor("#666666"))
            c.drawString(margin, y, "号码校验：")

            if is_valid:
                c.setFillColor(HexColor("#27ae60"))
                c.drawString(margin + 120, y, "✓ 校验通过")
            else:
                c.setFillColor(HexColor("#e74c3c"))
                c.drawString(margin + 120, y, f"✗ {error_msg}")

        c.setFont(config.PDF_FONT_NAME, 8)
        c.setFillColor(HexColor("#999999"))
        footer = "本文件由身份证 OCR 系统自动生成"
        footer_width = c.stringWidth(footer, config.PDF_FONT_NAME, 8)
        c.drawString((page_w - footer_width) / 2, margin - 20, footer)

        c.showPage()
