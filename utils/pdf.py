from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.colors import Color
import os


class PDFGenerator:
    @staticmethod
    def generate_idcard_pdf(
        pdf_path, scan_image_path, ocr_result, ocr_positions, page_size=A4
    ):
        """
        生成真正电子化身份证PDF
        - 底层：扫描风格图像
        - 上层：OCR文本（透明，可搜索）
        """

        # 注册中文字体
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

        c = canvas.Canvas(pdf_path, pagesize=page_size)
        page_w, page_h = page_size

        # 1.绘制扫描图（铺满页面）
        img = ImageReader(scan_image_path)
        img_w, img_h = img.getSize()

        scale = min(page_w / img_w, page_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        x0 = (page_w - draw_w) / 2
        y0 = (page_h - draw_h) / 2

        c.drawImage(img, x0, y0, width=draw_w, height=draw_h)

        # 2. OCR 文本层（透明）
        c.setFont("STSong-Light", 10)
        c.setFillColor(Color(0, 0, 0, alpha=0))  # 关键：透明

        for text, pos in zip(ocr_result, ocr_positions):
            if pos is None or len(pos) == 0:
                continue

            # OCR 返回的是图像坐标，需要映射到 PDF
            x_min = min(p[0] for p in pos)
            y_min = min(p[1] for p in pos)

            pdf_x = x0 + x_min * scale
            pdf_y = y0 + (img_h - y_min) * scale

            # 去掉 label
            real_text = text.split(":", 1)[-1]

            c.drawString(pdf_x, pdf_y, real_text)

        c.showPage()
        c.save()
