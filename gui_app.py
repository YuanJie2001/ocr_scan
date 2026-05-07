from __future__ import annotations

import logging
import os
import sys

from ocr_engine import OCREngine, preload_onnxruntime

preload_onnxruntime()

import cv2
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import config
from image_scan_extractor import ImageScanExtractor
from parser import IDCardParser
from pdf_generator import PDFGenerator
from roi_selector import select_roi_polygon

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGES = 2
THUMBNAIL_SIZE = 160


def _to_pixmap(image, size: int = THUMBNAIL_SIZE) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    if size:
        pixmap = pixmap.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap


class CropApp(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.scanner = ImageScanExtractor()
        self._ocr: OCREngine | None = None
        self.parser = IDCardParser()

        self.import_btn = QPushButton("导入图片")
        self.remove_btn = QPushButton("移除")
        self.crop_btn = QPushButton("裁剪")
        self.export_btn = QPushButton("导出PDF")

        self.method_combo = QComboBox()
        self.method_combo.addItems(["自动截取", "手动裁剪"])

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.setSpacing(6)

        self.status = QLabel("最多可添加 2 张图片")
        self.status.setStyleSheet("color: #666;")

        controls = QHBoxLayout()
        controls.addWidget(self.import_btn)
        controls.addWidget(self.remove_btn)
        controls.addStretch(1)
        controls.addWidget(QLabel("裁剪方式:"))
        controls.addWidget(self.method_combo)
        controls.addWidget(self.crop_btn)
        controls.addWidget(self.export_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.status)

        self.setWindowTitle("证件裁剪")
        self.resize(720, 520)

        self.import_btn.clicked.connect(self.import_images)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.crop_btn.clicked.connect(self.crop_selected)
        self.export_btn.clicked.connect(self.export_pdf)
        self.list_widget.itemDoubleClicked.connect(self.crop_selected)

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _existing_paths(self) -> set[str]:
        paths: set[str] = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.ItemDataRole.UserRole) or {}
            path = data.get("path")
            if path:
                paths.add(path)
        return paths

    def import_images(self) -> None:
        if self.list_widget.count() >= MAX_IMAGES:
            self._set_status("最多只能添加 2 张图片")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not paths:
            return

        existing = self._existing_paths()
        slots = MAX_IMAGES - self.list_widget.count()
        added = 0
        skipped = 0

        for path in paths:
            if added >= slots:
                skipped += 1
                continue
            if path in existing:
                skipped += 1
                continue
            if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
                skipped += 1
                continue
            image = self.scanner.load_image(path)
            if image is None:
                skipped += 1
                continue

            item = QListWidgetItem(os.path.basename(path))
            item.setIcon(QIcon(_to_pixmap(image)))
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "path": path,
                    "image": image,
                },
            )
            self.list_widget.addItem(item)
            added += 1

        if added == 0:
            self._set_status("未添加图片")
        elif skipped:
            self._set_status(f"已导入 {added} 张，已忽略 {skipped} 张")
        else:
            self._set_status(f"已导入 {added} 张")

    def remove_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            self._set_status("请选择要移除的图片")
            return
        row = self.list_widget.row(item)
        self.list_widget.takeItem(row)
        self._set_status("已移除图片")

    def crop_selected(self, item: QListWidgetItem | None = None) -> None:
        item = item or self.list_widget.currentItem()
        if item is None:
            self._set_status("请选择要裁剪的图片")
            return

        data = item.data(Qt.ItemDataRole.UserRole) or {}
        image = data.get("image")
        if image is None:
            self._set_status("图片数据无效")
            return

        method = self.method_combo.currentIndex()
        if method == 1:
            polygon = select_roi_polygon(image)
            if polygon is None:
                self._set_status("已取消裁剪")
                return
            processed, _ = self.scanner.extract(
                image, roi=polygon, show_process=False, scan_style=False
            )
        else:
            processed, _ = self.scanner.extract(
                image, roi=None, show_process=False, scan_style=False
            )

        if processed is None:
            self._set_status("裁剪失败")
            return

        data["image"] = processed
        item.setData(Qt.ItemDataRole.UserRole, data)
        item.setIcon(QIcon(_to_pixmap(processed)))
        self._set_status("裁剪完成，列表已更新")

    def _ensure_ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = OCREngine()
        return self._ocr

    def export_pdf(self) -> None:
        total = self.list_widget.count()
        if total == 0:
            self._set_status("列表为空")
            return

        default_dir = os.path.abspath(config.OUTPUT_DIR)
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, "cards.pdf")
        pdf_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出PDF",
            default_path,
            "PDF (*.pdf)",
        )
        if not pdf_path:
            return
        if not pdf_path.lower().endswith(".pdf"):
            pdf_path += ".pdf"

        self._set_status("正在导出PDF，请稍候...")

        items: list[dict] = []
        failed = 0
        pdf_dir = os.path.dirname(pdf_path) or default_dir
        ocr = None
        ocr_failed = False

        for i in range(total):
            item = self.list_widget.item(i)
            data = item.data(Qt.ItemDataRole.UserRole) or {}
            image = data.get("image")
            if image is None:
                failed += 1
                continue

            base_name = os.path.splitext(os.path.basename(data.get("path") or f"card_{i+1}"))[0]
            card_image_path = os.path.join(pdf_dir, f"{base_name}_{i+1}_card.jpg")
            if not cv2.imwrite(card_image_path, image):
                failed += 1
                continue

            ocr_results = []
            id_info = None
            if not ocr_failed:
                try:
                    ocr = ocr or self._ensure_ocr()
                    ocr_results = ocr.recognize(image)
                    id_info = self.parser.parse(ocr_results) if ocr_results else None
                except Exception as exc:
                    logger.error("OCR failed: %s", exc)
                    ocr_failed = True
                    ocr_results = []
                    id_info = None

            items.append(
                {
                    "card_image_path": card_image_path,
                    "id_info": id_info,
                    "ocr_results": ocr_results,
                }
            )

        if not items:
            self._set_status("导出失败：没有可用图片")
            return

        try:
            PDFGenerator.generate_batch(
                pdf_path=pdf_path,
                items=items,
            )
        except Exception as exc:
            logger.error("PDF export failed: %s", exc)
            self._set_status("导出失败")
            return

        if failed:
            self._set_status(f"已导出PDF: {pdf_path}（成功 {len(items)} 张，失败 {failed} 张）")
        else:
            self._set_status(f"已导出PDF: {pdf_path}")


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = CropApp()
    window.show()
    sys.exit(app.exec())
