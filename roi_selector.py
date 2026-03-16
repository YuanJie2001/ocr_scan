"""
Manual ROI selector for ID card cropping.

Requires PyQt6 (no OpenCV fallback).
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_QT_APP = None


def select_roi_bbox(image: np.ndarray) -> np.ndarray | None:
    """
    Select a ROI and return bbox as [x1, y1, x2, y2] in original image coords.
    """
    if image is None:
        return None

    return _select_roi_qt(image)


def _ensure_qt_app():
    global _QT_APP
    if _QT_APP is not None:
        return _QT_APP

    from PyQt6.QtWidgets import QApplication

    _QT_APP = QApplication.instance() or QApplication([])
    _QT_APP.setQuitOnLastWindowClosed(False)
    return _QT_APP


def _select_roi_qt(image: np.ndarray) -> np.ndarray | None:
    try:
        from PyQt6.QtCore import Qt, QRect, QSize
        from PyQt6.QtGui import QImage, QPainter, QPixmap
        from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QRubberBand, QWidget
    except Exception as exc:
        raise ImportError("PyQt6 is required for manual ROI selection") from exc

    class ImageCropWidget(QWidget):
        def __init__(self, pixmap: QPixmap, done_callback):
            super().__init__()
            self._pixmap = pixmap
            self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._origin = None
            self._done_callback = done_callback

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._pixmap)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin = event.position().toPoint()
                self._rubber.setGeometry(QRect(self._origin, QSize()))
                self._rubber.show()

        def mouseMoveEvent(self, event):
            if self._origin is not None:
                self._rubber.setGeometry(
                    QRect(self._origin, event.position().toPoint()).normalized()
                )

        def mouseReleaseEvent(self, event):
            if self._origin is not None:
                rect = self._rubber.geometry()
                self._rubber.hide()
                self._done_callback(rect)

    class ROISelector(QDialog):
        def __init__(self, pixmap: QPixmap):
            super().__init__()
            self._selection = None

            def _done(rect: QRect):
                self._selection = rect
                self.accept()

            widget = ImageCropWidget(pixmap, _done)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            self.setLayout(layout)
            self.setWindowTitle("Select ID Card Region")
            self.setFixedSize(pixmap.width(), pixmap.height())

        @property
        def selection(self):
            return self._selection

    app = _ensure_qt_app()
    platform = app.platformName() if hasattr(app, "platformName") else ""
    if platform in {"offscreen", "minimal", "headless"}:
        raise RuntimeError(f"Qt platform is '{platform}', cannot show window")

    img_h, img_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, img_w, img_h, rgb.strides[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    screen = app.primaryScreen()
    if screen:
        geom = screen.availableGeometry()
        max_w = int(geom.width() * 0.9)
        max_h = int(geom.height() * 0.9)
    else:
        max_w, max_h = img_w, img_h

    scale = min(max_w / img_w, max_h / img_h, 1.0)
    if scale != 1.0:
        pixmap = pixmap.scaled(
            int(img_w * scale),
            int(img_h * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    dlg = ROISelector(pixmap)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    app.processEvents()
    if dlg.exec() != QDialog.DialogCode.Accepted or dlg.selection is None:
        return None

    rect = dlg.selection
    x = int(rect.x() / scale)
    y = int(rect.y() / scale)
    w = int(rect.width() / scale)
    h = int(rect.height() / scale)

    if w <= 2 or h <= 2:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return np.array([x1, y1, x2, y2], dtype=float)
