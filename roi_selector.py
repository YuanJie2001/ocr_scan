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


def select_roi_polygon(image: np.ndarray) -> np.ndarray | None:
    """
    Select a polygon ROI and return Nx2 points in original image coords.
    """
    if image is None:
        return None

    return _select_roi_polygon_qt(image)


def select_roi_bbox(image: np.ndarray) -> np.ndarray | None:
    """
    Select a polygon ROI and return bbox as [x1, y1, x2, y2] in original image coords.
    """
    polygon = select_roi_polygon(image)
    if polygon is None:
        return None

    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    return np.array([x, y, x + w, y + h], dtype=float)


def _ensure_qt_app():
    global _QT_APP
    if _QT_APP is not None:
        return _QT_APP

    from PyQt6.QtWidgets import QApplication

    _QT_APP = QApplication.instance() or QApplication([])
    _QT_APP.setQuitOnLastWindowClosed(False)
    return _QT_APP


def _select_roi_polygon_qt(image: np.ndarray) -> np.ndarray | None:
    try:
        from PyQt6.QtCore import Qt, QPoint
        from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygon, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QDialog,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        raise ImportError("PyQt6 is required for manual ROI selection") from exc

    class PolygonCropWidget(QWidget):
        def __init__(self, pixmap: QPixmap, on_change):
            super().__init__()
            self._pixmap = pixmap
            self._points: list[QPoint] = []
            self._closed = False
            self._hover_pos: QPoint | None = None
            self._on_change = on_change
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.CrossCursor)

        @property
        def points(self) -> list[QPoint]:
            return self._points

        @property
        def closed(self) -> bool:
            return self._closed

        def reset(self) -> None:
            self._points = []
            self._closed = False
            self._hover_pos = None
            self._on_change()
            self.update()

        def _close_polygon(self) -> None:
            if len(self._points) >= 3 and not self._closed:
                self._closed = True
                self._on_change()
                self.update()

        def _near_first(self, pos: QPoint) -> bool:
            if not self._points:
                return False
            first = self._points[0]
            dx = pos.x() - first.x()
            dy = pos.y() - first.y()
            return dx * dx + dy * dy <= 64

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                if self._closed:
                    return
                pos = event.position().toPoint()
                if len(self._points) >= 3 and self._near_first(pos):
                    self._close_polygon()
                    return
                self._points.append(pos)
                self._on_change()
                self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                self._close_polygon()

        def mouseMoveEvent(self, event):
            if not self._closed and self._points:
                self._hover_pos = event.position().toPoint()
                self.update()

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._close_polygon()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.drawPixmap(0, 0, self._pixmap)

            if not self._points:
                return

            line_pen = QPen(QColor(0, 200, 0), 2)
            painter.setPen(line_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for i in range(len(self._points) - 1):
                painter.drawLine(self._points[i], self._points[i + 1])

            if self._closed:
                painter.drawLine(self._points[-1], self._points[0])
                painter.setBrush(QBrush(QColor(0, 200, 0, 60)))
                painter.drawPolygon(QPolygon(self._points))
            elif self._hover_pos is not None:
                painter.drawLine(self._points[-1], self._hover_pos)

            point_pen = QPen(QColor(220, 0, 0), 2)
            painter.setPen(point_pen)
            painter.setBrush(QBrush(QColor(220, 0, 0)))
            for point in self._points:
                painter.drawEllipse(point, 3, 3)

    class ROISelector(QDialog):
        def __init__(self, pixmap: QPixmap):
            super().__init__()
            self._selection: list[QPoint] | None = None

            self._confirm = QPushButton("Confirm")
            self._confirm.setEnabled(False)
            self._confirm.clicked.connect(self._confirm_selection)

            self._reset = QPushButton("Reset")
            self._reset.clicked.connect(self._reset_selection)

            self._cancel = QPushButton("Cancel")
            self._cancel.clicked.connect(self.reject)

            def _on_change():
                self._confirm.setEnabled(widget.closed and len(widget.points) >= 3)

            widget = PolygonCropWidget(pixmap, _on_change)
            widget.setFixedSize(pixmap.width(), pixmap.height())

            hint = QLabel(
                "Left click to add points. Right click or double click to close. "
                "Click Confirm to crop."
            )
            hint.setWordWrap(True)

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            buttons.addWidget(self._reset)
            buttons.addWidget(self._cancel)
            buttons.addWidget(self._confirm)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.addWidget(hint)
            layout.addWidget(widget)
            layout.addLayout(buttons)
            self.setLayout(layout)
            self.setWindowTitle("Select ID Card Polygon")

            self._widget = widget

        def _confirm_selection(self):
            if not self._widget.closed or len(self._widget.points) < 3:
                return
            self._selection = list(self._widget.points)
            self.accept()

        def _reset_selection(self):
            self._widget.reset()

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

    points = []
    for point in dlg.selection:
        x = int(point.x() / scale)
        y = int(point.y() / scale)
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        points.append([x, y])

    if len(points) < 3:
        return None

    return np.array(points, dtype=float)
