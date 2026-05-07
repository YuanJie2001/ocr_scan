from __future__ import annotations

import logging

import cv2
import numpy as np

from preprocess import IDCardPreprocessor

logger = logging.getLogger(__name__)


class ImageScanExtractor:
    def __init__(self, preprocessor: IDCardPreprocessor | None = None) -> None:
        self._preprocessor = preprocessor or IDCardPreprocessor()

    def load_image(self, image_path: str) -> np.ndarray | None:
        image = cv2.imread(image_path)
        if image is None:
            logger.error("无法加载图片: %s", image_path)
            return None
        logger.info("图像加载完成: %dx%d", image.shape[1], image.shape[0])
        return image

    def extract(
        self,
        image: np.ndarray,
        roi: np.ndarray | None = None,
        show_process: bool = False,
        scan_style: bool = False,
        scan_output_path: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._preprocessor.preprocess(
            image,
            bbox=roi,
            show_process=show_process,
            scan_style=scan_style,
            scan_output_path=scan_output_path,
        )
