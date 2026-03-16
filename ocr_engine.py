"""
OCR 识别引擎模块

封装 PaddleOCR，提供统一的文本识别接口。
"""
import logging
from dataclasses import dataclass, field

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """单条 OCR 识别结果"""
    text: str                           # 识别文本
    confidence: float                   # 置信度
    position: list[list[float]] = field(default_factory=list)  # 四个角点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class OCREngine:
    """
    PaddleOCR 引擎封装 (兼容 PaddleOCR 2.x)

    懒加载 PaddleOCR 实例，提供统一的识别结果格式。
    """

    def __init__(
        self,
        lang: str | None = None,
        use_gpu: bool | None = None,
        use_angle_cls: bool | None = None,
    ):
        """
        初始化 OCR 引擎

        Args:
            lang: 识别语言，默认 'ch'
            use_gpu: 是否使用 GPU
            use_angle_cls: 是否启用方向分类器
        """
        self._lang = lang or config.OCR_LANG
        self._use_gpu = use_gpu if use_gpu is not None else config.OCR_USE_GPU
        self._use_angle_cls = use_angle_cls if use_angle_cls is not None else config.OCR_USE_ANGLE_CLS
        self._engine = None

    def _load_engine(self) -> None:
        """懒加载 PaddleOCR 引擎"""
        if self._engine is not None:
            return

        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_angle_cls=self._use_angle_cls,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info(
                "PaddleOCR 引擎初始化成功 (lang=%s, gpu=%s)",
                self._lang,
                self._use_gpu,
            )
        except ImportError:
            logger.error("paddleocr 未安装，请运行: pip install paddleocr")
            raise
        except Exception as e:
            logger.error("PaddleOCR 初始化失败: %s", e)
            raise

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """
        对图像进行 OCR 识别

        Args:
            image: BGR 格式的输入图像 (numpy array)

        Returns:
            识别结果列表
        """
        self._load_engine()

        try:
            raw_results = self._engine.ocr(image, cls=self._use_angle_cls)

            ocr_results = []

            if raw_results is None:
                logger.warning("OCR 返回空结果")
                return ocr_results

            # PaddleOCR 返回格式: [[[position], (text, confidence)], ...]
            for page_result in raw_results:
                if page_result is None:
                    continue
                for line in page_result:
                    if line is None or len(line) < 2:
                        continue

                    position = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    text, confidence = line[1]

                    if text and text.strip():
                        ocr_results.append(OCRResult(
                            text=text.strip(),
                            confidence=float(confidence),
                            position=position,
                        ))

            logger.info("OCR 识别完成，共 %d 条结果", len(ocr_results))
            for i, r in enumerate(ocr_results):
                logger.debug("  [%d] %.2f%% | %s", i, r.confidence * 100, r.text)

            return ocr_results

        except Exception as e:
            logger.error("OCR 识别过程出错: %s", e)
            raise
