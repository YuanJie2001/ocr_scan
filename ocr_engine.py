"""
OCR 识别引擎模块

封装 RapidOCR，提供统一的文本识别接口。
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)


def preload_onnxruntime() -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pass


@dataclass
class OCRResult:
    """单条 OCR 识别结果"""
    text: str
    confidence: float
    position: list[list[float]] = field(default_factory=list)


class OCREngine:
    """
    RapidOCR 引擎封装

    懒加载 RapidOCR 实例，提供统一的识别结果格式。
    """

    def __init__(
        self,
        lang: str | None = None,
        use_angle_cls: bool | None = None,
    ):
        """
        初始化 OCR 引擎

        Args:
            lang: 识别语言，默认 'ch'
            use_angle_cls: 是否启用方向分类器
        """
        self._lang = lang or config.OCR_LANG
        self._use_angle_cls = use_angle_cls if use_angle_cls is not None else config.OCR_USE_ANGLE_CLS
        self._engine = None

    def _load_engine(self) -> None:
        """懒加载 RapidOCR 引擎"""
        if self._engine is not None:
            return

        try:
            from rapidocr import (
                EngineType,
                LangCls,
                LangDet,
                LangRec,
                ModelType,
                OCRVersion,
                RapidOCR,
            )

            ocr_version = self._enum_value(OCRVersion, config.OCR_VERSION)
            params = {
                "Det.engine_type": self._enum_value(EngineType, config.OCR_ENGINE_TYPE),
                "Det.lang_type": self._enum_value(LangDet, self._lang),
                "Det.model_type": self._enum_value(ModelType, config.OCR_MODEL_TYPE),
                "Det.ocr_version": ocr_version,
                "Det.thresh": config.OCR_DET_DB_THRESH,
                "Det.box_thresh": config.OCR_DET_DB_BOX_THRESH,
                "Cls.engine_type": self._enum_value(EngineType, config.OCR_ENGINE_TYPE),
                "Cls.lang_type": self._enum_value(LangCls, self._lang),
                "Cls.model_type": self._enum_value(ModelType, config.OCR_MODEL_TYPE),
                "Cls.ocr_version": ocr_version,
                "Rec.engine_type": self._enum_value(EngineType, config.OCR_ENGINE_TYPE),
                "Rec.lang_type": self._enum_value(LangRec, self._lang),
                "Rec.model_type": self._enum_value(ModelType, config.OCR_MODEL_TYPE),
                "Rec.ocr_version": ocr_version,
            }
            self._engine = RapidOCR(params=params)
            logger.info(
                "RapidOCR 引擎初始化成功 (lang=%s, engine=%s, model=%s, version=%s)",
                self._lang,
                config.OCR_ENGINE_TYPE,
                config.OCR_MODEL_TYPE,
                config.OCR_VERSION,
            )
        except ModuleNotFoundError:
            logger.error("rapidocr 或 onnxruntime 未安装，请运行: pip install rapidocr onnxruntime")
            raise
        except ImportError as e:
            logger.error("RapidOCR 依赖加载失败: %s", e)
            raise
        except Exception as e:
            logger.error("RapidOCR 初始化失败: %s", e)
            raise

    @staticmethod
    def _enum_value(enum_cls: type[Enum], value: str) -> Enum:
        normalized = value.replace("-", "_").upper()
        if normalized in enum_cls.__members__:
            return enum_cls[normalized]
        for member in enum_cls:
            if str(member.value).lower() == value.lower():
                return member
        raise ValueError(f"Unsupported RapidOCR parameter {enum_cls.__name__}: {value}")

    @staticmethod
    def _to_position(box: Any) -> list[list[float]]:
        if box is None:
            return []
        return np.asarray(box, dtype=float).tolist()

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
            result = self._engine(
                image,
                use_det=True,
                use_cls=self._use_angle_cls,
                use_rec=True,
            )
            boxes = getattr(result, "boxes", None)
            texts = getattr(result, "txts", None)
            scores = getattr(result, "scores", None)
            if boxes is None or texts is None or scores is None:
                return []

            ocr_results = []
            for box, text, score in zip(boxes, texts, scores):
                if text and str(text).strip():
                    ocr_results.append(
                        OCRResult(
                            text=str(text).strip(),
                            confidence=float(score),
                            position=self._to_position(box),
                        )
                    )

            logger.info("OCR 识别完成，共 %d 条结果", len(ocr_results))
            for i, r in enumerate(ocr_results):
                logger.debug("  [%d] %.2f%% | %s", i, r.confidence * 100, r.text)

            return ocr_results

        except Exception as e:
            logger.error("OCR 识别过程出错: %s", e)
            raise
