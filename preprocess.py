"""
图像预处理模块

提供身份证图像的裁剪、透视矫正、旋转校正、扫描件风格转换等功能。
内置通用图像处理方法，减少外部依赖。
"""
import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class IDCardPreprocessor:
    """
    身份证图像预处理器

    处理流程: 裁剪 → 透视矫正 → 旋转校正 → 尺寸标准化 → (可选)扫描件风格
    """
    @staticmethod
    def show(image: np.ndarray, window_name: str = "default") -> None:
        """显示图像（调试用）"""
        if image is None:
            logger.warning("图像为空，无法显示")
            return
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def gray_image(image: np.ndarray) -> np.ndarray:
        """BGR → Gray"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def filter_gray(gray: np.ndarray) -> np.ndarray:
        """多阶滤波，抑制噪声并增强边缘"""
        median = cv2.medianBlur(gray, ksize=5)
        bilateral = cv2.bilateralFilter(median, d=9, sigmaColor=25, sigmaSpace=50)
        gaussian = cv2.GaussianBlur(bilateral, (3, 3), 0)
        return gaussian

    @staticmethod
    def binary_filter(blur: np.ndarray) -> np.ndarray:
        """自适应阈值二值化"""
        return cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

    @staticmethod
    def edge_binary(binary: np.ndarray) -> np.ndarray | None:
        """边缘检测 + 形态学膨胀，返回最大外轮廓"""
        edges = cv2.Canny(binary, 100, 150, 3)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=5)

        contours_info = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def convert_to_scan_style_v2(
        self,
        image: np.ndarray,
        out_path: str | None = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        将图像转换为扫描件风格（对比度、降噪、色彩保持）
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=config.SCAN_CLAHE_CLIP, tileGridSize=config.SCAN_CLAHE_GRID
        )
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        denoise = cv2.fastNlMeansDenoisingColored(
            balanced,
            None,
            h=config.SCAN_DENOISE_H,
            hColor=config.SCAN_DENOISE_H_COLOR,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        hsv = cv2.cvtColor(denoise, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = (s * config.SCAN_SATURATION_FACTOR).astype(np.uint8)
        scan_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        scan_img = cv2.convertScaleAbs(
            scan_img,
            alpha=config.SCAN_CONTRAST_ALPHA,
            beta=config.SCAN_CONTRAST_BETA,
        )

        if out_path:
            cv2.imwrite(out_path, scan_img)

        if show:
            self.show(scan_img, "scan_style")

        return scan_img

    def process_image(self, image: np.ndarray, show_process: bool = False) -> np.ndarray | None:
        """
        完整的图像预处理流水线（不含裁剪，裁剪由 preprocess 方法处理）

        Args:
            image: 输入图像 (BGR)
            show_process: 是否显示处理过程

        Returns:
            处理后的标准化图像，失败返回 None
        """
        try:
            corrected = self.perspective_correct(image, show_process)
            if corrected is None:
                logger.warning("透视矫正失败，使用原图进行后续处理")
                corrected = image

            # 旋转校正（确保宽 > 高）
            corrected = self.rotate_correct(corrected)

            # 标准化尺寸
            resized = cv2.resize(
                corrected,
                (config.IDCARD_STANDARD_WIDTH, config.IDCARD_STANDARD_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            if show_process:
                self.show(resized, "preprocessed")

            return resized

        except Exception as e:
            logger.error("图像预处理失败: %s", e)
            return None

    def process_result(self, result):
        """预处理模块不处理识别结果，直接返回"""
        return result

    def crop(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        根据检测框裁剪图像

        Args:
            image: 输入图像 (BGR)
            bbox: [x1, y1, x2, y2] 格式的边界框

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # 添加 5% 的边距，避免裁剪过紧
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        cropped = image[y1:y2, x1:x2]
        logger.debug("裁剪区域: (%d,%d)-(%d,%d)，图像尺寸: %dx%d", x1, y1, x2, y2, cropped.shape[1], cropped.shape[0])
        return cropped

    def crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Polygon crop: mask outside area and return the bounded region.

        Args:
            image: Input image (BGR)
            polygon: Nx2 polygon points

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            logger.warning("Polygon ROI format invalid, skip crop")
            return image

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = pts.astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_i], 255)

        if image.ndim == 2:
            masked = np.full(image.shape, 255, dtype=image.dtype)
            masked[mask > 0] = image[mask > 0]
        else:
            masked = np.full_like(image, 255)
            masked[mask > 0] = image[mask > 0]

        x, y, bw, bh = cv2.boundingRect(pts_i)
        if bw <= 2 or bh <= 2:
            logger.warning("Polygon ROI bounding box invalid, skip crop")
            return masked

        margin_x = int(bw * 0.03)
        margin_y = int(bh * 0.03)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + bw + margin_x)
        y2 = min(h, y + bh + margin_y)

        cropped = masked[y1:y2, x1:x2]
        logger.debug(
            "Polygon crop area: (%d,%d)-(%d,%d), size: %dx%d",
            x1,
            y1,
            x2,
            y2,
            cropped.shape[1],
            cropped.shape[0],
        )
        return cropped

    def perspective_correct(self, image: np.ndarray, show_process: bool = False) -> np.ndarray | None:
        """
        透视矫正：检测身份证轮廓并进行透视变换

        Args:
            image: 输入图像 (BGR)
            show_process: 是否显示中间过程

        Returns:
            透视矫正后的图像，失败返回 None
        """
        try:
            # 灰度 → 滤波 → CLAHE → 二值化 → 边缘检测
            gray = self.gray_image(image)
            blur = self.filter_gray(gray)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blur)

            binary = self.binary_filter(enhanced)

            if show_process:
                self.show(binary, "binary")

            contour = self.edge_binary(binary)
            if contour is None:
                logger.warning("未检测到有效轮廓，跳过透视矫正")
                return None

            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                logger.warning(
                    "轮廓不是四边形 (检测到 %d 个顶点)，使用最小外接矩形回退",
                    len(approx),
                )
                rect = cv2.minAreaRect(contour)
                points = cv2.boxPoints(rect)
            else:
                points = approx.reshape(4, 2)

            if len(points) != 4:
                logger.warning("无法得到有效四边形顶点，跳过透视矫正")
                return None

            # 排序顶点: 左上 → 右上 → 右下 → 左下
            rect = np.zeros((4, 2), dtype="float32")
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]  # 左上
            rect[2] = points[np.argmax(s)]  # 右下
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]  # 右上
            rect[3] = points[np.argmax(diff)]  # 左下

            # 计算目标尺寸
            tl, tr, br, bl = rect
            max_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

            if max_width <= 0 or max_height <= 0:
                logger.warning("计算得到的目标尺寸无效: %dx%d", max_width, max_height)
                return None

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (max_width, max_height))

            if show_process:
                self.show(warped, "perspective_corrected")

            logger.debug("透视矫正完成: %dx%d", max_width, max_height)
            return warped

        except Exception as e:
            logger.error("透视矫正过程出错: %s", e)
            return None

    def rotate_correct(self, image: np.ndarray) -> np.ndarray:
        """
        旋转校正：确保身份证水平放置（宽 > 高）

        Args:
            image: 输入图像 (BGR)

        Returns:
            旋转校正后的图像
        """
        h, w = image.shape[:2]
        if w < h:
            image = np.rot90(image)
            logger.debug("图像已旋转 90°（原 %dx%d → %dx%d）", w, h, image.shape[1], image.shape[0])
        return image

    def to_scan_style(self, image: np.ndarray, output_path: str | None = None) -> np.ndarray:
        """
        将图像转换为扫描件风格

        Args:
            image: 输入图像 (BGR)
            output_path: 可选的输出路径

        Returns:
            扫描件风格图像
        """
        return self.convert_to_scan_style_v2(image, out_path=output_path)

    def preprocess(
        self,
        image: np.ndarray,
        bbox: np.ndarray | None = None,
        show_process: bool = False,
        scan_style: bool = True,
        scan_output_path: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        完整预处理流水线

        Args:
            image: 输入图像 (BGR)
            bbox: 可选的检测框 [x1, y1, x2, y2]
            show_process: 是否显示处理过程
            scan_style: 是否转换为扫描件风格
            scan_output_path: 扫描件输出路径

        Returns:
            (预处理后的图像, 扫描件风格图像) 元组
        """
        # 1. 裁剪（如有检测框）
        if bbox is not None:
            roi = np.asarray(bbox)
            if roi.ndim == 1 and roi.shape[0] == 4:
                image = self.crop(image, roi)
            elif roi.ndim == 2 and roi.shape[1] == 2 and roi.shape[0] >= 3:
                image = self.crop_polygon(image, roi)
            else:
                logger.warning("ROI format invalid, skip crop")

        # 2. 透视矫正 + 旋转校正 + 标准化
        processed = self.process_image(image, show_process)
        if processed is None:
            logger.warning("预处理失败，尝试使用裁剪后的原图")
            processed = self.rotate_correct(image)
            processed = cv2.resize(
                processed,
                (config.IDCARD_STANDARD_WIDTH, config.IDCARD_STANDARD_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        # 3. 扫描件风格（可选）
        scan_image = None
        if scan_style:
            scan_image = self.to_scan_style(processed, scan_output_path)

        return processed, scan_image


