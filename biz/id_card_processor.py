import cv2
import numpy as np
from cnocr import CnOcr
from utils import validate_id_card, extract_info_from_id_card
from base import ImageABC

class IDCardProcessor(ImageABC):
    """
    身份证图像处理器，继承自ImageABC基类
    
    该类实现了特定于身份证识别的图像处理和结果处理逻辑
    """
    
    def __init__(self, model_name='scene-densenet_lite_136-gru'):
        """
        初始化身份证图像处理器
        
        参数:
        model_name: 使用的OCR模型名称
        """
        super().__init__()
        self._ocr_instance = None
        self.model_name = model_name
    
    def init_ocr(self):
        """
        初始化OCR模型
        """
        if self._ocr_instance is None:
            self._ocr_instance = CnOcr(self.model_name)
    
    def process_image(self, image, show_process=False):
        """
        处理身份证图像，包括灰度处理、滤波、对比度增强、二值化、边缘检测、透视变换等
        
        参数:
        image: 输入图像
        show_process: 是否显示处理过程
        
        返回:
        处理后的图像
        """
        # 1.灰度处理
        gray = self.gray_image(image)
        if show_process:
            self.show(gray, "gray")
            
        # 2.滤波
        blur = self.filter_gray(gray)
        if show_process:
            self.show(blur, "blur")
            
        # 3.CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blur)
        if show_process:
            self.show(enhanced, "enhanced")

        # 4.二值化
        binary = self.binary_filter(enhanced)
        if show_process:
            self.show(binary, "binary")
            
        # 5.边缘检测并膨胀
        dilation = self.edge_binary(binary)
        if show_process:
            self.show(dilation, "dilation")
            
        # 6.轮廓检测
        contour = self.find_contours(dilation)
        if show_process:
            image_copy = image.copy()
            self.show(cv2.drawContours(image_copy, contour, -1, (255, 0, 0), 20), "contour")
            
        # 7.透视变换
        w, h, perspective = self._perspective_image(image, contour)
        if perspective is None:
            print("透视变换失败,无法继续处理.")
            return None
        if show_process:
            self.show(perspective, "perspective")
            
        # 8.固定位置和大小
        resized = self._fixed_perspective(w, h, perspective)
        if show_process:
            self.show(resized, "resized")
        
        return resized
    
    def process_result(self, result):
        """
        处理OCR识别结果，转换为字典格式并验证身份证号码
        
        参数:
        result: OCR识别结果列表
        
        返回:
        处理后的结果字典
        """
        # 将结果转换为字典格式
        result_dict = {}
        for item in result:
            key, value = item.split(':', 1)
            result_dict[key] = value
        
        # 验证身份证号码
        if '公民身份号码' in result_dict and result_dict['公民身份号码'] != '未识别' and result_dict['公民身份号码'] != '识别错误':
            id_number = result_dict['公民身份号码']
            # 清理可能的空格和特殊字符
            id_number = ''.join(c for c in id_number if c.isdigit() or c.upper() == 'X')
            
            # 验证身份证号码
            is_valid, error_msg = validate_id_card(id_number)
            result_dict['身份证号码验证'] = '有效' if is_valid else f'无效: {error_msg}'
            
            # 如果有效，提取更多信息
            if is_valid:
                extra_info = extract_info_from_id_card(id_number)
                # 添加额外信息到结果中
                for key, value in extra_info.items():
                    if key not in result_dict:
                        result_dict[key] = value
            
            # 更新身份证号码为清理后的值
            result_dict['公民身份号码'] = id_number
        
        return result_dict
    
    def process_id_card(self, img_path, out_path=None, show_process=False, convert_to_scan=False):
        """
        处理身份证图像并进行OCR识别
        
        参数:
        img_path: 图片路径
        out_path: 输出图片路径（仅在convert_to_scan=True时使用）
        show_process: 是否显示处理过程中的图像
        convert_to_scan: 是否将图像转换为扫描件样式
        
        返回:
        识别结果字典
        """
        # 初始化OCR模型
        self.init_ocr()
        
        # 加载图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"错误:无法加载图片 '{img_path}'.请检查路径是否正确.")
            return None
        
        try:
            # 图像处理流程
            processed_image = self.process_image(image, show_process)
            if processed_image is None:
                return None
                
            # 如果需要转换为电子扫描件样式
            if convert_to_scan and out_path:
                self.convert_to_scan_style(processed_image, out_path)
            
            # 提取文本区域并识别
            gray = self.gray_image(processed_image)
            result, positions = self._select_text(gray)
            
            # 将结果转换为字典格式并验证身份证号码
            result_dict = self.process_result(result)
            
            return result_dict
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            return None
    
    # 透视变换
    def _perspective_image(self, image, contour):
        """
        透视变换
        
        参数:
        image: 输入图像
        contour: 轮廓
        
        返回:
        (宽度, 高度, 透视变换后的图像)
        """
        # 使用 epsilon 来近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果近似多边形是四边形
        if len(approx) != 4:
            print(f"警告: 轮廓不是四边形 (检测到 {len(approx)} 个顶点)")
            return None, None, None
        # 将点整理成二维数组
        points = approx.reshape(4, 2)

        # 对点按左上、右上、右下、左下顺序排序
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # 左上
        rect[2] = points[np.argmax(s)]  # 右下

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # 右上
        rect[3] = points[np.argmax(diff)]  # 左下

        # 计算宽度和高度
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        # 目标变换后的点
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        # 执行透视变换
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return max_width, max_height, warped
    
    # 固定图像
    def _fixed_perspective(self, w, h, perspective):
        """
        固定图像大小和方向
        
        参数:
        w: 宽度
        h: 高度
        perspective: 透视变换后的图像
        
        返回:
        调整后的图像
        """
        if w < h:
            perspective = np.rot90(perspective)
        resized = cv2.resize(perspective, (1084, 669), interpolation=cv2.INTER_AREA)
        return resized
    
    def _select_text(self, gray):
        """
        对灰度图像进行OCR识别，提取身份证信息
        
        参数:
        gray: 灰度图像
        
        返回:
        识别结果和位置信息的元组 (result, positions)
        """
        result = []
        positions = []
        
        # 身份证正面信息标签
        labels = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
        
        # 使用OCR实例进行识别
        ocr_result = self._ocr_instance.ocr(gray)
        
        # 创建标签和值的映射
        label_values = {label: "未识别" for label in labels}
        label_positions = {label: None for label in labels}
        
        # 处理OCR结果
        if ocr_result:
            # 首先查找包含标签的文本
            for item in ocr_result:
                text = item['text']
                position = item['position']
                
                # 检查是否包含标签
                for label in labels:
                    if label in text:
                        # 提取标签后的内容
                        parts = text.split(label, 1)
                        if len(parts) > 1 and parts[1].strip():
                            label_values[label] = parts[1].strip()
                            label_positions[label] = position
            
            # 然后处理可能分开的标签和值
            # 例如，标签和值可能在不同的OCR结果项中
            for i, item in enumerate(ocr_result):
                text = item['text'].strip()
                position = item['position']
                
                # 检查是否是标签
                for label in labels:
                    # 如果这一项是标签，且下一项可能是值
                    if text == label and label_values[label] == "未识别" and i + 1 < len(ocr_result):
                        next_text = ocr_result[i + 1]['text'].strip()
                        next_position = ocr_result[i + 1]['position']
                        
                        # 检查下一项是否可能是值（不包含其他标签）
                        if not any(other_label in next_text for other_label in labels):
                            label_values[label] = next_text
                            label_positions[label] = next_position
        
        # 构建结果列表
        for label in labels:
            result.append(f"{label}:{label_values[label]}")
            positions.append(label_positions[label])
        
        return result, positions