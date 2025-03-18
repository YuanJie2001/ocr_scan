import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from cnocr import CnOcr
from abstract import OCRProcessor

class IDCardProcessor(OCRProcessor):
    """身份证识别处理器，实现身份证特定的文本区域定位和识别功能"""
    
    def __init__(self, model_name='scene-densenet_lite_136-gru'):
        """初始化身份证识别处理器
        Args:
            model_name: OCR模型名称
        """
        super().__init__()
        self.model_name = model_name
        self._init_ocr()
    
    def _init_ocr(self, retry=3):
        """初始化OCR模型"""
        for attempt in range(retry):
            try:
                print(f"正在初始化OCR模型({attempt+1}/{retry}): {self.model_name}...")
                self.ocr_instance = CnOcr(self.model_name)
                print("OCR模型初始化完成")
                return
            except Exception as e:
                if attempt < retry - 1:
                    print(f"OCR初始化失败，正在重试 ({attempt+1}/{retry}): {str(e)}")
                    continue
                else:
                    print(f"OCR模型初始化最终失败: {str(e)}")
                    raise
    
    def _locate_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """定位身份证文本区域"""
        # 1. 边缘检测
        edges = cv2.Canny(image, 30, 100, 3, L2gradient=True)
        
        # 2. 膨胀处理
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=7)
        
        # 3. 轮廓检测
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. 筛选文本区域
        text_regions = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if (h > 12 and w > 20 and
                x < 290 and y < 290 and
                0.5 < aspect_ratio < 15 and
                w * h > 400):
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def _recognize_text(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Dict[str, str]:
        """识别文本区域内容"""
        # 身份证信息标签
        labels = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
        
        # 按垂直位置排序区域
        regions.sort(key=lambda r: (r[1]//50, r[0]))
        
        # 识别结果
        result = {}
        for i, (x, y, w, h) in enumerate(regions[:6]):
            if i >= len(labels):
                break
                
            # 提取并增强文本区域
            margin = 3
            y_start = max(0, y - margin)
            y_end = min(image.shape[0], y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(image.shape[1], x + w + margin)
            
            area = image[y_start:y_end, x_start:x_end]
            
            # OCR识别
            try:
                ocr_data = self.ocr_instance.ocr(area)
                if ocr_data and isinstance(ocr_data, list) and len(ocr_data) > 0:
                    text = ''.join([''.join(item[0]) for item in ocr_data if item and len(item) > 0]).replace(' ', '')
                    
                    # 处理特殊字段
                    if labels[i] == '出生':
                        import re
                        date_match = re.search(r'(19|20)\d{2}[年\\-]\d{1,2}[月\\-]\d{1,2}', text)
                        if date_match:
                            clean_date = re.sub(r'[^0-9]', '-', date_match.group())
                            year, month, day = clean_date.split('-')
                            text = f"{year}年{month.zfill(2)}月{day.zfill(2)}日"
                    
                    elif labels[i] == '公民身份号码':
                        id_match = re.search(r'\d{17}[\dXx]', text)
                        if id_match:
                            text = id_match.group().upper()
                    
                    result[labels[i]] = text
            except Exception as e:
                print(f"识别区域 {labels[i]} 时出错: {str(e)}")
                result[labels[i]] = '识别错误'
        
        return result