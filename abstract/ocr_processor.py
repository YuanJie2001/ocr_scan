import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

class OCRProcessor(ABC):
    """OCR处理基类，提供通用的图像处理和文本识别功能"""
    
    def __init__(self, ocr_instance=None):
        """初始化OCR处理器
        Args:
            ocr_instance: OCR实例，如果为None则需要在子类中初始化
        """
        self.ocr_instance = ocr_instance
    
    def process_image(self, image_path: str, show_process: bool = False) -> Optional[Dict[str, str]]:
        """处理图像的主流程
        Args:
            image_path: 图像文件路径
            show_process: 是否显示处理过程
        Returns:
            识别结果字典，如果处理失败返回None
        """
        try:
            # 1. 加载并预处理图像
            image = self._load_image(image_path)
            if image is None:
                return None
            
            # 2. 图像增强
            enhanced = self._enhance_image(image)
            if show_process:
                self._show_image(enhanced, 'enhanced')
            
            # 3. 定位文本区域
            text_regions = self._locate_text_regions(enhanced)
            if not text_regions:
                print("未能定位到文本区域")
                return None
            
            # 4. 识别文本
            result = self._recognize_text(enhanced, text_regions)
            
            return result
            
        except Exception as e:
            print(f"图像处理过程中发生错误: {str(e)}")
            return None
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"错误: 无法加载图片 '{image_path}'")
                return None
            return image
        except Exception as e:
            print(f"加载图片时出错: {str(e)}")
            return None
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强处理"""
        # 1. 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 3. 降噪
        denoised = cv2.medianBlur(enhanced, 3)
        
        # 4. 二值化
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
    
    def _show_image(self, image: np.ndarray, window_name: str = "default"):
        """显示图像"""
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @abstractmethod
    def _locate_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """定位文本区域，返回区域坐标列表 [(x, y, w, h)]"""
        pass
    
    @abstractmethod
    def _recognize_text(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Dict[str, str]:
        """识别文本区域内容，返回识别结果字典"""
        pass