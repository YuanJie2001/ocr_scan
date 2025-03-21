from abc import ABC, abstractmethod
import cv2
import numpy as np

class ImageABC(ABC):
    """
    图像处理抽象基类，提供通用的图像处理方法
    
    该类定义了图像处理的基本流程和通用方法，子类需要实现特定的处理逻辑
    """
    
    def __init__(self):
        """
        初始化图像处理器
        """
        pass
    
    @abstractmethod
    def process_image(self, image, show_process=False):
        """
        处理图像的主方法，需要由子类实现
        
        参数:
        image: 输入图像
        show_process: 是否显示处理过程
        
        返回:
        处理后的图像
        """
        pass
    
    @abstractmethod
    def process_result(self, result):
        """
        处理识别结果，需要由子类实现
        
        参数:
        result: 识别结果
        
        返回:
        处理后的结果
        """
        pass
    
    # 通用方法 - 显示图片
    def show(self, image, window_name="default"):
        """
        显示图片
        
        参数:
        image: 要显示的图像
        window_name: 窗口名称
        """
        if image is None:
            print("错误:镜像数据为None.请检查镜像路径或格式.")
            return
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 通用方法 - 灰度处理
    def gray_image(self, image):
        """
        将BGR三色图图像转换为灰度图
        
        参数:
        image: 输入图像
        
        返回:
        灰度图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    # 通用方法 - 滤波噪点处理
    def filter_gray(self, gray):
        """
        滤波噪点处理
        
        如果需要保留文字和边缘细节,优先使用 bilateralFilter.
        d 滤波器邻域的直径,决定了每个像素参与计算的范围 值越大,处理越慢,但模糊范围也越大
        sigmaColor 空间高斯函数标准差 值越大,颜色相近的像素更容易被模糊  值越小,滤波更注重保留颜色差异
        sigmaSpace 空间距离高斯函数标准差 值越大,较远像素也会对结果产生影响 值越小,只有靠近的像素会参与模糊计算
        如果噪声是椒盐噪声,使用 medianBlur.

        如果需要简单的平滑处理,使用 GaussianBlur.
        测试发现双边滤波对椒盐噪声的去除不好.因此混用中值滤波和双边滤波
        
        参数:
        gray: 灰度图像
        
        返回:
        滤波后的图像
        """
        median = cv2.medianBlur(gray, ksize=5)
        bilateral = cv2.bilateralFilter(median, d=9, sigmaColor=25, sigmaSpace=50)
        gaussian = cv2.GaussianBlur(bilateral, (3, 3), 0)
        return gaussian
    
    # 通用方法 - 二值化
    def binary_filter(self, blur):
        """
        二值化处理
        
        参数:
        blur: 模糊处理后的图像
        
        返回:
        二值化后的图像
        """
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return binary
    
    # 通用方法 - 边缘检测和膨胀处理
    def edge_binary(self, binary):
        """
        边缘检测和膨胀处理
        
        参数:
        binary: 二值化后的图像
        
        返回:
        边缘检测和膨胀处理后的图像
        """
        edges = cv2.Canny(binary, 100, 150, 3)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(edges, kernel, iterations=5)
        return dilation
    
    # 通用方法 - 轮廓检测
    def find_contours(self, dilation):
        """
        轮廓检测
        
        参数:
        dilation: 膨胀处理后的图像
        
        返回:
        最大轮廓
        """
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        return contour
    
    # 通用方法 - 将图像转换为电子扫描件样式
    def convert_to_scan_style(self, image, out_path=None):
        """
        将图像转换为电子扫描件样式，保持原图的彩色信息
        
        参数:
        image: 输入图像
        out_path: 输出图像路径
        
        返回:
        转换后的图像
        """
        # 1. 调整亮度和对比度，轻微增强（降低对比度强度）
        alpha = 1.2  # 对比度增强因子（降低）
        beta = 1     # 亮度增强因子（降低）
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 2. 锐化处理，增强文字边缘和色彩
        kernel = np.array([[-0.2, -0.2, -0.2],
                        [-0.2, 3, -0.2],
                        [-0.2, -0.2, -0.2]])  # 降低锐化强度
        sharpened = cv2.filter2D(adjusted, -1, kernel)
        
        # 3. 添加轻微的扫描线效果
        height, width = sharpened.shape[:2]
        scan_lines = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 每隔一定像素添加一条轻微的扫描线
        for i in range(0, height, 15):  # 增加间隔，减少扫描线
            if i + 1 < height:
                scan_lines[i:i+1, :] = [0, 0, 0]
        
        # 将扫描线与图像混合
        scan_alpha = 0.003  # 降低扫描线强度
        result = cv2.addWeighted(sharpened, 1 - scan_alpha, scan_lines, scan_alpha, 0)
        
        # 6. 增强色彩饱和度
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.05)  # 轻微增加饱和度
        hsv = cv2.merge([h, s, v])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        scanned = bgr.astype(np.uint8)
        
        # 保存图像
        if out_path:
            print("\n保存扫描样式图像...")
            result = cv2.imwrite(out_path, scanned)
            if result:
                print(f"扫描样式图像已保存到: {out_path}")
            else:
                print(f"保存图像失败: {out_path}")
            self.show(scanned, "scanned")
        
        return scanned