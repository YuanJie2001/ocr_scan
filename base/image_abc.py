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

        目的: 1.降低图像计算复杂度 2.标准化预处理，为二值化，滤波提供统一基础 3.突出结构和纹理
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    # 通用方法 - 滤波噪点处理
    def filter_gray(self, gray):
        """
        滤波噪点处理
        1.medianBlur
        原理: 用邻域像素的中值代替中心像素值，直接消除离群点。
        优点: 对椒盐噪声（黑白噪点）效果极佳，能保留边缘。
        缺点: 对高斯噪声效果一般，大核可能导致纹理细节丢失。
        适用场景: 去除扫描文档中的斑点、摄像头传感器噪声、旧照片修复。
        核心参数: ksize 滤波器的大小，必须为正奇数。
        2.bilateralFilter
        原理: 结合空间邻近度（高斯核）和像素值相似度（颜色差异）进行加权平均，仅在颜色相近区域平滑。
        优点: 在平滑的同时保留边缘，适合处理纹理和强边缘图像。
        缺点: 计算复杂度高，对高频噪声敏感。
        核心参数:
        d 滤波器邻域的直径,决定了每个像素参与计算的范围 值越大,处理越慢,但模糊范围也越大
        sigmaColor 空间高斯函数标准差 值越大,颜色相近的像素更容易被模糊  值越小,滤波更注重保留颜色差异
        sigmaSpace 空间距离高斯函数标准差 值越大,较远像素也会对结果产生影响 值越小,只有靠近的像素会参与模糊计算.
        3.GaussianBlur
        原理: 基于高斯核（权重随距离中心像素的远近呈高斯分布）对图像进行加权平均，模糊均匀且平滑。
        优点: 计算快，有效抑制高斯噪声和随机噪声，模糊效果均匀。
        缺点: 糊边缘，对椒盐噪声效果差。
        核心参数: ksize 滤波器的大小，必须为正奇数 ; sigmaX X方向的高斯核标准差(控制模糊程度)
        适用场景: 通用去噪、预处理（如特征提取前的平滑）、图像降分辨率。
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
        目的: 1.极端数据简化 2.目标与背景分离,便于后续特征提取
        
        参数:
        threshold: 模糊处理后的图像
        
        1.大津法（Otsu）
        原理: （1）通过最大化图像灰度直方图的类间方差（目标与背景的分离度），自动寻找最佳全局阈值。
                （2）图像的灰度直方图呈双峰分布（即背景和目标灰度值差异明显）。
        优点: 全自动计算阈值，无需手动指定。 对双峰直方图的图像（如扫描文档、黑白对比明显的场景）效果极佳。
        缺点: 对光照不均匀、背景复杂的图像效果不佳。
        适用场景: 文档扫描、图像分割、目标识别（如车牌识别）。若直方图为单峰或重叠严重（如自然光照下的复杂图像），效果差。
        2.自适应阈值（Adaptive Thresholding）
        原理: （1）根据像素邻域的灰度值特性（如均值、高斯加权均值）自动计算局部阈值。
                （2）根据阈值类型（二值化或反二值化），将像素值分为前景和背景。
        优点：
            可处理光照不均的图像（如阴影、反光、渐变背景）。
            保留局部细节，适合自然场景下的文本或物体提取。
        缺点：计算量较大，实时性较差。对噪声敏感，需提前滤波（如高斯模糊）。
        返回:
        二值化后的图像
        """
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return threshold
    
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
    
    # 通用方法 - 轮廓检测 TODO 更新为轮廓检测模型
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