import cv2
import numpy as np
from cnocr import CnOcr
from .id_validator import validate_id_card, extract_info_from_id_card
import random

# 全局OCR实例
_ocr_instance = None

def init_ocr(img_path, model_name='scene-densenet_lite_136-gru', show_process=False, convert_to_scan=False):
    '''
    加载CnOcr的模型并处理身份证图像
    
    参数:
    img_path: 图片路径
    model_name: 使用的OCR模型名称
    show_process: 是否显示处理过程中的图像
    
    返回:
    识别结果字典
    '''
    global _ocr_instance
    # 初始化OCR模型
    if _ocr_instance is None:
        _ocr_instance = CnOcr(model_name)
    
    # 加载图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"错误:无法加载图片 '{img_path}'.请检查路径是否正确.")
        return None
        
    # 如果需要转换为电子扫描件样式
    if convert_to_scan:
        image = convert_to_scan_style(image)
        
    try:
        # 1.灰度处理
        gray = __gray_image(image)
        if show_process:
            show(gray, "gray")
            
        # 2.滤波
        blur = __filter_gray(gray)
        if show_process:
            show(blur, "blur")
            
        # 3.二值化
        binary = __binary_filter(blur)
        if show_process:
            show(binary, "binary")
            
        # 4.边缘检测并膨胀
        dilation = __edge_binary(binary)
        if show_process:
            show(dilation, "dilation")
            
        # 5.轮廓检测
        contour = __find_contours(dilation)
        if contour is None:
            print("轮廓检测失败,无法继续处理.")
            return None
            
        # 6.透视变换
        w, h, perspective = __perspective_image(image, contour)
        if perspective is None:
            print("透视变换失败,无法继续处理.")
            return None
        if show_process:
            show(perspective, "perspective")
            
        # 7. 固定位置
        resized = __fixed_perspective(w, h, perspective)
        if show_process:
            show(resized, "resized")
            
        # 8. 检测文本位置并识别
        contours, resize_copy = __check_id_card_text_location(resized)
        if show_process:
            show(resize_copy, "resize_copy")
            
        # 9. 提取文本区域并识别
        gray = __gray_image(resized)
        result, positions = __select_text(gray, contours, resize_copy, _ocr_instance, show_process)
        
        # 10. 将结果转换为字典格式
        result_dict = {}
        for item in result:
            key, value = item.split(':', 1)
            result_dict[key] = value
        
        # 11. 验证身份证号码
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
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

    
# 将图像转换为电子扫描件样式
def convert_to_scan_style(image):
    """
    将图像转换为电子扫描件样式，保持原图的彩色信息
    
    参数:
    image: 输入图像
    
    返回:
    转换后的图像
    """
    # 1. 调整亮度和对比度，增强色彩鲜艳度
    alpha = 1.2  # 对比度增强因子 - 适度增强对比度
    beta = 10    # 亮度增强因子 - 适度增加亮度
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 2. 添加轻微高斯模糊，模拟扫描时的轻微模糊效果
    blurred = cv2.GaussianBlur(adjusted, (3, 3), 0.5)  # 减小模糊程度
    
    # 3. 添加轻微噪点，模拟扫描时的噪声
    noise = np.zeros(blurred.shape, np.uint8)
    cv2.randn(noise, 0, 8)  # 减小噪声强度
    noisy = cv2.add(blurred, noise)
    
    # 4. 锐化处理，增强文字边缘和色彩
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5, 7, -0.5],
                       [-0.5, -0.5, -0.5]])  # 减小锐化强度
    sharpened = cv2.filter2D(noisy, -1, kernel)
    
    # 5. 添加轻微的扫描线效果
    height, width = sharpened.shape[:2]
    scan_lines = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 每隔更多像素添加一条非常轻微的扫描线
    for i in range(0, height, 8):  # 增加间隔，减少扫描线
        if i + 1 < height:
            scan_lines[i:i+1, :] = [0, 0, 0]
    
    # 将扫描线与图像混合，减小扫描线强度
    scan_alpha = 0.01  # 减小扫描线强度
    result = cv2.addWeighted(sharpened, 1 - scan_alpha, scan_lines, scan_alpha, 0)
    
    # 6. 添加非常轻微的边缘暗角效果（Vignette）
    # 创建径向渐变遮罩
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (height//5*2+1, width//5*2+1), 0)
    
    # 应用暗角效果，减小暗角强度
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    vignette_alpha = 0.95  # 增大该值，减小暗角强度
    result = result * (mask * vignette_alpha + (1 - vignette_alpha))
    
    # 7. 增强色彩饱和度
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 增加饱和度
    s = cv2.multiply(s, 1.2)  # 增加饱和度
    # 合并通道
    hsv = cv2.merge([h, s, v])
    # 转回BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result.astype(np.uint8)

# 显示图片
def show(image, window_name="default"):
    if image is None:
        print("错误:镜像数据为None.请检查镜像路径或格式.")
        return
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 灰度处理
def __gray_image(image):
    # 将BGR三色图图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# 滤波噪点处理
def __filter_gray(gray):
    '''
    如果需要保留文字和边缘细节,优先使用 bilateralFilter.
    d 滤波器邻域的直径,决定了每个像素参与计算的范围 值越大,处理越慢,但模糊范围也越大
    sigmaColor 空间高斯函数标准差 值越大,颜色相近的像素更容易被模糊  值越小,滤波更注重保留颜色差异
    sigmaSpace 空间距离高斯函数标准差 值越大,较远像素也会对结果产生影响 值越小,只有靠近的像素会参与模糊计算
    如果噪声是椒盐噪声,使用 medianBlur.

    如果需要简单的平滑处理,使用 GaussianBlur.
    测试发现双边滤波对椒盐噪声的去除不好.因此混用中值滤波和双边滤波
    '''
    median = cv2.medianBlur(gray, ksize=5)
    blur = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)
    '''
    image: 输入图像(一般是灰度图像)
    (5, 5):高斯核的大小,表示高斯模糊窗口的宽度和高度(必须为奇数,例如 (3, 3) 或 (5, 5)).
     核越大,模糊效果越明显,但也可能导致细节丢失更多.
     0: 标准差(σ)
     若设置为 0,则 OpenCV 会根据高斯核的大小自动计算一个适合的值.
     如果手动设置,较大的标准差会导致更强的模糊效果.
    '''
    # 高斯模糊 cv2.GaussianBlur(image, (5, 5), 0)
    return blur


# 二值化
def __binary_filter(blur):
    """
    mavValue 二值化后分配给像素的最大值(通常为 255,表示白色)

    adaptiveMethod(cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    自适应阈值计算方法:
    cv2.ADAPTIVE_THRESH_MEAN_C:使用邻域内所有像素的均值来计算阈值.
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C:使用邻域内所有像素的高斯加权平均值来计算阈值(对边缘更敏感).

    thresholdType(cv2.THRESH_BINARY):
    二值化类型:
    cv2.THRESH_BINARY:像素值大于阈值时为 maxValue,否则为 0.
    cv2.THRESH_BINARY_INV:像素值小于阈值时为 maxValue,否则为 0

    blockSize(11):
    用于计算阈值的局部区域的大小,必须是奇数(如 3, 5, 7, 11).
    值越大,考虑的局部区域越大.

    C(2):
    从计算出的均值或加权均值中减去的常数,用于微调阈值.
    值越大,整体阈值越低(生成更多的黑色像素).
    """
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return binary


# 边缘检测和膨胀处理
def __edge_binary(binary):
    """
    threshold1(50):
    第一阈值(低阈值).
    用于确定像素梯度的最小值.如果梯度值小于 threshold1,这些像素将被认为不是边缘.

    threshold2(150):
    第二阈值(高阈值).
    用于确定像素梯度的最大值.如果梯度值大于 threshold2,这些像素被认为是强边缘.

    threshold1 是低阈值,用于连接强边缘(高阈值内确定的边缘)和弱边缘.
    通常建议 threshold2 设置为 threshold1 的 2-3 倍.

    可选参数
    apertureSize(默认值:3):
    用于计算图像梯度的 Sobel 算子的大小.
    可选值:3, 5, 7.
    较大的值可能更能平滑图像,但也可能模糊细节

    L2gradient(默认值:False)是否使用更精确的梯度幅值计算
    True:使用公式 根号下 (G上底2下底X + G上底2下底Y)
    False: |G下底Z|+|G下底Y|
    """
    edges = cv2.Canny(binary, 50, 150, 3, L2gradient=True)
    # 创建一个 3x3 的结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 膨胀操作
    dilation = cv2.dilate(edges, kernel, iterations=5)
    return dilation


# 轮廓检测
def __find_contours(dilation):
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    '''
    由于可能因为噪点的影响 导致检测多个噪点单位
    对其进行排序,取最大范围即为所求
    '''
    # contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    contour = max(contours, key=cv2.contourArea)
    return contour


# 透视变换
def __perspective_image(image, contour):
    # 使用 epsilon 来近似多边形
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果近似多边形是四边形
    if len(approx) != 4:
        raise ValueError("轮廓不是四边形")
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
def __fixed_perspective(w, h, perspective):
    if w < h:
        perspective = np.rot90(perspective)
    resized = cv2.resize(perspective, (300, 300), interpolation=cv2.INTER_AREA)
    return resized


# 检验身份证文本位置
def __check_id_card_text_location(resized):
    # 对截取到得身份证重新 灰度、滤波、二值化...
    gray = __gray_image(resized)
    blur = __filter_gray(gray)
    binary = __binary_filter(blur)
    dilation = __edge_binary(binary)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resize_copy = resized.copy()
    return contours, resize_copy


def __select_text(gray, contours, resize_copy, ocr_instance, show_process=False):
    positions = []
    data_areas = {}
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h > 20 and w > 30 and x < 290:  # 调整参数以适应300x300的图像
            cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = gray[y:(y + h), x:(x + w)]
            blur = cv2.medianBlur(area, 3)
            data_area = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            positions.append((x, y))
            data_areas['{}-{}'.format(x, y)] = data_area
    res, positions = __sort_id_card_text_location(data_areas, positions, ocr_instance, show_process)
    return res, positions


def __sort_id_card_text_location(data_areas, positions, ocr_instance, show_process=False):
    # 身份证正面信息标签
    labels = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
    
    # 按y坐标排序（从上到下）
    positions.sort(key=lambda p: p[1])
    
    # 处理同一行的多个文本区域
    result = []
    index = 0
    while index < len(positions) - 1:
        # 如果两个区域的y坐标接近，认为它们在同一行
        if index + 1 < len(positions) and abs(positions[index + 1][1] - positions[index][1]) < 10:
            # 收集同一行的所有区域
            same_row = [positions[index], positions[index + 1]]
            i = index + 1
            while i + 1 < len(positions) and abs(positions[i + 1][1] - positions[i][1]) < 10:
                same_row.append(positions[i + 1])
                i += 1
            
            # 按x坐标排序（从左到右）
            same_row.sort(key=lambda p: p[0])
            
            # 更新positions列表
            positions[index:index + len(same_row)] = same_row
            index += len(same_row)
        else:
            index += 1
    
    # 识别每个区域的文本
    recognized_results = []
    for i, position in enumerate(positions):
        if i >= len(labels):
            break  # 防止超出标签数量
            
        data_area = data_areas['{}-{}'.format(position[0], position[1])]
        
        # 显示处理中的区域
        if show_process:
            show(data_area, f"区域_{i}_{labels[i]}")
        
        # OCR识别
        try:
            ocr_data = ocr_instance.ocr(data_area)
            if ocr_data and len(ocr_data) > 0:
                # 合并OCR结果
                ocr_result = ''.join([''.join(item[0]) for item in ocr_data if item]).replace(' ', '')
                
                # 处理特殊字段
                if labels[i] == '出生':
                    # 尝试提取年月日
                    import re
                    date_match = re.search(r'(\d{4})年?(\d{1,2})月?(\d{1,2})日?', ocr_result)
                    if date_match:
                        year, month, day = date_match.groups()
                        ocr_result = f"{year}年{month}月{day}日"
                
                recognized_results.append(f"{labels[i]}:{ocr_result}")
            else:
                recognized_results.append(f"{labels[i]}:未识别")
        except Exception as e:
            print(f"识别区域 {labels[i]} 时出错: {str(e)}")
            recognized_results.append(f"{labels[i]}:识别错误")
    
    # 输出识别结果
    for item in recognized_results:
        print(item)
        
    return recognized_results, positions



