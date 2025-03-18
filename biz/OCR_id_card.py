import cv2
import numpy as np
from cnocr import CnOcr

# 全局OCR实例
_ocr_instance = None

def init_ocr(img_path, model_name='scene-densenet_lite_136-gru', show_process=False, retry=3):
    '''
    加载CnOcr的模型并处理身份证图像
    
    参数:
    img_path: 图片路径
    model_name: 使用的OCR模型名称
    show_process: 是否显示处理过程中的图像
    retry: 初始化重试次数
    
    返回:
    识别结果字典
    '''
    global _ocr_instance
    # 带重试机制的初始化
    for attempt in range(retry):
        try:
            if _ocr_instance is None:
                print(f"正在初始化OCR模型({attempt+1}/{retry}): {model_name}...")
                _ocr_instance = CnOcr(model_name)
                print("OCR模型初始化完成")
        except Exception as e:
            if attempt < retry - 1:
                print(f"OCR初始化失败，正在重试 ({attempt+1}/{retry}): {str(e)}")
                continue
            else:
                print(f"OCR模型初始化最终失败: {str(e)}")
                return None
    
    # 加载图片
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"错误:无法加载图片 '{img_path}'.请检查路径是否正确.")
            return None
        print(f"成功加载图片: {img_path}, 尺寸: {image.shape}")
    except Exception as e:
        print(f"加载图片时出错: {str(e)}")
        return None
        
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
        try:
            w, h, perspective = __perspective_image(image, contour)
            if perspective is None:
                print("透视变换失败,无法继续处理.")
                return None
            if show_process:
                show(perspective, "perspective")
        except Exception as e:
            print(f"透视变换过程中发生错误: {str(e)}")
            return None
            
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
            
        return result_dict
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

    
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
    threshold1(30):
    第一阈值(低阈值).
    用于确定像素梯度的最小值.如果梯度值小于 threshold1,这些像素将被认为不是边缘.

    threshold2(100):
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
    # 降低阈值以检测更多边缘
    edges = cv2.Canny(binary, 30, 100, 3, L2gradient=True)
    # 创建一个 3x3 的结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 增加膨胀迭代次数，使轮廓更连续
    dilation = cv2.dilate(edges, kernel, iterations=7)
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
    try:
        # 尝试不同的epsilon值来获取四边形
        epsilon_values = [0.02, 0.03, 0.01, 0.04, 0.05, 0.06, 0.07, 0.08]
        approx = None
        
        # 尝试不同的epsilon值
        for epsilon_factor in epsilon_values:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            current_approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果找到四边形，使用它
            if len(current_approx) == 4:
                approx = current_approx
                print(f"找到四边形轮廓，使用epsilon值: {epsilon_factor}")
                break
        
        # 如果没有找到四边形，使用最小外接矩形
        if approx is None or len(approx) != 4:
            print(f"警告: 未能找到四边形轮廓，使用最小外接矩形作为备选方案")
            # 使用最小外接矩形作为备选方案
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = np.int32(box)
            
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
        
        # 检查尺寸是否合理
        if max_width < 10 or max_height < 10:
            raise ValueError(f"透视变换后的尺寸过小: {max_width}x{max_height}")
        
        if max_width > 5000 or max_height > 5000:
            raise ValueError(f"透视变换后的尺寸过大: {max_width}x{max_height}")

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
    except Exception as e:
        print(f"透视变换失败: {str(e)}")
        # 如果透视变换失败，尝试使用原始图像作为备选方案
        try:
            h, w = image.shape[:2]
            print(f"使用原始图像作为备选方案，尺寸: {w}x{h}")
            return w, h, image
        except Exception as backup_error:
            print(f"备选方案也失败: {str(backup_error)}")
            return None, None, None

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
    
    # 按面积降序排序轮廓，优先处理较大的文本区域
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 限制处理的轮廓数量，避免处理太多噪点
    max_contours = min(30, len(sorted_contours))
    
    for i, contour in enumerate(sorted_contours[:max_contours]):
        try:
            # 使用自适应epsilon值，根据轮廓长度调整
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.002 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            # 计算宽高比，用于过滤不合理的文本区域
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 优化文本区域筛选条件
            # 1. 基本尺寸要求
            # 2. 位置限制（在图像范围内）
            # 3. 宽高比限制（避免过于狭长或过于方形的区域）
            if (h > 12 and w > 20 and  # 尺寸要求
                x < 290 and y < 290 and  # 位置限制
                0.5 < aspect_ratio < 15 and  # 宽高比限制
                w * h > 400):  # 面积限制
                
                # 在图像上标记检测到的文本区域
                cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resize_copy, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 扩大文本区域边界，确保完整捕获文字
                margin = 3  # 增加边界扩展量
                y_start = max(0, y - margin)
                y_end = min(gray.shape[0], y + h + margin)
                x_start = max(0, x - margin)
                x_end = min(gray.shape[1], x + w + margin)
                
                # 提取文本区域
                area = gray[y_start:y_end, x_start:x_end]
                
                # 检查区域是否为空
                if area.size == 0:
                    print(f"警告: 区域 {i} 为空，跳过处理")
                    continue
                
                # 图像增强处理
                # 1. 自适应直方图均衡化，提高对比度
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(area)
                
                # 2. 中值滤波去噪，保留文字边缘
                blur = cv2.medianBlur(enhanced, 3)
                
                # 3. 使用OTSU自适应阈值进行二值化
                data_area = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                
                # 4. 形态学操作，连接断开的文字
                kernel = np.ones((2,2), np.uint8)
                data_area = cv2.morphologyEx(data_area, cv2.MORPH_CLOSE, kernel)
                
                # 保存处理结果
                positions.append((x, y))
                data_areas['{}-{}'.format(x, y)] = data_area
                
                if show_process:
                    print(f"处理文本区域 {i}: 位置({x},{y}), 尺寸({w}x{h}), 宽高比({aspect_ratio:.2f})")
        except Exception as e:
            print(f"处理轮廓 {i} 时出错: {str(e)}")
            continue
    res, positions = __sort_id_card_text_location(data_areas, positions, ocr_instance, show_process)
    return res, positions


def __sort_id_card_text_location(data_areas, positions, ocr_instance, show_process=False):
    # 身份证正面信息标签
    labels = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
    
    # 增强版区域排序：先垂直后水平，结合身份证标准布局
    positions.sort(key=lambda p: (p[1]//50, p[0]))  # 50像素为一个垂直区间
    
    # 基于身份证标准布局过滤无效区域
    valid_positions = [p for p in positions if 50 < p[0] < 400 and 50 < p[1] < 300]
    positions = valid_positions[:6]  # 只保留前6个有效区域
    
    # 处理同一行的多个文本区域
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
            # 检查data_area是否为有效的图像数据
            if data_area is None or data_area.size == 0:
                raise ValueError("无效的图像数据")
                
            ocr_data = ocr_instance.ocr(data_area)
            
            # 增强对OCR结果的处理逻辑
            if ocr_data is not None:
                # 检查OCR结果的格式
                if isinstance(ocr_data, list) and len(ocr_data) > 0:
                    # 尝试不同的结果格式处理方式
                    try:
                        # 标准格式: [[['text'], confidence], ...]
                        ocr_result = ''.join([''.join(item[0]) for item in ocr_data if item and len(item) > 0]).replace(' ', '')
                    except (IndexError, TypeError):
                        try:
                            # 备选格式1: [['text', confidence], ...]
                            ocr_result = ''.join([item[0] for item in ocr_data if item and len(item) > 0]).replace(' ', '')
                        except (IndexError, TypeError):
                            try:
                                # 备选格式2: ['text', ...]
                                ocr_result = ''.join([item for item in ocr_data if item]).replace(' ', '')
                            except (TypeError):
                                # 如果以上都失败，尝试直接转换为字符串
                                ocr_result = str(ocr_data).replace(' ', '')
                    
                    # 处理特殊字段
                    if labels[i] == '出生':
                        # 尝试提取年月日
                        import re
                        # 增强日期格式校验
                    date_match = re.search(r'(19|20)\d{2}[年\\-]\d{1,2}[月\\-]\d{1,2}', ocr_result)
                    if date_match:
                        clean_date = re.sub(r'[^0-9]', '-', date_match.group())
                        year, month, day = clean_date.split('-')
                        ocr_result = f"{year}年{month.zfill(2)}月{day.zfill(2)}日"
                    
                    # 身份证号码校验
                    if labels[i] == '公民身份号码':
                        id_match = re.search(r'\d{17}[\dXx]', ocr_result)
                        if id_match:
                            ocr_result = id_match.group().upper()
                    
                    if ocr_result:
                        recognized_results.append(f"{labels[i]}:{ocr_result}")
                    else:
                        recognized_results.append(f"{labels[i]}:未识别")
                else:
                    recognized_results.append(f"{labels[i]}:未识别")
            else:
                recognized_results.append(f"{labels[i]}:未识别")
        except Exception as e:
            print(f"识别区域 {labels[i]} 时出错: {str(e)}")
            recognized_results.append(f"{labels[i]}:识别错误")
    
    # 输出识别结果
    for item in recognized_results:
        print(item)
        
    return recognized_results, positions



