import cv2
import numpy as np
from cnocr import CnOcr

def init_ocr(img_path,model_name='scene-densenet_lite_136-gru'):
    '''
    加载CnOcr的模型 默认加载的是 densenet_lite_136-gru 模型 中文识别模型
    CnOCR V2.3 重新训练了所有的模型,模型较 V2.2.* 精度更高.V2.3 按使用场景把模型分为几大类场景:
    scene:场景图片,适合识别一般拍照图片中的文字.此类模型以 scene- 开头,如模型 scene-densenet_lite_136-gru.
    doc:文档图片,适合识别规则文档的截图图片,如书籍扫描件等.此类模型以 doc- 开头,如模型 doc-densenet_lite_136-gru.
    number:仅识别纯数字(只能识别 0~9 十个数字)图片,适合银行卡号、身份证号等场景.此类模型以 number- 开头,如模型 number-densenet_lite_136-gru.
    general: 通用场景,适合图片无明显倾向的一般图片.此类模型无特定开头,与旧版模型名称保持一致,如模型 densenet_lite_136-gru.
    '''
    ocr = CnOcr(model_name)
    # show(image, "image")

    # 加载图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"错误:无法加载图片 '{img_path}'.请检查路径是否正确.")
        return
    # 1.灰度处理
    gray = __gray_image(image)
    # show(gray, "gray")
    # 2.滤波
    blur = __filter_gray(gray)
    # show(blur, "blur")
    # 3.二值化
    binary = __binary_filter(blur)
    # show(binary, "binary")
    # 4.边缘检测并膨胀
    dilation = __edge_binary(binary)
    show(dilation, "dilation")
    # 5.轮廓检测
    contour = __find_contours(dilation)
    if contour is None:
        print("轮廓检测失败,无法继续处理.")
        return
    # res = cv2.drawContours(dilation, contour, -1, (0, 255, 0), 2)
    # show(res, "contour")
    # 6.透视变换
    w, h, perspective = __perspective_image(image, contour)
    if perspective is None:
        print("透视变换失败,无法继续处理.")
        return
    show(perspective, "perspective")
    # 7. 固定位置
    resized = __fixed_perspective(w, h, perspective)
    # show(resized, "resized")

    
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
    gray = gray_image(resized)
    blur = filter_gray(gray)
    binary = binary_filter(blur)
    dilation = edge_binary(binary)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resize_copy = resized.copy()
    return contours, resize_copy


def __select_text(gray, contours, resize_copy):
    positions = []
    data_areas = {}
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h > 50 and x < 670:
            res = cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = gray[y:(y + h), x:(x + w)]
            blur = cv2.medianBlur(area, 3)
            data_area = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            positions.append((x, y))
            data_areas['{}-{}'.format(x, y)] = data_area
    res, positions = sort_id_card_text_location(data_areas, positions)
    return res, positions


def __sort_id_card_text_location(data_areas, positions):
    labels = ['姓名', '性别', '民族', '出生年', '出生月', '出生日', '住址', '公民身份号码', '签发机关', '有效期限']
    positions.sort(key=lambda p: p[1])
    result = []
    index = 0
    while index < len(positions) - 1:
        if positions[index + 1][1] - positions[index][1] < 10:
            temp_list = [positions[index + 1], positions[index]]
            for i in range(index + 1, len(positions)):
                if positions[i + 1][1] - positions[i][1] < 10:
                    temp_list.append(positions[i + 1])
                else:
                    break
            temp_list.sort(key=lambda p: p[0])
            positions[index:(index + len(temp_list))] = temp_list
            index = index + len(temp_list) - 1
        else:
            index += 1
    for index in range(len(positions)):
        position = positions[index]
        data_area = data_areas['{}-{}'.format(position[0], position[1])]
        ocr_data = ocr.ocr(data_area)
        ocr_result = ''.join([''.join(result[0]) for result in ocr_data]).replace(' ', '')
        # print('{}:{}'.format(labels[index], ocr_result))
        result.append('{}:{}'.format(labels[index], ocr_result))
        show(data_area, "data_area")

    for item in result:
        print(item)
    return result, positions



