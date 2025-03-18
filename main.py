from biz import init_ocr, show, convert_to_scan_style
import os
import cv2

def process_id_card(image_path, show_process=False, convert_to_scan=False):
    """处理身份证图像并显示结果"""
    # 确保文件路径存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return
    
    # 调用OCR识别
    result = init_ocr(image_path, show_process=show_process, convert_to_scan=convert_to_scan)
    
    # 显示识别结果
    if result:
        print("\n身份证识别结果:")
        print("-" * 30)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("-" * 30)
    else:
        print("识别失败，未能获取有效结果")

def save_scan_image(image_path, output_path):
    """将图像转换为扫描样式并保存"""
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图片 '{image_path}'")
        return False
    
    # 转换为扫描样式
    scanned = convert_to_scan_style(image)
    
    # 保存图像
    result = cv2.imwrite(output_path, scanned)
    if result:
        print(f"扫描样式图像已保存到: {output_path}")
    else:
        print(f"保存图像失败: {output_path}")
    
    return result

if __name__ == '__main__':
    # 1. 处理正面 (不转换为扫描样式)
    print("\n处理身份证正面...")
    process_id_card("assets/front.jpg", show_process=True)
    
    # 2. 处理背面 (不转换为扫描样式)
    print("\n处理身份证背面...")
    process_id_card("assets/back.jpg", show_process=True)
    
    # 3. 处理正面 (转换为扫描样式)
    print("\n处理身份证正面 (扫描样式)...")
    process_id_card("assets/front.jpg", show_process=True, convert_to_scan=True)
    
    # 4. 保存扫描样式图像示例
    print("\n保存扫描样式图像...")
    os.makedirs("assets/output", exist_ok=True)
    save_scan_image("assets/front.jpg", "assets/output/front_scanned.jpg")
    save_scan_image("assets/back.jpg", "assets/output/back_scanned.jpg")