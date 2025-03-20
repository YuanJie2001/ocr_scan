from biz import init_ocr, show, convert_to_scan_style
import os
import cv2




if __name__ == '__main__':
    # 3. 处理正面 (转换为扫描样式)
    print("\n处理身份证国徽面 (扫描样式)...")
    init_ocr("assets/front.jpg",out_path="assets/output/front_scanned.jpg", show_process=True, convert_to_scan=True)
    print("\n处理身份证人像面 (扫描样式)...")
    result = init_ocr("assets/back.jpg",out_path="assets/output/back_scanned.jpg", show_process=True, convert_to_scan=True)
    if result:
        print("\n身份证识别结果:")
        print("-" * 30)
    for key, value in result.items():
        print(f"{key}: {value}")
        print("-" * 30)
    else:
        print("识别失败，未能获取有效结果")  
    