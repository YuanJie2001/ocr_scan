from biz import IDCardProcessor
import os

def process_id_card(image_path, show_process=False):
    """处理身份证图像并显示结果"""
    # 确保文件路径存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return
    
    # 创建身份证处理器实例
    processor = IDCardProcessor()
    
    # 处理图像
    result = processor.process_image(image_path, show_process)
    
    # 显示识别结果
    if result:
        print("\n身份证识别结果:")
        print("-" * 30)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("-" * 30)
    else:
        print("识别失败，未能获取有效结果")

if __name__ == '__main__':
    # 处理正面
    print("\n处理身份证正面...")
    process_id_card("assets/front.jpg", show_process=True)
    
    # 处理背面
    print("\n处理身份证背面...")
    process_id_card("assets/back.jpg", show_process=True)