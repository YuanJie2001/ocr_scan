from .id_card_processor import IDCardProcessor

__version__="0.0.1"
__author__="YuanJie"
__all__ = ["init_ocr"]
print("package {} is imported!".format(__package__))

# 为了保持向后兼容，创建一个全局实例
_id_card_processor = None

def init_ocr(img_path, out_path=None, model_name='scene-densenet_lite_136-gru', show_process=False, convert_to_scan=False):
    """
    加载CnOcr的模型并处理身份证图像
    
    参数:
    img_path: 图片路径
    out_path: 输出图片路径（仅在convert_to_scan=True时使用）
    model_name: 使用的OCR模型名称
    show_process: 是否显示处理过程中的图像
    convert_to_scan: 是否将图像转换为扫描件样式
    
    返回:
    识别结果字典
    """
    global _id_card_processor
    if _id_card_processor is None:
        _id_card_processor = IDCardProcessor(model_name)
    
    return _id_card_processor.process_id_card(img_path, out_path, show_process, convert_to_scan)