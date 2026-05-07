"""
身份证照片 → 电子化 PDF 主入口

支持单张处理和批量处理，使用 argparse 提供 CLI 接口。

处理流程:
    manual crop(Qt6) → scan extract(OpenCV) → OCR(RapidOCR) → parse → validate → PDF
"""
import argparse
import logging
import os
import sys
import time

from ocr_engine import OCREngine, preload_onnxruntime

preload_onnxruntime()

import cv2

import config
from config import setup_logging
from image_scan_extractor import ImageScanExtractor
from parser import IDCardParser
from pdf_generator import PDFGenerator
from validator import IDCardValidator

logger = logging.getLogger(__name__)

# 支持的图像扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _select_manual_roi(image):
    """使用手动框选返回 ROI 边界框"""
    from roi_selector import select_roi_polygon

    return select_roi_polygon(image)


def process_single(
    image_path: str,
    output_dir: str,
    scanner: ImageScanExtractor,
    ocr: OCREngine,
    parser: IDCardParser,
    show_process: bool = False,
    scan_style: bool = False,
) -> dict | None:
    """
    处理单张身份证图片

    Args:
        image_path: 图片路径
        output_dir: 输出目录
        scanner: 图像扫描提取器
        ocr: OCR 引擎
        parser: 字段解析器
        show_process: 是否显示处理过程

    Returns:
        处理结果字典，失败返回 None
    """
    logger.info("=" * 60)
    logger.info("开始处理: %s", image_path)
    start_time = time.time()

    # 1. 加载图像
    image = scanner.load_image(image_path)
    if image is None:
        return None

    # 2. 强制手动裁剪
    try:
        bbox = _select_manual_roi(image)
    except Exception as exc:
        logger.error("Qt6 ROI failed: %s", exc)
        logger.error("Ensure PyQt6 is installed and GUI is available. Try: uv sync --extra gui")
        return None
    if bbox is None:
        logger.error("未选择有效裁剪区域，终止处理")
        return None

    # 3. 预处理（裁剪 + 透视矫正 + 旋转校正）
    basename = os.path.splitext(os.path.basename(image_path))[0]
    scan_output_path = os.path.join(output_dir, f"{basename}_scan.jpg") if scan_style else None

    processed, scan_image = scanner.extract(
        image,
        roi=bbox,
        show_process=show_process,
        scan_style=scan_style,
        scan_output_path=scan_output_path,
    )

    if processed is None:
        logger.error("图像预处理失败")
        return None

    # 4. OCR 识别
    ocr_results = ocr.recognize(processed)
    if not ocr_results:
        logger.warning("OCR 未识别到任何文本")
        return None

    # 5. 解析结构化字段
    id_info = parser.parse(ocr_results)

    # 6. 校验身份证号码
    validation_result = "未识别"
    if id_info.id_number:
        is_valid, error_msg = IDCardValidator.validate(id_info.id_number)
        validation_result = "✓ 通过" if is_valid else f"✗ {error_msg}"
        logger.info("身份证号码校验: %s → %s", id_info.id_number, validation_result)
    else:
        logger.warning("未能识别出身份证号码")

    # 7. 保存证件图并生成 PDF
    card_output_path = os.path.join(output_dir, f"{basename}_card.jpg")
    cv2.imwrite(card_output_path, processed)

    pdf_path = os.path.join(output_dir, f"{basename}.pdf")
    PDFGenerator.generate(
        pdf_path=pdf_path,
        card_image_path=card_output_path,
        id_info=id_info,
        ocr_results=ocr_results,
    )

    elapsed = time.time() - start_time
    logger.info("处理完成 (%.2fs): %s → %s", elapsed, image_path, pdf_path)

    # 构造返回结果
    result = id_info.to_dict()
    result["身份证号码校验"] = validation_result
    result["PDF路径"] = pdf_path
    result["证件图片路径"] = card_output_path
    if scan_image is not None and scan_output_path:
        result["扫描件路径"] = scan_output_path
    return result


def process_batch(
    input_dir: str,
    output_dir: str,
    scanner: ImageScanExtractor,
    ocr: OCREngine,
    parser: IDCardParser,
    show_process: bool = False,
    scan_style: bool = False,
) -> list[dict]:
    """
    批量处理目录下的所有身份证图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        scanner: 图像扫描提取器
        ocr: OCR 引擎
        parser: 字段解析器
        show_process: 是否显示处理过程

    Returns:
        所有处理结果列表
    """
    image_files = []
    for f in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(os.path.join(input_dir, f))

    if not image_files:
        logger.warning("目录中没有找到图像文件: %s", input_dir)
        return []

    logger.info("找到 %d 个图像文件，开始批量处理...", len(image_files))

    results = []
    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        logger.info("[%d/%d] 处理中...", i, len(image_files))
        result = process_single(
            img_path,
            output_dir,
            scanner,
            ocr,
            parser,
            show_process,
            scan_style,
        )
        if result:
            result["源文件"] = img_path
            results.append(result)
            success_count += 1
        else:
            logger.warning("[%d/%d] 处理失败: %s", i, len(image_files), img_path)

    logger.info("批量处理完成: 成功 %d/%d", success_count, len(image_files))
    return results


def print_results(results: list[dict]) -> None:
    """格式化打印处理结果"""
    for i, result in enumerate(results, 1):
        print(f"\n{'=' * 50}")
        print(f"  结果 #{i}")
        print(f"{'=' * 50}")
        for key, value in result.items():
            print(f"  {key}: {value}")
    print()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    ap = argparse.ArgumentParser(
        description="身份证照片 → 电子化 PDF 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单张图片
  python main.py --input assets/back.jpg

  # 批量处理目录
  python main.py --input-dir assets --output-dir output

  # 手动框选裁剪（每张图均会弹窗）
  python main.py --input photo.jpg
        """,
    )

    # 输入参数（互斥）
    input_group = ap.add_mutually_exclusive_group()
    input_group.add_argument("--input", "-i", help="单张图片路径")
    input_group.add_argument("--input-dir", "-d", help="批量处理的输入目录")

    # 输出参数
    ap.add_argument("--output-dir", "-o", default=config.OUTPUT_DIR, help="输出目录 (默认: %(default)s)")

    # 其他参数
    ap.add_argument("--show", action="store_true", help="显示处理过程中的图像")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None, help="日志级别")
    ap.add_argument("--scan-style", action="store_true", help="输出扫描风格图（可能偏亮）")

    return ap


def _run_gui() -> None:
    try:
        from gui_app import run
    except Exception as exc:
        logger.error("GUI launch failed: %s", exc)
        sys.exit(1)
    run()


def main():
    """主函数"""
    args = build_parser().parse_args()

    # 初始化日志
    setup_logging(args.log_level)

    if not args.input and not args.input_dir:
        _run_gui()
        return

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化各模块
    logger.info("初始化处理模块...")
    scanner = ImageScanExtractor()
    ocr = OCREngine()
    parser = IDCardParser()

    if args.input:
        # 单张处理
        if not os.path.exists(args.input):
            logger.error("输入文件不存在: %s", args.input)
            sys.exit(1)

        result = process_single(
            args.input,
            args.output_dir,
            scanner,
            ocr,
            parser,
            show_process=args.show,
            scan_style=args.scan_style,
        )
        if result:
            print_results([result])
        else:
            logger.error("处理失败")
            sys.exit(1)

    elif args.input_dir:
        # 批量处理
        if not os.path.isdir(args.input_dir):
            logger.error("输入目录不存在: %s", args.input_dir)
            sys.exit(1)

        results = process_batch(
            args.input_dir,
            args.output_dir,
            scanner,
            ocr,
            parser,
            show_process=args.show,
            scan_style=args.scan_style,
        )
        if results:
            print_results(results)
        else:
            logger.warning("没有成功处理的结果")
            sys.exit(1)


if __name__ == "__main__":
    main()
