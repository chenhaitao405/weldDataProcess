"""
脚本名称: image_slicer.py
功能概述: 图像批处理工具，实现滑动窗口裁剪和图像增强（无标签处理）
详细说明:
    - 输入格式: JPG格式图像（也支持其他格式）
    - 处理流程: 滑动窗口裁剪 → 图像增强
    - 输出格式: 处理后的图像patches (JPG格式)
依赖模块: utils.image_processing
使用示例:
    # 基本使用
    python image_slicer.py --input_dir ./images --output_dir ./output

    # 使用窗宽窗位增强
    python image_slicer.py --input_dir ./images --output_dir ./output --enhance_mode windowing

    # 自定义重叠率
    python image_slicer.py --input_dir ./images --output_dir ./output --overlap 0.3

    # 自定义窗口大小
    python image_slicer.py --input_dir ./images --output_dir ./output --window_size 1024
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    # 图像处理
    enhance_image, sliding_window_crop, calculate_stride
)
from utils.constants import DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE, DEFAULT_JPEG_QUALITY


class ImageSlicer:
    """图像批处理器，用于切片和增强处理"""

    def __init__(self,
                 overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
                 enhance_mode: str = 'original',
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 output_format: str = 'jpg',
                 jpeg_quality: int = DEFAULT_JPEG_QUALITY):
        """
        初始化处理器

        Args:
            overlap_ratio: 滑动窗口重叠率 (0.0-1.0)
            enhance_mode: 增强模式 ('original' 或 'windowing')
            window_size: 滑动窗口大小（正方形边长）
            output_format: 输出图像格式 ('jpg', 'png', 'tif')
            jpeg_quality: JPEG压缩质量 (1-100)
        """
        self.overlap_ratio = overlap_ratio
        self.enhance_mode = enhance_mode
        self.window_size = window_size
        self.output_format = output_format.lower()
        self.jpeg_quality = jpeg_quality

        # 统计信息
        self.total_images_processed = 0
        self.total_patches_generated = 0
        self.failed_images = []

        print(f"ImageSlicer initialized with:")
        print(f"  - Window size: {self.window_size}x{self.window_size}")
        print(f"  - Overlap ratio: {self.overlap_ratio}")
        print(f"  - Enhancement mode: {self.enhance_mode}")
        print(f"  - Output format: {self.output_format}")
        if self.output_format == 'jpg':
            print(f"  - JPEG quality: {self.jpeg_quality}")

    def process_single_image(self, image_path: str, output_dir: str,
                           return_patches: bool = False) -> Optional[List[Dict]]:
        """
        处理单张图像

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录路径
            return_patches: 是否返回处理后的patches信息

        Returns:
            如果return_patches为True，返回patches信息列表；否则返回None
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            self.failed_images.append(image_path)
            return None

        # 确定滑动窗口大小
        h, w = image.shape[:2]
        actual_window_size = min(self.window_size, min(h, w))
        window_size_tuple = (actual_window_size, actual_window_size)

        # 如果图像小于窗口大小，直接处理整张图像
        if h <= actual_window_size and w <= actual_window_size:
            print(f"图像 {Path(image_path).name} 尺寸小于窗口大小，直接处理整张图像")
            patches = [{'patch': image, 'position': (0, 0), 'size': (w, h)}]
        else:
            # 计算步长
            stride = calculate_stride(window_size_tuple, self.overlap_ratio)

            # 滑动窗口裁剪
            patches = sliding_window_crop(image, window_size_tuple, stride)

        # 用于返回的patches信息
        processed_patches = [] if return_patches else None

        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 处理每个patch
        base_name = Path(image_path).stem
        patches_count = 0

        for i, patch_info in enumerate(patches):
            # 图像增强
            enhanced_patch = enhance_image(patch_info['patch'], self.enhance_mode)

            # 生成文件名
            patch_name = f"{base_name}_patch_{i:04d}"

            # 确定文件扩展名
            if self.output_format == 'jpg':
                ext = '.jpg'
            elif self.output_format == 'png':
                ext = '.png'
            elif self.output_format == 'tif':
                ext = '.tif'
            else:
                ext = '.jpg'  # 默认

            # 保存图像
            save_path = output_path / f"{patch_name}{ext}"

            if self.output_format == 'jpg':
                cv2.imwrite(str(save_path), enhanced_patch,
                           [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            elif self.output_format == 'png':
                cv2.imwrite(str(save_path), enhanced_patch,
                           [cv2.IMWRITE_PNG_COMPRESSION, 1])  # 低压缩，高质量
            else:
                cv2.imwrite(str(save_path), enhanced_patch)

            patches_count += 1

            # 如果需要返回patches信息
            if return_patches:
                processed_patches.append({
                    'original_patch': patch_info['patch'],
                    'enhanced_patch': enhanced_patch,
                    'position': patch_info['position'],
                    'size': patch_info['size'],
                    'patch_name': patch_name,
                    'save_path': str(save_path)
                })

        # 更新统计信息
        self.total_images_processed += 1
        self.total_patches_generated += patches_count

        print(f"处理完成: {Path(image_path).name} -> 生成 {patches_count} 个patches")

        return processed_patches

    def process_directory(self, input_dir: str, output_dir: str,
                         pattern: str = '*.jpg', recursive: bool = False):
        """
        批量处理目录中的图像

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            pattern: 文件匹配模式（支持通配符）
            recursive: 是否递归处理子目录
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 收集所有需要处理的图像文件
        if recursive:
            image_files = list(input_path.rglob(pattern))
        else:
            image_files = list(input_path.glob(pattern))

        if not image_files:
            print(f"在 {input_dir} 中没有找到匹配 {pattern} 的文件")
            return

        print(f"找到 {len(image_files)} 个图像文件")

        # 处理每个图像文件
        for image_file in tqdm(image_files, desc="处理图像"):
            # 如果是递归模式，保持目录结构
            if recursive:
                relative_path = image_file.parent.relative_to(input_path)
                current_output_dir = output_path / relative_path
            else:
                current_output_dir = output_path

            # 处理单张图像
            self.process_single_image(str(image_file), str(current_output_dir))

        # 打印统计信息
        self.print_statistics()

    def process_file_list(self, file_list: List[str], output_dir: str):
        """
        处理文件列表中的图像

        Args:
            file_list: 图像文件路径列表
            output_dir: 输出目录路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"处理 {len(file_list)} 个图像文件")

        for image_file in tqdm(file_list, desc="处理图像"):
            if not Path(image_file).exists():
                print(f"警告: 文件不存在 {image_file}")
                self.failed_images.append(image_file)
                continue

            self.process_single_image(image_file, str(output_path))

        # 打印统计信息
        self.print_statistics()

    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        return {
            'total_images_processed': self.total_images_processed,
            'total_patches_generated': self.total_patches_generated,
            'failed_images': self.failed_images,
            'average_patches_per_image': (
                self.total_patches_generated / self.total_images_processed
                if self.total_images_processed > 0 else 0
            )
        }

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("处理统计:")
        print(f"  成功处理图像数: {stats['total_images_processed']}")
        print(f"  生成patches总数: {stats['total_patches_generated']}")
        print(f"  平均每张图像patches数: {stats['average_patches_per_image']:.2f}")

        if stats['failed_images']:
            print(f"  失败图像数: {len(stats['failed_images'])}")
            print("  失败图像列表:")
            for failed_img in stats['failed_images']:
                print(f"    - {failed_img}")
        print("="*50)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='图像批处理工具 - 切片和增强处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（处理JPG图像）
  python image_slicer.py --input_dir ./images --output_dir ./output
  
  # 使用窗宽窗位增强
  python image_slicer.py --input_dir ./images --output_dir ./output --enhance_mode windowing
  
  # 自定义窗口大小和重叠率
  python image_slicer.py --input_dir ./images --output_dir ./output --window_size 1024 --overlap 0.3
  
  # 递归处理子目录
  python image_slicer.py --input_dir ./images --output_dir ./output --recursive
  
  # 处理PNG图像
  python image_slicer.py --input_dir ./images --output_dir ./output --pattern "*.png"
  
  # 处理TIF图像
  python image_slicer.py --input_dir ./images --output_dir ./output --pattern "*.tif"
  
  # 处理所有图像格式
  python image_slicer.py --input_dir ./images --output_dir ./output --pattern "*.*"
  
  # 输出PNG格式
  python image_slicer.py --input_dir ./images --output_dir ./output --output_format png
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径，包含图像文件')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                       help='文件匹配模式（支持通配符），默认: *.jpg')
    parser.add_argument('--recursive', action='store_true',
                       help='递归处理子目录')
    parser.add_argument('--enhance_mode', type=str, choices=['original', 'windowing'],
                       default='original',
                       help='图像增强模式: original(保持原始) 或 windowing(窗宽窗位)')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                       help=f'滑动窗口大小（正方形边长），默认: {DEFAULT_WINDOW_SIZE}')
    parser.add_argument('--overlap', type=float, default=DEFAULT_OVERLAP_RATIO,
                       help=f'滑动窗口重叠率 (0.0-1.0)，默认: {DEFAULT_OVERLAP_RATIO}')
    parser.add_argument('--output_format', type=str, choices=['jpg', 'png', 'tif'],
                       default='jpg',
                       help='输出图像格式，默认: jpg')
    parser.add_argument('--jpeg_quality', type=int, default=DEFAULT_JPEG_QUALITY,
                       help=f'JPEG压缩质量 (1-100)，默认: {DEFAULT_JPEG_QUALITY}')

    args = parser.parse_args()

    # 验证参数
    if args.overlap < 0 or args.overlap >= 1:
        parser.error("重叠率必须在 [0, 1) 范围内")

    if args.window_size <= 0:
        parser.error("窗口大小必须大于0")

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("JPEG质量必须在 1-100 范围内")

    # 创建处理器
    processor = ImageSlicer(
        overlap_ratio=args.overlap,
        enhance_mode=args.enhance_mode,
        window_size=args.window_size,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality
    )

    # 处理图像
    processor.process_directory(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()