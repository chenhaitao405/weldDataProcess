import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import shutil


class WeldImagePreprocessor:
    """焊缝X射线图像预处理器，实现滑动窗口裁剪和图像增强"""

    def __init__(self, overlap_ratio: float = 0.5):
        """
        初始化预处理器

        Args:
            overlap_ratio: 滑动窗口重叠率，默认0.5（50%）
        """
        self.overlap_ratio = overlap_ratio

        # 统计信息
        self.total_patches = 0
        self.patches_with_defects = 0
        self.patches_without_defects = 0

    def sliding_window_crop(self, image: np.ndarray, window_size: Tuple[int, int],
                            stride: Tuple[int, int]) -> List[Dict]:
        """
        滑动窗口裁剪

        Args:
            image: 输入图像
            window_size: 窗口大小 (height, width)
            stride: 滑动步长 (y_stride, x_stride)

        Returns:
            裁剪后的图像patches列表，每个元素包含patch和位置信息
        """
        h, w = image.shape[:2]
        window_h, window_w = window_size
        y_stride, x_stride = stride

        patches = []

        # 从上到下，从左到右滑动
        for y in range(0, h - window_h + 1, y_stride):
            for x in range(0, w - window_w + 1, x_stride):
                # 裁剪patch
                patch = image[y:y + window_h, x:x + window_w]

                # 保存patch信息
                patch_info = {
                    'patch': patch,
                    'position': (x, y),  # 左上角坐标
                    'size': (window_w, window_h)
                }
                patches.append(patch_info)

        # 处理边缘情况：右边缘
        if w % x_stride != 0 and w > window_w:
            for y in range(0, h - window_h + 1, y_stride):
                x = w - window_w
                patch = image[y:y + window_h, x:x + window_w]
                patch_info = {
                    'patch': patch,
                    'position': (x, y),
                    'size': (window_w, window_h)
                }
                # 检查是否重复
                if not any(p['position'] == (x, y) for p in patches):
                    patches.append(patch_info)

        # 处理边缘情况：下边缘
        if h % y_stride != 0 and h > window_h:
            for x in range(0, w - window_w + 1, x_stride):
                y = h - window_h
                patch = image[y:y + window_h, x:x + window_w]
                patch_info = {
                    'patch': patch,
                    'position': (x, y),
                    'size': (window_w, window_h)
                }
                # 检查是否重复
                if not any(p['position'] == (x, y) for p in patches):
                    patches.append(patch_info)

        # 处理边缘情况：右下角
        if w % x_stride != 0 and h % y_stride != 0 and w > window_w and h > window_h:
            x = w - window_w
            y = h - window_h
            patch = image[y:y + window_h, x:x + window_w]
            patch_info = {
                'patch': patch,
                'position': (x, y),
                'size': (window_w, window_h)
            }
            # 检查是否重复
            if not any(p['position'] == (x, y) for p in patches):
                patches.append(patch_info)

        return patches

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像增强处理

        Args:
            image: 输入图像（可能是16位）

        Returns:
            增强后的8位3通道图像
        """
        # 1. 转换为8位并进行直方图均衡
        if image.dtype == np.uint16:
            # 16位转8位
            image_8bit = (image / 256).astype(np.uint8)
        else:
            # 已经是8位
            image_8bit = image

        # 直方图均衡
        image_equalized = cv2.equalizeHist(image_8bit)

        # 2. CLAHE处理
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_equalized)

        # 3. 转换为3通道图像
        image_3ch = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)

        return image_3ch

    def adjust_annotations(self, annotations: Dict, crop_x: int, crop_y: int,
                           crop_w: int, crop_h: int) -> Dict:
        """
        调整标注以适应裁剪后的图像

        Args:
            annotations: 原始标注数据
            crop_x, crop_y: 裁剪区域左上角坐标
            crop_w, crop_h: 裁剪区域尺寸

        Returns:
            调整后的标注数据
        """
        new_annotations = {
            'version': annotations.get('version', '4.5.7'),
            'flags': annotations.get('flags', {}),
            'shapes': [],
            'imagePath': annotations.get('imagePath', ''),
            'imageData': None,
            'imageHeight': crop_h,
            'imageWidth': crop_w
        }

        # 处理每个缺陷标注
        for shape in annotations.get('shapes', []):
            # 获取标注的边界框
            points = np.array(shape['points'])

            # 检查标注是否在裁剪区域内
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)

            # 判断是否与裁剪区域有交集
            if (max_x >= crop_x and min_x < crop_x + crop_w and
                    max_y >= crop_y and min_y < crop_y + crop_h):

                # 调整坐标
                new_points = []
                for point in points:
                    new_x = point[0] - crop_x
                    new_y = point[1] - crop_y

                    # 确保坐标在裁剪区域内
                    new_x = max(0, min(new_x, crop_w - 1))
                    new_y = max(0, min(new_y, crop_h - 1))

                    new_points.append([float(new_x), float(new_y)])

                # 创建新的shape
                new_shape = {
                    'label': shape['label'],
                    'points': new_points,
                    'group_id': shape.get('group_id'),
                    'shape_type': shape['shape_type'],
                    'flags': shape.get('flags', {})
                }

                new_annotations['shapes'].append(new_shape)

        return new_annotations

    def process_single_image(self, image_path: str, json_path: str,
                             output_image_dir: str, output_label_dir: str,
                             return_patches: bool = False) -> Optional[List[Dict]]:
        """
        处理单张图像

        Args:
            image_path: 输入图像路径
            json_path: 输入标注文件路径
            output_image_dir: 输出图像目录
            output_label_dir: 输出标注目录
            return_patches: 是否返回处理后的patches（用于可视化）

        Returns:
            如果return_patches为True，返回patches列表，否则返回None
        """
        # 确保输出目录存在
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)
        Path(output_label_dir).mkdir(parents=True, exist_ok=True)

        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None

        # 读取标注
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        # 确定滑动窗口大小（较短边的一半）
        #TODO: 短边一半太小
        h, w = image.shape[:2]
        window_size = min(640,  min(h, w))
        window_size = (window_size, window_size)

        # 计算步长（基于重叠率）
        stride = (int(window_size[0] * (1 - self.overlap_ratio)),
                  int(window_size[1] * (1 - self.overlap_ratio)))

        # 滑动窗口裁剪
        patches = self.sliding_window_crop(image, window_size, stride)

        # 用于返回的patches信息
        processed_patches = [] if return_patches else None

        # 处理每个patch
        base_name = Path(image_path).stem
        for i, patch_info in enumerate(patches):
            # 图像增强
            enhanced_patch = self.enhance_image(patch_info['patch'])

            # 调整标注
            x, y = patch_info['position']
            w, h = patch_info['size']
            adjusted_annotations = self.adjust_annotations(annotations, x, y, w, h)

            # 生成文件名
            patch_name = f"{base_name}_patch_{i:04d}"

            # 保存图像
            image_save_path = Path(output_image_dir) / f"{patch_name}.jpg"
            cv2.imwrite(str(image_save_path), enhanced_patch, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 保存标注
            label_save_path = Path(output_label_dir) / f"{patch_name}.json"
            with open(label_save_path, 'w', encoding='utf-8') as f:
                json.dump(adjusted_annotations, f, ensure_ascii=False, indent=2)

            # 如果需要返回patches信息
            if return_patches:
                processed_patches.append({
                    'original_patch': patch_info['patch'],
                    'enhanced_patch': enhanced_patch,
                    'position': patch_info['position'],
                    'size': patch_info['size'],
                    'annotations': adjusted_annotations,
                    'patch_name': patch_name
                })

            # 更新统计信息
            self.total_patches += 1
            if len(adjusted_annotations['shapes']) > 0:
                self.patches_with_defects += 1
            else:
                self.patches_without_defects += 1

        return processed_patches

    def process_batch(self, image_paths: List[str], json_paths: List[str],
                      output_image_dir: str, output_label_dir: str):
        """
        批量处理图像

        Args:
            image_paths: 图像路径列表
            json_paths: 对应的标注文件路径列表
            output_image_dir: 输出图像目录
            output_label_dir: 输出标注目录
        """
        # 重置统计信息
        self.total_patches = 0
        self.patches_with_defects = 0
        self.patches_without_defects = 0

        # 确保输入列表长度相同
        assert len(image_paths) == len(json_paths), "图像和标注文件数量不匹配"

        # 处理每张图像
        for image_path, json_path in tqdm(zip(image_paths, json_paths),
                                          total=len(image_paths),
                                          desc="处理图像"):
            self.process_single_image(image_path, json_path,
                                      output_image_dir, output_label_dir)

    def balance_dataset(self, label_dir: str, image_dir: str, image_ext: str = '.jpg'):
        """
        平衡数据集，确保有缺陷和无缺陷的样本数量相等

        Args:
            label_dir: 标注文件目录
            image_dir: 图像文件目录
            image_ext: 图像文件扩展名
        """
        print("\n平衡数据集...")

        # 收集所有无缺陷的patches
        no_defect_patches = []

        label_path = Path(label_dir)
        for json_file in label_path.rglob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if len(data.get('shapes', [])) == 0:
                # 找到对应的图像文件
                relative_path = json_file.relative_to(label_path)
                image_file = Path(image_dir) / relative_path.with_suffix(image_ext)

                no_defect_patches.append({
                    'json_path': json_file,
                    'image_path': image_file
                })

        # 随机选择要保留的无缺陷patches
        num_to_keep = min(self.patches_with_defects, len(no_defect_patches))
        patches_to_keep = random.sample(no_defect_patches, num_to_keep)
        patches_to_remove = [p for p in no_defect_patches if p not in patches_to_keep]

        # 删除多余的无缺陷patches
        for patch in patches_to_remove:
            if patch['json_path'].exists():
                patch['json_path'].unlink()
            if patch['image_path'].exists():
                patch['image_path'].unlink()

        print(f"保留了 {num_to_keep} 个无缺陷样本，删除了 {len(patches_to_remove)} 个")

        # 更新统计信息
        self.patches_without_defects = num_to_keep

    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        return {
            'total_patches': self.total_patches,
            'patches_with_defects': self.patches_with_defects,
            'patches_without_defects': self.patches_without_defects
        }


def process_weld_dataset(input_base_dir: str, output_base_dir: str,
                         overlap_ratio: float = 0.5, balance: bool = True):
    """
    处理焊缝数据集的包装函数

    Args:
        input_base_dir: 输入根目录，包含crop_weld_images和crop_weld_jsons
        output_base_dir: 输出根目录
        overlap_ratio: 滑动窗口重叠率
        balance: 是否平衡数据集
    """
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    # 创建输出目录
    output_image_base = output_base / 'images'
    output_label_base = output_base / 'labels'

    # 创建预处理器
    preprocessor = WeldImagePreprocessor(overlap_ratio=overlap_ratio)

    # 收集所有需要处理的文件
    all_image_paths = []
    all_json_paths = []
    all_output_image_dirs = []
    all_output_label_dirs = []

    # 遍历所有焊缝类型和子类型
    for weld_type in ['L', 'T']:
        for sub_type in ['1', '2']:
            image_dir = input_base / 'crop_weld_images' / weld_type / sub_type
            json_dir = input_base / 'crop_weld_jsons' / weld_type / sub_type

            if not image_dir.exists():
                continue

            # 创建输出目录
            output_image_dir = output_image_base / weld_type / sub_type
            output_label_dir = output_label_base / weld_type / sub_type
            output_image_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir.mkdir(parents=True, exist_ok=True)

            # 获取所有图像文件
            image_files = sorted(list(image_dir.glob('*.tif')))

            for image_file in image_files:
                json_file = json_dir / f"{image_file.stem}.json"

                if json_file.exists():
                    all_image_paths.append(str(image_file))
                    all_json_paths.append(str(json_file))
                    all_output_image_dirs.append(str(output_image_dir))
                    all_output_label_dirs.append(str(output_label_dir))
                else:
                    print(f"警告: 找不到标注文件 {json_file}")

    # 批量处理所有图像
    print(f"找到 {len(all_image_paths)} 张图像待处理")

    for i, (img_path, json_path, out_img_dir, out_lbl_dir) in enumerate(
            tqdm(zip(all_image_paths, all_json_paths, all_output_image_dirs, all_output_label_dirs),
                 total=len(all_image_paths), desc="处理进度")):
        preprocessor.process_single_image(img_path, json_path, out_img_dir, out_lbl_dir)

    # 打印统计信息
    stats = preprocessor.get_statistics()
    print("\n处理完成！")
    print(f"总patch数: {stats['total_patches']}")
    print(f"有缺陷的patch数: {stats['patches_with_defects']}")
    print(f"无缺陷的patch数: {stats['patches_without_defects']}")

    # 平衡数据集
    if balance:
        preprocessor.balance_dataset(str(output_label_base), str(output_image_base))

        # 最终统计
        final_stats = preprocessor.get_statistics()
        print("\n最终数据集统计:")
        print(f"有缺陷的patch数: {final_stats['patches_with_defects']}")
        print(f"无缺陷的patch数: {final_stats['patches_without_defects']}")
        print(f"总patch数: {final_stats['patches_with_defects'] + final_stats['patches_without_defects']}")


def main():
    """主函数"""
    # 设置输入输出路径
    input_dir = "/home/num2/datasets/Xray/crop_weld_data"  # 当前目录，包含crop_weld_images和crop_weld_jsons
    output_dir = "./preprocessed_data2"  # 输出目录

    # 处理数据集
    process_weld_dataset(input_dir, output_dir, overlap_ratio=0.5, balance=False)


if __name__ == "__main__":
    main()