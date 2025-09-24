import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import shutil
import sys

# 导入labelme2yolo模块
from convert.labelme2yolo import Labelme2YOLO


class WeldImagePreprocessor:
    """焊缝X射线图像预处理器，实现滑动窗口裁剪、图像增强和标签格式转换"""

    def __init__(self, overlap_ratio: float = 0.5, enhance_mode: str = 'original',
                 label_format: str = 'labelme', to_seg: bool = False,
                 filter_label: str = None, unify_to_crack: bool = False):
        """
        初始化预处理器

        Args:
            overlap_ratio: 滑动窗口重叠率，默认0.5（50%）
            enhance_mode: 增强模式，可选值:
                - 'original': 原始方法（直方图均衡 + CLAHE）
                - 'windowing': 窗宽窗位方法（自适应窗口映射）
            label_format: 标签格式，可选值:
                - 'labelme': 保持LabelMe JSON格式（默认）
                - 'yolo': 转换为YOLO格式
            to_seg: 是否为分割任务（仅在label_format='yolo'时有效）
            filter_label: 要过滤的标签名称（仅在label_format='yolo'时有效）
            unify_to_crack: 是否将所有标签统一为"crack"（仅在label_format='yolo'时有效）
        """
        self.overlap_ratio = overlap_ratio

        # 验证增强模式
        if enhance_mode not in ['original', 'windowing']:
            raise ValueError(f"Invalid enhance_mode: {enhance_mode}. Must be 'original' or 'windowing'")
        self.enhance_mode = enhance_mode

        # 验证标签格式
        if label_format not in ['labelme', 'yolo']:
            raise ValueError(f"Invalid label_format: {label_format}. Must be 'labelme' or 'yolo'")
        self.label_format = label_format
        self.to_seg = to_seg
        self.filter_label = filter_label
        self.unify_to_crack = unify_to_crack

        # 统计信息
        self.total_patches = 0
        self.patches_with_defects = 0
        self.patches_without_defects = 0

        # 用于存储临时labelme格式文件路径（当需要转换为YOLO时）
        self.temp_labelme_dir = None

        # 打印初始化信息
        print(f"WeldImagePreprocessor initialized with:")
        print(f"  - Overlap ratio: {self.overlap_ratio}")
        print(f"  - Enhancement mode: {self.enhance_mode}")
        print(f"  - Label format: {self.label_format}")
        if self.label_format == 'yolo':
            print(f"  - Segmentation mode: {self.to_seg}")
            if self.filter_label:
                print(f"  - Filter label: {self.filter_label}")
            if self.unify_to_crack:
                print(f"  - Unify to crack: {self.unify_to_crack}")

    def apply_window_level(self, image: np.ndarray, window_width: int,
                           window_level: int, output_bits: int = 8) -> np.ndarray:
        """
        应用窗宽窗位变换（从windowing.py移植）

        Args:
            image: 输入图像（float32）
            window_width: 窗宽
            window_level: 窗位
            output_bits: 输出位深度（8或16）

        Returns:
            变换后的图像
        """
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2

        output = np.zeros_like(image)

        if output_bits == 8:
            max_val = 255
            dtype = np.uint8
        else:
            max_val = 65535
            dtype = np.uint16

        # 窗口内的像素进行线性映射
        mask = (image >= window_min) & (image <= window_max)
        output[mask] = ((image[mask] - window_min) / window_width * max_val)

        # 窗口外的像素
        output[image < window_min] = 0
        output[image > window_max] = max_val

        return output.astype(dtype)

    def auto_window_level(self, image: np.ndarray) -> Tuple[int, int]:
        """
        自动计算窗宽窗位（基于统计信息）

        Args:
            image: 输入图像

        Returns:
            (window_width, window_level)
        """
        # 转换为float32进行计算
        img_float = image.astype(np.float32)

        # 计算统计信息
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        img_mean = np.mean(img_float)
        img_std = np.std(img_float)

        # 使用均值作为窗位，4倍标准差作为窗宽（与windowing.py保持一致）
        window_level = int(img_mean)
        window_width = int(min(4 * img_std, img_max - img_min))

        # 确保窗宽至少为1
        window_width = max(1, window_width)

        return window_width, window_level

    def enhance_image_windowing(self, image: np.ndarray) -> np.ndarray:
        """
        使用窗宽窗位方法增强图像

        Args:
            image: 输入图像（可能是16位）

        Returns:
            增强后的8位3通道图像
        """
        # 确保图像是float32类型
        if image.dtype == np.uint16:
            img_float = image.astype(np.float32)
        elif image.dtype == np.uint8:
            img_float = image.astype(np.float32)
        else:
            img_float = image

        # 自动计算窗宽窗位
        window_width, window_level = self.auto_window_level(img_float)

        # 应用窗宽窗位变换，输出8位图像
        enhanced_8bit = self.apply_window_level(img_float, window_width, window_level, 8)

        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(enhanced_8bit)
        # 转换为3通道图像
        image_3ch = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)

        return image_3ch

    def enhance_image_original(self, image: np.ndarray) -> np.ndarray:
        """
        原始图像增强处理方法

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
        image_3ch = cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2BGR)

        return image_3ch

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像增强处理（根据模式选择方法）

        Args:
            image: 输入图像

        Returns:
            增强后的图像
        """
        if self.enhance_mode == 'windowing':
            return self.enhance_image_windowing(image)
        else:
            return self.enhance_image_original(image)

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
        # TODO: 短边一半太小
        h, w = image.shape[:2]
        window_size = min(640, min(h, w))
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
            # 图像增强（使用选定的模式）
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

            # 保存标注（LabelMe格式）
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

    def convert_to_yolo(self, base_dir: str, val_size: float = 0.2):
        """
        将LabelMe格式转换为YOLO格式

        Args:
            base_dir: 包含images和labels目录的基础目录
            val_size: 验证集比例
        """
        print("\n开始转换为YOLO格式...")

        # 创建临时目录存储所有labelme格式文件
        temp_dir = Path(base_dir) / 'temp_labelme'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有子目录的文件到临时目录
        images_base = Path(base_dir) / 'images'
        labels_base = Path(base_dir) / 'labels'

        print("收集文件到临时目录...")
        for weld_type in ['L', 'T']:
            for sub_type in ['1', '2']:
                # 图像文件
                img_dir = images_base / weld_type / sub_type
                if img_dir.exists():
                    for img_file in img_dir.glob('*.jpg'):
                        shutil.copy2(img_file, temp_dir / img_file.name)

                # 标签文件
                label_dir = labels_base / weld_type / sub_type
                if label_dir.exists():
                    for json_file in label_dir.glob('*.json'):
                        shutil.copy2(json_file, temp_dir / json_file.name)

        # 创建YOLO输出目录
        yolo_output_dir = Path(base_dir) / 'YOLODataset'
        if self.to_seg:
            yolo_output_dir = Path(base_dir) / 'YOLODataset_seg'

        # 调用labelme2yolo转换器
        print(f"转换标签格式到YOLO...")
        converter = Labelme2YOLO(
            json_dir=str(temp_dir),
            to_seg=self.to_seg,
            filter_label=self.filter_label,
            unify_to_crack=self.unify_to_crack,
            output_dir=str(yolo_output_dir)
        )

        # 执行转换
        converter.convert(val_size=val_size)

        # 清理临时目录
        print("清理临时文件...")
        shutil.rmtree(temp_dir)

        # 删除原始的images和labels目录
        if images_base.exists():
            shutil.rmtree(images_base)
        if labels_base.exists():
            shutil.rmtree(labels_base)

        print(f"YOLO格式数据集已保存到: {yolo_output_dir}")

    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        return {
            'total_patches': self.total_patches,
            'patches_with_defects': self.patches_with_defects,
            'patches_without_defects': self.patches_without_defects
        }


def process_weld_dataset(input_base_dir: str, output_base_dir: str,
                         overlap_ratio: float = 0.5, balance: bool = True,
                         enhance_mode: str = 'original', label_format: str = 'labelme',
                         to_seg: bool = False, val_size: float = 0.2,
                         filter_label: str = None, unify_to_crack: bool = False):
    """
    处理焊缝数据集的包装函数

    Args:
        input_base_dir: 输入根目录，包含crop_weld_images和crop_weld_jsons
        output_base_dir: 输出根目录
        overlap_ratio: 滑动窗口重叠率
        balance: 是否平衡数据集
        enhance_mode: 图像增强模式 ('original' 或 'windowing')
        label_format: 标签格式 ('labelme' 或 'yolo')
        to_seg: 是否为分割任务（仅在label_format='yolo'时有效）
        val_size: 验证集比例（仅在label_format='yolo'时有效）
        filter_label: 要过滤的标签名称（仅在label_format='yolo'时有效）
        unify_to_crack: 是否将所有标签统一为"crack"（仅在label_format='yolo'时有效）
    """
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    # 创建输出目录
    output_image_base = output_base / 'images'
    output_label_base = output_base / 'labels'

    # 创建预处理器
    preprocessor = WeldImagePreprocessor(
        overlap_ratio=overlap_ratio,
        enhance_mode=enhance_mode,
        label_format=label_format,
        to_seg=to_seg,
        filter_label=filter_label,
        unify_to_crack=unify_to_crack
    )

    print(f"\n使用增强模式: {enhance_mode}")
    if enhance_mode == 'original':
        print("  - 原始方法：直方图均衡 + CLAHE")
    else:
        print("  - 窗宽窗位方法：自适应窗口映射")

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

    # 如果需要转换为YOLO格式
    if label_format == 'yolo':
        preprocessor.convert_to_yolo(str(output_base), val_size=val_size)


def main():
    """主函数"""
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='焊缝X射线图像预处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
图像增强模式说明:
  original  - 原始方法：使用直方图均衡 + CLAHE (Contrast Limited Adaptive Histogram Equalization)
  windowing - 窗宽窗位方法：基于每个patch的统计信息自适应调整对比度

标签格式说明:
  labelme - 保持LabelMe JSON格式（默认）
  yolo    - 转换为YOLO格式，生成txt标签文件和dataset.yaml

YOLO特定参数:
  --seg           - 转换为分割数据集格式（多边形标注）
  --filter_label  - 过滤掉指定的标签
  --unify_to_crack - 将所有标签统一为"crack"类别

示例:
  # 使用原始方法处理数据集，保持LabelMe格式
  python WeldImagePreprocessor.py --input_dir ./data --output_dir ./output --mode original

  # 使用窗宽窗位方法处理数据集，转换为YOLO检测格式
  python WeldImagePreprocessor.py --input_dir ./data --output_dir ./output --mode windowing --label_format yolo

  # 转换为YOLO分割格式，设置验证集比例为30%
  python WeldImagePreprocessor.py --input_dir ./data --output_dir ./output --label_format yolo --seg --val_size 0.3

  # 统一所有标签为crack类别
  python WeldImagePreprocessor.py --input_dir ./data --output_dir ./output --label_format yolo --unify_to_crack

  # 过滤掉"焊缝"标签，只保留缺陷标签
  python WeldImagePreprocessor.py --input_dir ./data --output_dir ./output --label_format yolo --filter_label 焊缝
        """)

    parser.add_argument('--input_dir', type=str,
                        default="/home/lenovo/code/CHT/datasets/Xray/opensource/crop_weld_data_part",
                        help='输入目录路径，包含crop_weld_images和crop_weld_jsons')
    parser.add_argument('--output_dir', type=str, default="./preprocessed_data2/part",
                        help='输出目录路径')
    parser.add_argument('--mode', type=str, choices=['original', 'windowing'], default='windowing',
                        help='图像增强模式: original(直方图均衡+CLAHE) 或 windowing(窗宽窗位)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='滑动窗口重叠率 (0.0-1.0)，默认0.5')
    parser.add_argument('--balance', action='store_true', default=False,
                        help='是否平衡数据集（使有缺陷和无缺陷样本数相等）')
    parser.add_argument('--no-balance', dest='balance', action='store_false',
                        help='不平衡数据集')
    parser.add_argument('--label_format', type=str, choices=['labelme', 'yolo'], default='labelme',
                        help='标签格式: labelme(JSON格式) 或 yolo(TXT格式)')
    parser.add_argument('--seg', action='store_true', default=False,
                        help='转换为YOLOv5 v7.0分割数据集格式（仅在label_format=yolo时有效）')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例（仅在label_format=yolo时有效），默认0.2')
    parser.add_argument('--filter_label', type=str, default=None,
                        help='要过滤的标签名称，例如"焊缝"（仅在label_format=yolo时有效）')
    parser.add_argument('--unify_to_crack', action='store_true', default=False,
                        help='将所有标签统一为"crack"类别（仅在label_format=yolo时有效）')

    args = parser.parse_args()

    # 打印配置信息
    print("=" * 60)
    print("焊缝X射线图像预处理")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"增强模式: {args.mode}")
    print(f"重叠率: {args.overlap}")
    print(f"平衡数据集: {args.balance}")
    print(f"标签格式: {args.label_format}")
    if args.label_format == 'yolo':
        print(f"分割模式: {args.seg}")
        print(f"验证集比例: {args.val_size}")
        if args.filter_label:
            print(f"过滤标签: {args.filter_label}")
        if args.unify_to_crack:
            print(f"统一为crack类别: 是")
    print("=" * 60)

    # 处理数据集
    process_weld_dataset(
        args.input_dir,
        args.output_dir,
        overlap_ratio=args.overlap,
        balance=args.balance,
        enhance_mode=args.mode,
        label_format=args.label_format,
        to_seg=args.seg,
        val_size=args.val_size,
        filter_label=args.filter_label,
        unify_to_crack=args.unify_to_crack
    )


if __name__ == "__main__":
    main()