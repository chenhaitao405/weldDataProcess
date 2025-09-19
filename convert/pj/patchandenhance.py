"""
YOLO ROI数据滑动窗口处理脚本（优化版）
对yolo_roi_extractor.py输出的YOLO格式数据进行滑动窗口裁剪和图像增强
支持检测(det)和分割(seg)两种模式
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import yaml
import random

# 导入WeldImagePreprocessor类
import sys
sys.path.append('./convert')
from WeldImagePreprocessor import WeldImagePreprocessor


class YOLOSlidingWindowProcessor(WeldImagePreprocessor):
    """扩展WeldImagePreprocessor以支持YOLO格式的输入和输出"""

    def __init__(self, overlap_ratio: float = 0.5, enhance_mode: str = 'original',
                 label_mode: str = 'det', min_bbox_ratio: float = 0.01,
                 min_polygon_area_ratio: float = 0.001):
        """
        初始化处理器

        Args:
            overlap_ratio: 滑动窗口重叠率
            enhance_mode: 增强模式 ('original' 或 'windowing')
            label_mode: 标签模式 ('det' 检测框 或 'seg' 分割多边形)
            min_bbox_ratio: 最小边界框尺寸比例（相对于窗口大小）
            min_polygon_area_ratio: 最小多边形面积比例（相对于窗口面积）
        """
        super().__init__(overlap_ratio=overlap_ratio, enhance_mode=enhance_mode)
        self.label_mode = label_mode
        self.min_bbox_ratio = min_bbox_ratio
        self.min_polygon_area_ratio = min_polygon_area_ratio

    def read_yolo_labels(self, label_path: str) -> List[List[float]]:
        """
        读取YOLO格式的标签文件

        Args:
            label_path: 标签文件路径

        Returns:
            标签列表，根据label_mode不同：
            - det模式: [class_id, x_center, y_center, width, height]
            - seg模式: [class_id, x1, y1, x2, y2, ...]
        """
        labels = []
        if not os.path.exists(label_path):
            return labels

        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:  # 跳过空行
                    continue

                try:
                    if self.label_mode == 'det':
                        # 检测模式：需要正好5个值（class_id + bbox）
                        if len(parts) == 5:
                            label = [float(x) for x in parts]
                            # 验证边界框值的合理性
                            if all(0 <= label[i] <= 1 for i in range(1, 5)):
                                labels.append(label)
                            else:
                                print(f"警告: {label_path} 第{line_num}行包含无效的边界框值")

                    elif self.label_mode == 'seg':
                        # 分割模式：需要至少7个值（class_id + 至少3个点 = 1 + 6）
                        if len(parts) >= 7:
                            # 确保除了class_id外，剩余的是成对的坐标
                            if (len(parts) - 1) % 2 == 0:
                                label = [float(x) for x in parts]
                                # 验证所有坐标值的合理性
                                coords_valid = all(0 <= label[i] <= 1 for i in range(1, len(label)))
                                if coords_valid:
                                    labels.append(label)
                                else:
                                    print(f"警告: {label_path} 第{line_num}行包含无效的坐标值")
                            else:
                                print(f"警告: {label_path} 第{line_num}行坐标数量不成对")

                except ValueError as e:
                    print(f"警告: {label_path} 第{line_num}行解析失败: {e}")

        return labels

    def save_yolo_labels(self, labels: List[List[float]], output_path: str):
        """
        保存YOLO格式的标签文件

        Args:
            labels: 标签列表
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for label in labels:
                if len(label) == 0:
                    continue

                if self.label_mode == 'det' and len(label) >= 5:
                    # 检测模式：class_id为整数，坐标保留6位小数
                    class_id = int(label[0])
                    coords = ' '.join([f"{x:.6f}" for x in label[1:5]])
                    label_str = f"{class_id} {coords}"

                elif self.label_mode == 'seg' and len(label) >= 7:
                    # 分割模式：class_id为整数，坐标保留6位小数
                    class_id = int(label[0])
                    coords = ' '.join([f"{x:.6f}" for x in label[1:]])
                    label_str = f"{class_id} {coords}"

                else:
                    continue  # 跳过无效标签

                f.write(label_str + '\n')

    def convert_yolo_to_pixel(self, label: List[float], img_width: int, img_height: int) -> Tuple:
        """
        将YOLO格式的归一化坐标转换为像素坐标

        Args:
            label: [class_id, x_center, y_center, width, height]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            (class_id, x1, y1, x2, y2) 像素坐标
        """
        class_id = int(label[0])
        x_center = label[1] * img_width
        y_center = label[2] * img_height
        width = label[3] * img_width
        height = label[4] * img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        return class_id, x1, y1, x2, y2

    def convert_pixel_to_yolo(self, class_id: int, x1: int, y1: int, x2: int, y2: int,
                              img_width: int, img_height: int) -> List[float]:
        """
        将像素坐标转换为YOLO格式的归一化坐标

        Args:
            class_id: 类别ID
            x1, y1, x2, y2: 边界框的像素坐标
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            [class_id, x_center, y_center, width, height] YOLO格式
        """
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return [class_id, x_center, y_center, width, height]

    def calculate_polygon_area(self, points: List[float]) -> float:
        """
        计算多边形面积（使用Shoelace公式）

        Args:
            points: 归一化的多边形顶点坐标 [x1, y1, x2, y2, ...]

        Returns:
            归一化的面积
        """
        if len(points) < 6:  # 至少需要3个点
            return 0.0

        area = 0.0
        n = len(points) // 2

        for i in range(n):
            x1 = points[2 * i]
            y1 = points[2 * i + 1]
            x2 = points[2 * ((i + 1) % n)]
            y2 = points[2 * ((i + 1) % n) + 1]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2.0

    def clip_polygon_to_window(self, polygon: List[float], window_bounds: Tuple[float, float, float, float]) -> List[float]:
        """
        使用Sutherland-Hodgman算法裁剪多边形到窗口内

        Args:
            polygon: 多边形顶点坐标 [x1, y1, x2, y2, ...]
            window_bounds: (x_min, y_min, x_max, y_max) 窗口边界

        Returns:
            裁剪后的多边形顶点坐标
        """
        if len(polygon) < 6:
            return []

        x_min, y_min, x_max, y_max = window_bounds

        # 转换为点列表
        points = []
        for i in range(0, len(polygon), 2):
            points.append([polygon[i], polygon[i + 1]])

        # 对四条边依次进行裁剪
        edges = [
            (x_min, 0, 'left'),   # 左边界
            (x_max, 0, 'right'),  # 右边界
            (y_min, 1, 'bottom'), # 下边界
            (y_max, 1, 'top')     # 上边界
        ]

        for edge_val, coord_idx, edge_type in edges:
            if not points:
                break

            new_points = []
            for i in range(len(points)):
                curr_point = points[i]
                prev_point = points[i - 1]

                if edge_type in ['left', 'right']:
                    curr_inside = (curr_point[0] >= x_min) if edge_type == 'left' else (curr_point[0] <= x_max)
                    prev_inside = (prev_point[0] >= x_min) if edge_type == 'left' else (prev_point[0] <= x_max)
                else:
                    curr_inside = (curr_point[1] >= y_min) if edge_type == 'bottom' else (curr_point[1] <= y_max)
                    prev_inside = (prev_point[1] >= y_min) if edge_type == 'bottom' else (prev_point[1] <= y_max)

                if curr_inside:
                    if not prev_inside:
                        # 计算交点
                        t = 0.0
                        if edge_type in ['left', 'right']:
                            if curr_point[0] != prev_point[0]:
                                t = (edge_val - prev_point[0]) / (curr_point[0] - prev_point[0])
                        else:
                            if curr_point[1] != prev_point[1]:
                                t = (edge_val - prev_point[1]) / (curr_point[1] - prev_point[1])

                        intersection = [
                            prev_point[0] + t * (curr_point[0] - prev_point[0]),
                            prev_point[1] + t * (curr_point[1] - prev_point[1])
                        ]
                        new_points.append(intersection)
                    new_points.append(curr_point)
                elif prev_inside:
                    # 计算交点
                    t = 0.0
                    if edge_type in ['left', 'right']:
                        if curr_point[0] != prev_point[0]:
                            t = (edge_val - prev_point[0]) / (curr_point[0] - prev_point[0])
                    else:
                        if curr_point[1] != prev_point[1]:
                            t = (edge_val - prev_point[1]) / (curr_point[1] - prev_point[1])

                    intersection = [
                        prev_point[0] + t * (curr_point[0] - prev_point[0]),
                        prev_point[1] + t * (curr_point[1] - prev_point[1])
                    ]
                    new_points.append(intersection)

            points = new_points

        # 转换回扁平列表
        result = []
        for point in points:
            result.extend([
                max(0.0, min(1.0, point[0])),
                max(0.0, min(1.0, point[1]))
            ])

        return result

    def adjust_yolo_labels_for_crop(self, labels: List[List[float]],
                                   crop_x: int, crop_y: int,
                                   crop_w: int, crop_h: int,
                                   original_w: int, original_h: int) -> List[List[float]]:
        """
        调整YOLO标签以适应裁剪后的图像

        Args:
            labels: 原始YOLO标签列表
            crop_x, crop_y: 裁剪区域左上角坐标
            crop_w, crop_h: 裁剪区域尺寸
            original_w, original_h: 原始图像尺寸

        Returns:
            调整后的标签列表
        """
        adjusted_labels = []

        if self.label_mode == 'det':
            # 检测模式：处理边界框
            for label in labels:
                if len(label) < 5:
                    continue

                # 转换为像素坐标
                class_id, x1, y1, x2, y2 = self.convert_yolo_to_pixel(label, original_w, original_h)

                # 计算与裁剪区域的交集
                intersect_x1 = max(x1, crop_x)
                intersect_y1 = max(y1, crop_y)
                intersect_x2 = min(x2, crop_x + crop_w)
                intersect_y2 = min(y2, crop_y + crop_h)

                # 检查是否有交集
                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    # 调整坐标到裁剪图像的坐标系
                    new_x1 = max(0, intersect_x1 - crop_x)
                    new_y1 = max(0, intersect_y1 - crop_y)
                    new_x2 = min(crop_w, intersect_x2 - crop_x)
                    new_y2 = min(crop_h, intersect_y2 - crop_y)

                    # 转换回YOLO格式
                    new_label = self.convert_pixel_to_yolo(
                        class_id, new_x1, new_y1, new_x2, new_y2, crop_w, crop_h
                    )

                    # 过滤掉太小的边界框
                    if new_label[3] > self.min_bbox_ratio and new_label[4] > self.min_bbox_ratio:
                        adjusted_labels.append(new_label)

        elif self.label_mode == 'seg':
            # 分割模式：处理多边形
            for label in labels:
                if len(label) < 7:  # class_id + 至少3个点
                    continue

                class_id = int(label[0])
                polygon_points = label[1:]  # 归一化的x,y点对

                # 转换为像素坐标
                pixel_points = []
                for i in range(0, len(polygon_points), 2):
                    if i + 1 >= len(polygon_points):
                        break
                    x = polygon_points[i] * original_w
                    y = polygon_points[i + 1] * original_h
                    pixel_points.extend([x, y])

                # 调整到裁剪窗口坐标系
                adjusted_pixel_points = []
                for i in range(0, len(pixel_points), 2):
                    x = pixel_points[i] - crop_x
                    y = pixel_points[i + 1] - crop_y
                    adjusted_pixel_points.extend([x, y])

                # 转换为归一化坐标（相对于裁剪窗口）
                norm_points = []
                for i in range(0, len(adjusted_pixel_points), 2):
                    norm_x = adjusted_pixel_points[i] / crop_w
                    norm_y = adjusted_pixel_points[i + 1] / crop_h
                    norm_points.extend([norm_x, norm_y])

                # 裁剪多边形到窗口内
                clipped_points = self.clip_polygon_to_window(norm_points, (0.0, 0.0, 1.0, 1.0))

                if len(clipped_points) >= 6:  # 至少3个点
                    # 计算裁剪后的面积
                    area = self.calculate_polygon_area(clipped_points)

                    # 只保留面积足够大的多边形
                    if area > self.min_polygon_area_ratio:
                        new_label = [class_id] + clipped_points
                        adjusted_labels.append(new_label)

        return adjusted_labels

    def process_single_image_yolo(self, image_path: str, label_path: str,
                                 output_image_dir: str, output_label_dir: str,
                                 window_size: Tuple[int, int] = None) -> Dict:
        """
        处理单张YOLO格式标注的图像

        Args:
            image_path: 输入图像路径
            label_path: YOLO格式标签文件路径
            output_image_dir: 输出图像目录
            output_label_dir: 输出标签目录
            window_size: 窗口大小，默认为(640, 640)

        Returns:
            处理统计信息
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return {'processed': 0, 'with_defects': 0, 'without_defects': 0, 'error': 1}

        h, w = image.shape[:2]

        # 读取YOLO标签
        labels = self.read_yolo_labels(label_path)

        # 确定滑动窗口大小
        if window_size is None:
            window_size = min(640, min(h, w))
            window_size = (window_size, window_size)

        # 计算步长
        stride = (int(window_size[0] * (1 - self.overlap_ratio)),
                 int(window_size[1] * (1 - self.overlap_ratio)))

        # 滑动窗口裁剪
        patches = self.sliding_window_crop(image, window_size, stride)

        # 统计信息
        stats = {'processed': 0, 'with_defects': 0, 'without_defects': 0, 'error': 0}

        # 处理每个patch
        base_name = Path(image_path).stem
        for i, patch_info in enumerate(patches):
            try:
                # 图像增强
                enhanced_patch = self.enhance_image(patch_info['patch'])

                # 调整标签
                x, y = patch_info['position']
                patch_w, patch_h = patch_info['size']
                adjusted_labels = self.adjust_yolo_labels_for_crop(
                    labels, x, y, patch_w, patch_h, w, h
                )

                # 生成文件名
                patch_name = f"{base_name}_patch_{i:04d}"

                # 保存图像
                image_save_path = Path(output_image_dir) / f"{patch_name}.jpg"
                cv2.imwrite(str(image_save_path), enhanced_patch, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 保存标签
                label_save_path = Path(output_label_dir) / f"{patch_name}.txt"
                self.save_yolo_labels(adjusted_labels, str(label_save_path))

                # 更新统计
                stats['processed'] += 1
                if len(adjusted_labels) > 0:
                    stats['with_defects'] += 1
                else:
                    stats['without_defects'] += 1

            except Exception as e:
                print(f"处理patch {i} 时出错: {e}")
                stats['error'] += 1

        return stats


def process_yolo_roi_dataset(input_dir: str, output_dir: str,
                            overlap_ratio: float = 0.5,
                            enhance_mode: str = 'original',
                            label_mode: str = 'det',
                            window_size: Tuple[int, int] = None,
                            balance: bool = False,
                            balance_ratio: float = 1.0):
    """
    处理YOLO ROI数据集

    Args:
        input_dir: 输入目录（包含images/和labels/子目录）
        output_dir: 输出目录
        overlap_ratio: 滑动窗口重叠率
        enhance_mode: 增强模式 ('original' 或 'windowing')
        label_mode: 标签模式 ('det' 检测框 或 'seg' 分割多边形)
        window_size: 窗口大小，默认为(640, 640)
        balance: 是否平衡数据集
        balance_ratio: 平衡比例（无缺陷/有缺陷）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 验证输入目录结构
    if not (input_path / 'images').exists() or not (input_path / 'labels').exists():
        print(f"错误：输入目录必须包含images/和labels/子目录")
        return

    # 创建处理器
    processor = YOLOSlidingWindowProcessor(
        overlap_ratio=overlap_ratio,
        enhance_mode=enhance_mode,
        label_mode=label_mode
    )

    print(f"=" * 60)
    print(f"YOLO ROI数据集滑动窗口处理")
    print(f"=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"重叠率: {overlap_ratio}")
    print(f"增强模式: {enhance_mode}")
    print(f"标签模式: {label_mode}")
    print(f"窗口大小: {window_size if window_size else '自动'}")
    print(f"平衡数据集: {balance}")
    if balance:
        print(f"平衡比例: {balance_ratio}")
    print(f"=" * 60)

    # 统计信息
    total_stats = {
        'train': {'processed': 0, 'with_defects': 0, 'without_defects': 0, 'error': 0},
        'val': {'processed': 0, 'with_defects': 0, 'without_defects': 0, 'error': 0}
    }

    # 处理train和val数据
    for split in ['train', 'val']:
        print(f"\n处理{split}数据集...")

        input_image_dir = input_path / 'images' / split
        input_label_dir = input_path / 'labels' / split
        output_image_dir = output_path / 'images' / split
        output_label_dir = output_path / 'labels' / split

        if not input_image_dir.exists():
            print(f"跳过{split}（不存在）")
            continue

        # 创建输出目录
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_files = list(input_image_dir.glob('*.jpg')) + \
                     list(input_image_dir.glob('*.png')) + \
                     list(input_image_dir.glob('*.bmp'))

        print(f"找到{len(image_files)}张图像")

        # 处理每张图像
        for image_file in tqdm(image_files, desc=f"处理{split}"):
            label_file = input_label_dir / f"{image_file.stem}.txt"

            # 即使标签文件不存在也处理（可能是负样本）
            if not label_file.exists():
                label_file = Path("/dev/null")  # 使用空标签

            stats = processor.process_single_image_yolo(
                str(image_file),
                str(label_file),
                str(output_image_dir),
                str(output_label_dir),
                window_size=window_size
            )

            # 累加统计
            for key in stats:
                total_stats[split][key] += stats[key]

    # 复制并更新dataset.yaml
    input_yaml = input_path / 'dataset.yaml'
    output_yaml = output_path / 'dataset.yaml'

    if input_yaml.exists():
        with open(input_yaml, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # 更新路径为绝对路径
        yaml_data['train'] = str(output_path.absolute() / 'images' / 'train')
        yaml_data['val'] = str(output_path.absolute() / 'images' / 'val')

        # 添加处理信息
        yaml_data['preprocessing'] = {
            'window_size': list(window_size) if window_size else 'auto',
            'overlap_ratio': overlap_ratio,
            'enhance_mode': enhance_mode,
            'label_mode': label_mode
        }

        with open(output_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"\ndataset.yaml已保存到: {output_yaml}")

    # 打印统计信息
    print(f"\n" + "=" * 60)
    print("处理完成统计:")
    print(f"=" * 60)

    for split in ['train', 'val']:
        stats = total_stats[split]
        if stats['processed'] > 0:
            print(f"\n{split.upper()}数据集:")
            print(f"  总patches数: {stats['processed']}")
            print(f"  有缺陷patches: {stats['with_defects']} ({stats['with_defects']/stats['processed']*100:.1f}%)")
            print(f"  无缺陷patches: {stats['without_defects']} ({stats['without_defects']/stats['processed']*100:.1f}%)")
            if stats['error'] > 0:
                print(f"  错误数: {stats['error']}")

    total_processed = sum(s['processed'] for s in total_stats.values())
    total_with_defects = sum(s['with_defects'] for s in total_stats.values())
    total_without_defects = sum(s['without_defects'] for s in total_stats.values())
    total_errors = sum(s['error'] for s in total_stats.values())

    print(f"\n总计:")
    print(f"  总patches数: {total_processed}")
    print(f"  有缺陷patches: {total_with_defects} ({total_with_defects/max(1,total_processed)*100:.1f}%)")
    print(f"  无缺陷patches: {total_without_defects} ({total_without_defects/max(1,total_processed)*100:.1f}%)")
    if total_errors > 0:
        print(f"  错误数: {total_errors}")

    # 平衡数据集（如果需要）
    if balance and total_without_defects > 0 and total_with_defects > 0:
        print(f"\n开始平衡数据集...")
        target_ratio = balance_ratio
        target_no_defect_count = int(total_with_defects * target_ratio)

        if target_no_defect_count < total_without_defects:
            balance_yolo_dataset(output_path, target_no_defect_count)
        else:
            print(f"无需平衡：当前比例已满足要求")


def balance_yolo_dataset(dataset_path: Path, target_no_defect_count: int):
    """
    平衡YOLO数据集，删除多余的无缺陷样本

    Args:
        dataset_path: 数据集路径
        target_no_defect_count: 目标无缺陷样本数量
    """
    no_defect_files = []

    # 收集所有无缺陷的样本
    for split in ['train', 'val']:
        label_dir = dataset_path / 'labels' / split
        image_dir = dataset_path / 'images' / split

        if not label_dir.exists():
            continue

        for label_file in label_dir.glob('*.txt'):
            # 检查是否为空文件（无缺陷）
            with open(label_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:  # 无缺陷
                # 查找对应的图像文件
                image_files = []
                for ext in ['.jpg', '.png', '.bmp']:
                    img_file = image_dir / f"{label_file.stem}{ext}"
                    if img_file.exists():
                        image_files.append(img_file)

                if image_files:
                    no_defect_files.append({
                        'split': split,
                        'label': label_file,
                        'images': image_files
                    })

    # 按split分组统计
    split_counts = {'train': 0, 'val': 0}
    for f in no_defect_files:
        split_counts[f['split']] += 1

    # 按比例分配要保留的数量
    train_ratio = split_counts['train'] / max(1, sum(split_counts.values()))
    target_train = int(target_no_defect_count * train_ratio)
    target_val = target_no_defect_count - target_train

    # 分别处理train和val
    for split in ['train', 'val']:
        split_files = [f for f in no_defect_files if f['split'] == split]
        target = target_train if split == 'train' else target_val

        if len(split_files) > target:
            # 随机选择要保留的文件
            random.seed(42)  # 固定种子以确保可重现性
            files_to_keep = random.sample(split_files, target)
            files_to_remove = [f for f in split_files if f not in files_to_keep]

            # 删除多余文件
            for file_pair in files_to_remove:
                file_pair['label'].unlink()
                for img_file in file_pair['images']:
                    img_file.unlink()

            print(f"{split}: 删除了{len(files_to_remove)}个无缺陷样本，保留{len(files_to_keep)}个")

    print(f"平衡完成：总共保留{target_no_defect_count}个无缺陷样本")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLO ROI数据集滑动窗口处理工具（优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（检测模式）
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output
  
  # 分割模式
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --label_mode seg
  
  # 设置窗口大小为512x512
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --window_size 512 512
  
  # 设置重叠率为0.3
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --overlap 0.3
  
  # 使用窗宽窗位增强模式
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --mode windowing
  
  # 平衡数据集（1:1比例）
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --balance
  
  # 平衡数据集（1:2比例）
  python yolo_roi_sliding_processor.py --input_dir ./roi_dataset --output_dir ./sliding_output --balance --balance_ratio 2.0
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径（YOLO格式，包含images/和labels/）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--window_size', type=int, nargs=2, default=None,
                       help='窗口大小 [width height]，默认自动确定')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='滑动窗口重叠率 (0.0-1.0)，默认0.5')
    parser.add_argument('--mode', type=str, choices=['original', 'windowing'],
                       default='original',
                       help='图像增强模式: original(直方图均衡+CLAHE) 或 windowing(窗宽窗位)')
    parser.add_argument('--label_mode', type=str, choices=['det', 'seg'],
                       default='det',
                       help='标签模式: det(检测边界框) 或 seg(分割多边形)')
    parser.add_argument('--balance', action='store_true',
                       help='平衡数据集')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                       help='平衡比例（无缺陷/有缺陷），默认1.0')

    args = parser.parse_args()

    # 处理窗口大小参数
    window_size = tuple(args.window_size) if args.window_size else None

    # 处理数据集
    process_yolo_roi_dataset(
        args.input_dir,
        args.output_dir,
        overlap_ratio=args.overlap,
        enhance_mode=args.mode,
        label_mode=args.label_mode,
        window_size=window_size,
        balance=args.balance,
        balance_ratio=args.balance_ratio
    )


if __name__ == "__main__":
    main()