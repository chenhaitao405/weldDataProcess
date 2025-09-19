'''
YOLO ROI区域提取与标签重计算脚本
支持检测(det)和分割(seg)两种模式的标签处理

@author: Assistant
'''
import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


class YOLOROIExtractor:
    def __init__(self,
                 input_dir,
                 output_dir,
                 model_path,
                 mode='det',  # 'det' for detection, 'seg' for segmentation
                 roi_conf_threshold=0.25,
                 roi_iou_threshold=0.45,
                 padding_ratio=0.1):
        """
        初始化ROI提取器

        Args:
            input_dir: 输入的YOLO数据集目录（包含images和labels文件夹）
            output_dir: 输出的YOLO数据集目录
            model_path: YOLO模型权重路径
            mode: 'det' for detection mode, 'seg' for segmentation mode
            roi_conf_threshold: ROI检测的置信度阈值
            roi_iou_threshold: ROI检测的IOU阈值
            padding_ratio: ROI区域的padding比例（扩展边界框）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.mode = mode
        self.roi_conf_threshold = roi_conf_threshold
        self.roi_iou_threshold = roi_iou_threshold
        self.padding_ratio = padding_ratio

        # 加载YOLO模型
        self.model = YOLO(model_path)

        # 创建输出目录结构
        self._create_output_dirs()

    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs_to_create = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val'
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _read_yolo_labels(self, label_path):
        """
        读取YOLO格式的标签文件

        Returns:
            对于det模式: list of [class_id, x_center, y_center, width, height] (归一化坐标)
            对于seg模式: list of [class_id, x1, y1, x2, y2, ...] (归一化的多边形坐标点)
        """
        labels = []
        if not os.path.exists(label_path):
            return labels

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # 至少需要类别ID和最少的坐标信息
                    continue

                if self.mode == 'det':
                    # 检测模式：class_id, x_center, y_center, width, height
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
                else:  # seg mode
                    # 分割模式：class_id, x1, y1, x2, y2, x3, y3, ...
                    class_id = int(parts[0])
                    coords = []
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i + 1])
                            coords.extend([x, y])
                    if len(coords) >= 6:  # 至少3个点才能构成多边形
                        labels.append([class_id] + coords)
        return labels

    def _detect_roi(self, image_path):
        """
        使用YOLO模型检测ROI区域

        Returns:
            list of ROI边界框 [(x1, y1, x2, y2), ...]
        """
        results = self.model(image_path,
                            conf=self.roi_conf_threshold,
                            iou=self.roi_iou_threshold)

        roi_boxes = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
                for box in boxes:
                    x1, y1, x2, y2 = box
                    roi_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return roi_boxes

    def _add_padding(self, x1, y1, x2, y2, img_width, img_height):
        """为ROI区域添加padding"""
        width = x2 - x1
        height = y2 - y1

        # 计算padding
        pad_x = int(width * self.padding_ratio)
        pad_y = int(height * self.padding_ratio)

        # 添加padding并确保不超出图像边界
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        return x1_padded, y1_padded, x2_padded, y2_padded

    def _process_detection_label(self, label, roi_x1, roi_y1, roi_x2, roi_y2,
                                img_width, img_height, cropped_width, cropped_height):
        """
        处理检测模式的标签

        Args:
            label: [class_id, x_center, y_center, width, height] (归一化坐标)
            roi_*: ROI区域的像素坐标
            img_*: 原始图像尺寸
            cropped_*: 裁剪后图像尺寸

        Returns:
            新的标签 [class_id, x_center, y_center, width, height] 或 None
        """
        class_id, x_center, y_center, width, height = label

        # 转换为像素坐标
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height

        # 计算边界框
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # 计算与ROI的交集
        intersect_x1 = max(x1, roi_x1)
        intersect_y1 = max(y1, roi_y1)
        intersect_x2 = min(x2, roi_x2)
        intersect_y2 = min(y2, roi_y2)

        # 如果没有交集
        if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
            return None

        # 转换为相对于裁剪图像的坐标
        new_x1 = max(0, intersect_x1 - roi_x1)
        new_y1 = max(0, intersect_y1 - roi_y1)
        new_x2 = min(cropped_width, intersect_x2 - roi_x1)
        new_y2 = min(cropped_height, intersect_y2 - roi_y1)

        # 计算新的中心点和尺寸（归一化）
        new_x_center = (new_x1 + new_x2) / 2.0 / cropped_width
        new_y_center = (new_y1 + new_y2) / 2.0 / cropped_height
        new_width = (new_x2 - new_x1) / cropped_width
        new_height = (new_y2 - new_y1) / cropped_height

        # 过滤掉太小的边界框
        if new_width <= 0.01 or new_height <= 0.01:
            return None

        return [class_id, new_x_center, new_y_center, new_width, new_height]

    def _process_segmentation_label(self, label, roi_x1, roi_y1, roi_x2, roi_y2,
                                   img_width, img_height, cropped_width, cropped_height):
        """
        处理分割模式的标签

        Args:
            label: [class_id, x1, y1, x2, y2, ...] (归一化的多边形坐标)
            roi_*: ROI区域的像素坐标
            img_*: 原始图像尺寸
            cropped_*: 裁剪后图像尺寸

        Returns:
            新的标签 [class_id, x1, y1, x2, y2, ...] 或 None
        """
        class_id = label[0]
        points = label[1:]  # 获取所有坐标点

        # 转换为像素坐标并裁剪到ROI区域
        new_points = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x = points[i] * img_width
                y = points[i + 1] * img_height

                # 裁剪到ROI区域
                x_clipped = np.clip(x, roi_x1, roi_x2)
                y_clipped = np.clip(y, roi_y1, roi_y2)

                # 转换为相对于裁剪图像的坐标
                new_x = (x_clipped - roi_x1) / cropped_width
                new_y = (y_clipped - roi_y1) / cropped_height

                new_points.extend([new_x, new_y])

        # 检查多边形是否在ROI内（至少要有一部分在内）
        # 计算多边形的边界框
        x_coords = new_points[::2]
        y_coords = new_points[1::2]

        if not x_coords or not y_coords:
            return None

        poly_x_min = min(x_coords)
        poly_x_max = max(x_coords)
        poly_y_min = min(y_coords)
        poly_y_max = max(y_coords)

        # 如果多边形完全在ROI外或太小
        poly_width = poly_x_max - poly_x_min
        poly_height = poly_y_max - poly_y_min

        if poly_width <= 0.01 or poly_height <= 0.01:
            return None

        # 检查是否至少有部分在图像内
        if poly_x_max <= 0 or poly_x_min >= 1 or poly_y_max <= 0 or poly_y_min >= 1:
            return None

        return [class_id] + new_points

    def _process_single_image(self, image_path, label_path, split_type):
        """
        处理单张图像：检测ROI并裁剪保存

        Args:
            image_path: 图像路径
            label_path: 对应的标签文件路径
            split_type: 'train' 或 'val'
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Cannot read image {image_path}")
            return

        img_height, img_width = img.shape[:2]

        # 检测ROI区域
        roi_boxes = self._detect_roi(str(image_path))

        if not roi_boxes:
            print(f"Warning: No ROI detected in {image_path}")
            return

        # 读取原始标签
        original_labels = self._read_yolo_labels(label_path)

        # 处理每个ROI区域
        base_name = image_path.stem
        for roi_idx, (roi_x1, roi_y1, roi_x2, roi_y2) in enumerate(roi_boxes):
            # 添加padding
            roi_x1, roi_y1, roi_x2, roi_y2 = self._add_padding(
                roi_x1, roi_y1, roi_x2, roi_y2, img_width, img_height
            )

            # 裁剪图像
            cropped_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
            cropped_height, cropped_width = cropped_img.shape[:2]

            # 生成新的文件名
            new_img_name = f"{base_name}_roi_{roi_idx}.jpg"
            new_label_name = f"{base_name}_roi_{roi_idx}.txt"

            # 保存裁剪后的图像
            output_img_path = self.output_dir / 'images' / split_type / new_img_name
            cv2.imwrite(str(output_img_path), cropped_img)

            # 处理标签
            new_labels = []
            for label in original_labels:
                if self.mode == 'det':
                    new_label = self._process_detection_label(
                        label, roi_x1, roi_y1, roi_x2, roi_y2,
                        img_width, img_height, cropped_width, cropped_height
                    )
                else:  # seg mode
                    new_label = self._process_segmentation_label(
                        label, roi_x1, roi_y1, roi_x2, roi_y2,
                        img_width, img_height, cropped_width, cropped_height
                    )

                if new_label is not None:
                    new_labels.append(new_label)

            # 保存新的标签文件
            output_label_path = self.output_dir / 'labels' / split_type / new_label_name
            with open(output_label_path, 'w') as f:
                for label in new_labels:
                    if self.mode == 'det':
                        # 检测模式：格式化为固定小数位
                        label_str = f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}"
                    else:  # seg mode
                        # 分割模式：class_id + 所有坐标点
                        label_str = str(int(label[0]))
                        for coord in label[1:]:
                            label_str += f" {coord:.6f}"
                    f.write(label_str + '\n')

    def process_dataset(self):
        """处理整个数据集"""
        print(f"Processing dataset in {self.mode} mode...")

        # 处理训练集和验证集
        for split_type in ['train', 'val']:
            image_dir = self.input_dir / 'images' / split_type
            label_dir = self.input_dir / 'labels' / split_type

            if not image_dir.exists():
                print(f"Warning: {image_dir} does not exist")
                continue

            # 获取所有图像文件
            image_files = list(image_dir.glob('*.jpg')) + \
                         list(image_dir.glob('*.jpeg')) + \
                         list(image_dir.glob('*.png')) + \
                         list(image_dir.glob('*.bmp'))

            print(f"Processing {split_type} set: {len(image_files)} images")

            # 处理每张图像
            for image_path in tqdm(image_files, desc=f"Processing {split_type}"):
                # 构造对应的标签文件路径
                label_name = image_path.stem + '.txt'
                label_path = label_dir / label_name

                self._process_single_image(image_path, label_path, split_type)

        # 复制并更新dataset.yaml文件
        self._update_dataset_yaml()

    def _update_dataset_yaml(self):
        """更新dataset.yaml文件"""
        input_yaml = self.input_dir / 'dataset.yaml'
        output_yaml = self.output_dir / 'dataset.yaml'

        if input_yaml.exists():
            # 读取原始yaml文件
            with open(input_yaml, 'r') as f:
                yaml_content = f.read()

            # 更新路径
            yaml_lines = yaml_content.split('\n')
            new_lines = []
            for line in yaml_lines:
                if line.startswith('train:'):
                    new_lines.append(f'train: {str(self.output_dir / "images" / "train")}')
                elif line.startswith('val:'):
                    new_lines.append(f'val: {str(self.output_dir / "images" / "val")}')
                else:
                    new_lines.append(line)

            # 写入新的yaml文件
            with open(output_yaml, 'w') as f:
                f.write('\n'.join(new_lines))

            print(f"Dataset.yaml saved to: {output_yaml}")
        else:
            print(f"Warning: {input_yaml} not found")


def main():
    parser = argparse.ArgumentParser(description='Extract ROI regions from YOLO dataset using YOLO model')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input YOLO dataset directory (containing images/ and labels/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for ROI dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--mode', type=str, choices=['det', 'seg'], default='det',
                        help='Dataset mode: det (detection) or seg (segmentation)')
    parser.add_argument('--roi_conf', type=float, default=0.25,
                        help='Confidence threshold for ROI detection (default: 0.25)')
    parser.add_argument('--roi_iou', type=float, default=0.45,
                        help='IOU threshold for ROI detection (default: 0.45)')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Padding ratio for ROI regions (default: 0.1)')

    args = parser.parse_args()

    # 创建ROI提取器
    extractor = YOLOROIExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mode=args.mode,
        roi_conf_threshold=args.roi_conf,
        roi_iou_threshold=args.roi_iou,
        padding_ratio=args.padding
    )

    # 处理数据集
    print(f"Starting ROI extraction in {args.mode} mode...")
    extractor.process_dataset()
    print("ROI extraction completed!")


if __name__ == '__main__':
    main()