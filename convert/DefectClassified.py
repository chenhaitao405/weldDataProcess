'''
缺陷图像分类脚本
根据是否有标注（缺陷）将图像分类到good和bad目录
Created on 2024
'''
import os
import sys
import argparse
import shutil
import json
from tqdm import tqdm
from collections import OrderedDict

class DefectImageClassifier(object):

    def __init__(self, json_dir, to_seg=False, filter_label=None):
        self._json_dir = json_dir
        self._to_seg = to_seg
        self._filter_label = filter_label

        # 输出目录
        self._output_dir = os.path.join(self._json_dir, 'DefectClassified')
        self._good_dir = os.path.join(self._output_dir, 'good')
        self._bad_dir = os.path.join(self._output_dir, 'bad')

        # 统计信息
        self._good_count = 0
        self._bad_count = 0
        self._missing_image_count = 0

    def _make_output_dirs(self):
        """创建输出目录结构"""
        # 如果输出目录存在，先删除
        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir)

        # 创建good和bad目录及其子目录
        for base_dir in [self._good_dir, self._bad_dir]:
            os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'labels'), exist_ok=True)

    def _get_all_json_paths(self):
        """获取所有json文件路径"""
        json_paths = []

        # 检查是否有labels/L和labels/T目录结构（参考原脚本的convert_mypath方法）
        labels_dir = os.path.join(self._json_dir, 'labels')
        if os.path.exists(labels_dir):
            for folder in ['L', 'T']:
                folder_path = os.path.join(labels_dir, folder)
                if os.path.exists(folder_path):
                    for split in ['train', 'val']:
                        split_path = os.path.join(folder_path, split)
                        if os.path.exists(split_path):
                            for json_file in os.listdir(split_path):
                                if json_file.endswith('.json'):
                                    json_paths.append(os.path.join(split_path, json_file))

        # 如果没有找到，则直接从json_dir读取
        if not json_paths:
            for file_name in os.listdir(self._json_dir):
                if file_name.endswith('.json'):
                    json_paths.append(os.path.join(self._json_dir, file_name))

        return json_paths

    def _check_has_defect(self, json_data):
        """
        检查json数据中是否有缺陷（有效的标注）
        返回：(has_defect, valid_shapes)
        """
        valid_shapes = []

        for shape in json_data.get('shapes', []):
            label = shape.get('label')

            # 过滤指定标签
            if self._filter_label and label == self._filter_label:
                continue

            # 检查points个数，如果小于3则跳过（除了circle类型）
            if shape.get('shape_type') != 'circle':
                if 'points' in shape and len(shape['points']) < 3:
                    continue

            valid_shapes.append(shape)

        return len(valid_shapes) > 0, valid_shapes

    def _get_label_id_map(self, all_json_paths):
        """获取所有标签的映射"""
        label_set = set()

        for json_path in all_json_paths:
            if os.path.exists(json_path):
                data = json.load(open(json_path))
                for shape in data.get('shapes', []):
                    label = shape.get('label')

                    # 如果label是None或空，转换为字符串"NONE"
                    if label is None or label == '':
                        label = 'NONE'

                    # 过滤指定标签
                    if self._filter_label and label == self._filter_label:
                        continue

                    label_set.add(label)

        return OrderedDict([(label, label_id)
                            for label_id, label in enumerate(sorted(label_set))])

    def _convert_to_yolo_format(self, shape, img_h, img_w, label_id):
        """将shape转换为YOLO格式（参考原脚本）"""
        if self._to_seg:
            # 分割格式
            retval = [label_id]
            if shape['shape_type'] == 'circle':
                # circle的特殊处理（省略具体实现，参考原脚本）
                pass
            else:
                for point in shape['points']:
                    retval.append(round(float(point[0]) / img_w, 6))
                    retval.append(round(float(point[1]) / img_h, 6))
            return retval
        else:
            # 检测框格式
            if shape['shape_type'] == 'circle':
                # circle的特殊处理
                center_x, center_y = shape['points'][0]
                import math
                radius = math.sqrt((center_x - shape['points'][1][0]) ** 2 +
                                   (center_y - shape['points'][1][1]) ** 2)

                yolo_center_x = round(float(center_x / img_w), 6)
                yolo_center_y = round(float(center_y / img_h), 6)
                yolo_w = round(float(2 * radius / img_w), 6)
                yolo_h = round(float(2 * radius / img_h), 6)
            else:
                # 其他形状
                x_coords = [p[0] for p in shape['points']]
                y_coords = [p[1] for p in shape['points']]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                obj_w = x_max - x_min
                obj_h = y_max - y_min

                yolo_center_x = round(float((x_min + obj_w / 2.0) / img_w), 6)
                yolo_center_y = round(float((y_min + obj_h / 2.0) / img_h), 6)
                yolo_w = round(float(obj_w / img_w), 6)
                yolo_h = round(float(obj_h / img_h), 6)

            return [label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h]

    def _save_yolo_label(self, valid_shapes, json_data, label_path, label_id_map):
        """保存YOLO格式的标签文件"""
        img_h = json_data['imageHeight']
        img_w = json_data['imageWidth']

        yolo_objects = []
        for shape in valid_shapes:
            label = shape.get('label')
            if label is None or label == '':
                label = 'NONE'

            if label in label_id_map:
                label_id = label_id_map[label]
                yolo_obj = self._convert_to_yolo_format(shape, img_h, img_w, label_id)
                yolo_objects.append(yolo_obj)

        # 写入标签文件
        with open(label_path, 'w') as f:
            for i, yolo_obj in enumerate(yolo_objects):
                line = ' '.join(str(x) for x in yolo_obj)
                if i < len(yolo_objects) - 1:
                    line += '\n'
                f.write(line)

    def _copy_image(self, json_path, target_image_dir, json_name):
        """复制图像文件"""
        # 尝试多种可能的图像路径
        img_name = json_name.replace('.json', '.jpg')

        # 可能的图像路径
        possible_paths = [
            json_path.replace('.json', '.jpg').replace('/labels/', '/images/'),
            json_path.replace('.json', '.jpg'),
            json_path.replace('.json', '.png').replace('/labels/', '/images/'),
            json_path.replace('.json', '.png'),
            os.path.join(os.path.dirname(json_path), img_name),
            os.path.join(self._json_dir, 'images', img_name)
        ]

        for src_path in possible_paths:
            if os.path.exists(src_path):
                dst_path = os.path.join(target_image_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
                return True

        print(f"警告: 未找到图像文件: {json_name}")
        self._missing_image_count += 1
        return False

    def classify(self):
        """执行分类主流程"""
        print("开始分类缺陷图像...")

        # 创建输出目录
        self._make_output_dirs()

        # 获取所有json路径
        json_paths = self._get_all_json_paths()
        print(f"找到 {len(json_paths)} 个JSON文件")

        if not json_paths:
            print("错误: 未找到任何JSON文件")
            return

        # 获取标签映射
        print("生成标签映射...")
        label_id_map = self._get_label_id_map(json_paths)
        print(f"找到 {len(label_id_map)} 个标签类别: {list(label_id_map.keys())}")

        # 处理每个json文件
        for json_path in tqdm(json_paths, desc="分类进度"):
            json_name = os.path.basename(json_path)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # 检查是否有缺陷
                has_defect, valid_shapes = self._check_has_defect(json_data)

                if has_defect:
                    # 有缺陷，放到bad目录
                    target_dir = self._bad_dir
                    self._bad_count += 1
                else:
                    # 无缺陷，放到good目录
                    target_dir = self._good_dir
                    self._good_count += 1

                # 复制图像
                img_copied = self._copy_image(json_path,
                                              os.path.join(target_dir, 'images'),
                                              json_name)

                if img_copied:
                    # 保存标签文件
                    label_name = json_name.replace('.json', '.txt')
                    label_path = os.path.join(target_dir, 'labels', label_name)

                    if has_defect:
                        # 有缺陷时保存YOLO格式标签
                        self._save_yolo_label(valid_shapes, json_data,
                                              label_path, label_id_map)
                    else:
                        # 无缺陷时创建空标签文件
                        open(label_path, 'w').close()

            except Exception as e:
                print(f"处理 {json_name} 时出错: {str(e)}")
                continue

        # 输出统计信息
        print("\n=== 分类完成 ===")
        print(f"无缺陷图像 (good): {self._good_count}")
        print(f"有缺陷图像 (bad): {self._bad_count}")
        print(f"缺失图像数: {self._missing_image_count}")
        print(f"输出目录: {self._output_dir}")

        # 生成数据集配置文件
        self._save_dataset_yaml(label_id_map)

    def _save_dataset_yaml(self, label_id_map):
        """生成dataset.yaml配置文件"""
        yaml_path = os.path.join(self._output_dir, 'dataset.yaml')

        with open(yaml_path, 'w') as f:
            f.write(f"# 缺陷分类数据集配置\n")
            f.write(f"# 无缺陷图像: {self._good_count}\n")
            f.write(f"# 有缺陷图像: {self._bad_count}\n\n")

            f.write(f"good: {os.path.join(self._good_dir, 'images')}\n")
            f.write(f"bad: {os.path.join(self._bad_dir, 'images')}\n\n")

            f.write(f"nc: {len(label_id_map)}\n\n")

            names_str = ', '.join([f"'{label}'" for label in label_id_map.keys()])
            f.write(f"names: [{names_str}]\n")

        print(f"已生成配置文件: {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将标注数据按是否有缺陷分类到good和bad目录')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='labelme json文件所在的目录路径')
    parser.add_argument('--seg', action='store_true',
                        help='是否转换为分割格式（默认为检测框格式）')
    parser.add_argument('--filter_label', type=str, default="焊缝",
                        help='Label to filter out (e.g., "焊缝")')

    args = parser.parse_args()

    classifier = DefectImageClassifier(args.json_dir,
                                       to_seg=args.seg,
                                       filter_label=args.filter_label)
    classifier.classify()