# -*- coding: utf-8 -*-
"""
将 labelme json 数据整理为 MVTec 兼容目录结构
Created on 2025
"""
import os
import sys
import argparse
import shutil
import json
from tqdm import tqdm
from collections import OrderedDict, defaultdict

from PIL import Image, ImageDraw

class MVTecPreparer(object):
    """
    从 labelme json 中读取标注：
    - 没有有效标注 => 作为 good
    - 有有效标注 => 放到 <anomaly_type>/ ，并在 ground_truth/<anomaly_type>/ 下生成二值掩码
      * 若一个图包含多个不同 label => anomaly_type = 'mixed'
    目录结构：
      <out_root>/<classname>/<split>/{good|<anomaly_type>}/image_files
      <out_root>/<classname>/ground_truth/<anomaly_type>/mask_files(.png)
    """

    def __init__(self, json_dir, out_root=None, classname="custom",
                 split="train", filter_label=None, clean=True):
        self._json_dir = json_dir
        self._classname = classname
        self._split = split  # 'train' or 'test' (与读取脚本一致)
        self._filter_label = filter_label
        self._clean = clean

        # 输出根目录
        self._out_root = out_root or os.path.join(self._json_dir, "MVTecLike")
        self._class_dir = os.path.join(self._out_root, self._classname)
        self._split_dir = os.path.join(self._class_dir, self._split)
        self._gt_dir = os.path.join(self._class_dir, "ground_truth")

        # 统计信息
        self._good_count = 0
        self._bad_count = 0
        self._missing_image_count = 0
        self._anomaly_stats = defaultdict(int)

    # ---------- 基础工具 ----------

    def _make_output_dirs(self):
        """创建输出目录结构"""
        if self._clean and os.path.exists(self._class_dir):
            shutil.rmtree(self._class_dir)

        # good 目录
        os.makedirs(os.path.join(self._split_dir, "good"), exist_ok=True)
        # ground_truth 根目录
        os.makedirs(self._gt_dir, exist_ok=True)

    def _get_all_json_paths(self):
        """获取所有 json 文件路径（兼容 labels/L|T/train|val 以及平铺的 json）"""
        json_paths = []

        labels_dir = os.path.join(self._json_dir, "labels")
        if os.path.exists(labels_dir):
            for folder in ["L", "T"]:
                folder_path = os.path.join(labels_dir, folder)
                if os.path.exists(folder_path):
                    for split in ["train", "val"]:
                        split_path = os.path.join(folder_path, split)
                        if os.path.exists(split_path):
                            for json_file in os.listdir(split_path):
                                if json_file.endswith(".json"):
                                    json_paths.append(os.path.join(split_path, json_file))

        if not json_paths:
            for file_name in os.listdir(self._json_dir):
                if file_name.endswith(".json"):
                    json_paths.append(os.path.join(self._json_dir, file_name))

        return sorted(json_paths)

    def _valid_shape(self, shape):
        """判断 shape 是否有效；过滤 shape_type 与点数不够的情况"""
        st = shape.get("shape_type", "").lower()
        pts = shape.get("points", [])

        if st == "circle":
            # circle 需要两个点（中心 + 圆上一点）
            return isinstance(pts, list) and len(pts) >= 2
        elif st == "rectangle":
            return isinstance(pts, list) and len(pts) >= 2
        else:
            # 多边形或未知类型，至少 3 点
            return isinstance(pts, list) and len(pts) >= 3

    def _filter_and_collect_shapes(self, json_data):
        """
        过滤无效/被排除标签，返回有效 shapes 与其标签集合
        """
        valid_shapes = []
        label_set = set()

        for shape in json_data.get("shapes", []):
            label = shape.get("label")
            if label is None or label == "":
                label = "NONE"

            # 过滤指定标签
            if self._filter_label and label == self._filter_label:
                continue

            if not self._valid_shape(shape):
                continue

            # 规范化 shape_type
            shape["shape_type"] = shape.get("shape_type", "").lower()
            valid_shapes.append({**shape, "label": label})
            label_set.add(label)

        return valid_shapes, label_set

    def _ensure_dirs_for_anomaly(self, anomaly):
        """确保 split 与 ground_truth 下的异常子目录存在"""
        os.makedirs(os.path.join(self._split_dir, anomaly), exist_ok=True)
        os.makedirs(os.path.join(self._gt_dir, anomaly), exist_ok=True)

    def _find_image_path(self, json_path):
        """在若干可能位置查找与 json 同名的图像"""
        base = os.path.splitext(os.path.basename(json_path))[0]
        candidates = [
            json_path.replace(".json", ".jpg").replace("/labels/", "/images/"),
            json_path.replace(".json", ".png").replace("/labels/", "/images/"),
            os.path.join(os.path.dirname(json_path), base + ".jpg"),
            os.path.join(os.path.dirname(json_path), base + ".png"),
            os.path.join(self._json_dir, "images", base + ".jpg"),
            os.path.join(self._json_dir, "images", base + ".png"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ---------- 掩码绘制 ----------

    @staticmethod
    def _draw_polygon(draw, points):
        draw.polygon(points, fill=255)

    @staticmethod
    def _draw_rectangle(draw, points):
        # 取两个对角点
        (x1, y1), (x2, y2) = points[0], points[1]
        draw.rectangle([x1, y1, x2, y2], fill=255)

    @staticmethod
    def _draw_circle(draw, points):
        # points[0] = center, points[1] = on-circle
        (cx, cy), (px, py) = points[0], points[1]
        import math
        r = math.hypot(px - cx, py - cy)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=255)

    def _rasterize_mask(self, img_w, img_h, shapes):
        """
        把一组 shapes 光栅化成二值掩码（uint8, 0/255）
        """
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)

        for s in shapes:
            st = s.get("shape_type", "")
            pts = s.get("points", [])

            # 容错：把坐标裁剪到图像范围内
            clipped = []
            for x, y in pts:
                x = max(0, min(img_w - 1, float(x)))
                y = max(0, min(img_h - 1, float(y)))
                clipped.append((x, y))

            try:
                if st == "rectangle":
                    if len(clipped) >= 2:
                        self._draw_rectangle(draw, clipped)
                elif st == "circle":
                    if len(clipped) >= 2:
                        self._draw_circle(draw, clipped)
                else:
                    if len(clipped) >= 3:
                        self._draw_polygon(draw, clipped)
            except Exception as e:
                # 某个 shape 失败时忽略，不中断整体流程
                print(f"[Warn] 绘制 shape 失败: {e}")

        return mask

    # ---------- 核心流程 ----------

    def prepare(self):
        print("开始生成 MVTec 目录结构...")

        self._make_output_dirs()

        json_paths = self._get_all_json_paths()
        print(f"找到 {len(json_paths)} 个 JSON 文件")
        if not json_paths:
            print("错误: 未找到任何 JSON")
            return

        for json_path in tqdm(json_paths, desc="处理进度"):
            json_name = os.path.basename(json_path)

            # 1) 载入 json
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Error] 读取 {json_name} 失败: {e}")
                continue

            # 2) 查找原图
            img_path = self._find_image_path(json_path)
            if not img_path or not os.path.exists(img_path):
                print(f"[Warn] 未找到图像: {json_name}")
                self._missing_image_count += 1
                continue

            # 3) 过滤与收集有效 shapes/labels
            valid_shapes, label_set = self._filter_and_collect_shapes(data)

            # 4) good 或 anomaly_type
            if len(valid_shapes) == 0:
                # 无缺陷 => good
                target_img_dir = os.path.join(self._split_dir, "good")
                os.makedirs(target_img_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(target_img_dir, os.path.basename(img_path)))
                self._good_count += 1
            else:
                # 有缺陷
                if len(label_set) == 1:
                    anomaly = next(iter(label_set))
                else:
                    anomaly = "mixed"

                self._ensure_dirs_for_anomaly(anomaly)

                # 复制图像到 split/anomaly/
                target_img_dir = os.path.join(self._split_dir, anomaly)
                os.makedirs(target_img_dir, exist_ok=True)
                dst_img = os.path.join(target_img_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst_img)

                # 若 split=test，或即便在 train 也可选择生成掩码（MVTec 规范仅 test 需要掩码）
                # 这里：不论 train/test 都生成掩码，方便可视化与后续使用；若你只想在 test 生成，可加判断。
                try:
                    img_w = data["imageWidth"]
                    img_h = data["imageHeight"]
                except KeyError:
                    # 回退：从原图读取尺寸
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size

                mask = self._rasterize_mask(img_w, img_h, valid_shapes)

                # 保存到 ground_truth/anomaly/ 同名 .png
                gt_anom_dir = os.path.join(self._gt_dir, anomaly)
                os.makedirs(gt_anom_dir, exist_ok=True)
                mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                mask.save(os.path.join(gt_anom_dir, mask_name))

                self._bad_count += 1
                self._anomaly_stats[anomaly] += 1

        # 统计
        print("\n=== 生成完成 ===")
        print(f"good 数量: {self._good_count}")
        print(f"缺陷图像数量: {self._bad_count}")
        if self._anomaly_stats:
            print("各异常类型计数：")
            for k, v in sorted(self._anomaly_stats.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {k}: {v}")
        print(f"缺失图像数: {self._missing_image_count}")
        print(f"输出根目录: {self._out_root}")
        print(f"类目录: {self._class_dir}")
        print(f"数据子目录: {self._split_dir}")
        print(f"掩码目录: {self._gt_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="将 labelme 标注数据整理为 MVTec 兼容结构")
    p.add_argument("--json_dir", type=str, required=True,
                   help="labelme json 所在目录（或包含 labels/L|T/train|val 结构的上级目录）")
    p.add_argument("--out_root", type=str, default=None,
                   help="输出根目录（默认在 json_dir 下创建 MVTecLike）")
    p.add_argument("--classname", type=str, default="custom",
                   help="类别名（MVTec 的 <classname> 目录名）")
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "val"],
                   help="输出到哪个划分目录（与读取脚本的 DatasetSplit 对应）")
    p.add_argument("--filter_label", type=str, default="焊缝",
                   help="需要过滤掉的标签名（例如 '焊缝'）")
    p.add_argument("--no_clean", action="store_true",
                   help="若指定，则不删除已存在的输出目录")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    preparer = MVTecPreparer(
        json_dir=args.json_dir,
        out_root=args.out_root,
        classname=args.classname,
        split=args.split,
        filter_label=args.filter_label,
        clean=(not args.no_clean),
    )
    preparer.prepare()
