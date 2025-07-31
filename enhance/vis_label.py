import os
import json
import cv2
import numpy as np
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# 解决中文显示问题（使用PIL库绘制文本）
def put_chinese_text(img, text, position, size=20, color=(255, 255, 255)):
    """
    在图像上绘制中文文本（使用PIL库避免乱码），添加半透明背景
    """
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')  # 使用RGBA模式支持透明度

    # 尝试加载中文字体
    try:
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "C:/Windows/Fonts/simhei.ttf"  # Windows
        ]

        # 选择可用的字体
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break

        if font is None:
            font = ImageFont.load_default()
            print("警告：未找到中文字体，可能仍有乱码")

    except:
        font = ImageFont.load_default()
        print("警告：加载字体失败，可能仍有乱码")

    # 计算文本大小
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 添加半透明背景
    padding = 5
    bg_position = (
        position[0] - padding,
        position[1] - padding,
        position[0] + text_width + padding,
        position[1] + text_height + padding
    )
    draw.rectangle(bg_position, fill=(0, 0, 0, 128))  # 半透明黑色背景

    # 绘制文本
    draw.text(position, text, font=font, fill=color)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)


import os


def get_matching_pairs(image_dir, label_dir):
    """获取图像和标签文件夹中文件名匹配的图像-标签对，支持json和txt格式的标签"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    label_extensions = ['.json', '.txt']  # 支持的标签文件扩展名

    image_label_pairs = []

    for img_filename in os.listdir(image_dir):
        img_name, img_ext = os.path.splitext(img_filename)
        if img_ext.lower() in image_extensions:
            # 检查所有支持的标签扩展名
            found = False
            for label_ext in label_extensions:
                label_filename = f"{img_name}{label_ext}"
                label_path = os.path.join(label_dir, label_filename)

                if os.path.exists(label_path):
                    image_path = os.path.join(image_dir, img_filename)
                    image_label_pairs.append((image_path, label_path))
                    found = True
                    break  # 找到一个匹配的标签就停止检查其他格式

            if not found:
                # 列出所有未找到的标签格式
                missing_labels = [f"{img_name}{ext}" for ext in label_extensions]
                print(f"警告：未找到 {img_filename} 对应的标签文件 {missing_labels}")

    return image_label_pairs


def visualize_labels(image_path, label_path):
    """在图像上绘制标签并返回带标注的图像（修复中文乱码）"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None, None

    # 转换16位图像为8位以便显示
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
        if len(image_8bit.shape) == 2:
            image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        display_image = image_8bit.copy()
        original_image = image_8bit.copy()
    else:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        display_image = image.copy()
        original_image = image.copy()

    # 读取标签文件
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
    except Exception as e:
        print(f"错误：解析标签文件 {label_path} 失败 - {e}")
        return None, None

    # 6种缺陷类型及颜色（BGR格式）
    label_colors = {
        "气孔": (0, 0, 255),  # 红色
        "夹渣": (0, 255, 0),  # 绿色
        "焊缝": (255, 0, 0),  # 蓝色
        "裂纹": (0, 255, 255),  # 青色
        "未焊透": (255, 0, 255),  # 紫色
        "未熔合": (255, 255, 0)  # 黄色
    }

    # 绘制标签形状
    shapes = label_data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label")
        points = shape.get("points")
        shape_type = shape.get("shape_type", "polygon")

        color = label_colors.get(label, (255, 255, 255))  # 默认白色

        # 转换点为整数坐标
        try:
            points_np = np.array(points, dtype=np.int32)
        except ValueError:
            print(f"警告：无效的点坐标 - {points}")
            continue

        # 绘制不同形状
        if shape_type == "polygon":
            cv2.polylines(display_image, [points_np], isClosed=True, color=color, thickness=2)
            # 半透明填充
            overlay = display_image.copy()
            cv2.fillPoly(overlay, [points_np], color)
            cv2.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
        elif shape_type == "rectangle":
            x1, y1 = points_np[0]
            x2, y2 = points_np[1]
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness=2)
        elif shape_type == "line":
            cv2.polylines(display_image, [points_np], isClosed=False, color=color, thickness=2)
        elif shape_type == "circle":
            center = tuple(points_np[0])
            radius = int(np.linalg.norm(points_np[0] - points_np[1]))
            cv2.circle(display_image, center, radius, color, thickness=2)
        else:
            cv2.polylines(display_image, [points_np], isClosed=True, color=color, thickness=2)

        # 绘制标签文本
        if len(points_np) > 0:
            text_pos = (points_np[0][0], points_np[0][1])
            # 绘制中文文本（带半透明背景）
            display_image = put_chinese_text(display_image, label, text_pos, size=20, color=color)

    # 绘制图例（修复中文显示）
    legend_y = 30
    for label, color in label_colors.items():
        # 绘制中文图例（带半透明背景）
        display_image = put_chinese_text(display_image, label, (10, legend_y), size=20, color=color)
        legend_y += 30

    # 添加文件名信息（中文支持）
    filename = os.path.basename(image_path)
    display_image = put_chinese_text(display_image, f"文件: {filename}", (10, display_image.shape[0] - 30), size=18,
                                     color=(255, 255, 255))

    return original_image, display_image


def visualize_yolo_seg(image_path, label_path, save_path=None, show=True):
    """
    可视化YOLO格式的分割标签

    参数:
        image_path: 图像文件路径
        label_path: 标签文件路径
        save_path: 保存可视化结果的路径，None则不保存
        show: 是否显示可视化结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = image.shape[:2]  # 获取图像高度和宽度

    # 定义不同类别的颜色（可根据需要扩展）
    colors = [
        (0, 0, 255),  # 红色
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (255, 255, 0),  # 青色
        (128, 0, 0),  # 深红色
        (0, 128, 0),  # 深绿色
        (0, 0, 128)  # 深蓝色
    ]

    # 读取并解析标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 分割一行数据为数值
        data = list(map(float, line.strip().split()))
        if len(data) < 5:  # 至少需要类别ID + 2个点（4个坐标）
            continue

        class_id = int(data[0])
        seg_points = data[1:]  # 后续为分割点坐标（x1, y1, x2, y2, ...）

        # 将归一化坐标转换为像素坐标
        points = []
        for i in range(0, len(seg_points), 2):
            if i + 1 >= len(seg_points):
                break
            x = seg_points[i] * w
            y = seg_points[i + 1] * h
            points.append([int(x), int(y)])

        # 转换为numpy数组用于绘图
        points_np = np.array(points, np.int32).reshape((-1, 1, 2))

        # 选择颜色（取模操作确保颜色循环使用）
        color = colors[class_id % len(colors)]

        # 绘制多边形轮廓
        cv2.polylines(image, [points_np], isClosed=True, color=color, thickness=2)

        # 在多边形上方绘制类别ID
        if points:  # 确保有坐标点
            first_point = points[0]
            cv2.putText(
                image,
                f"Class {class_id}",
                (first_point[0], first_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # 显示图像
    if show:
        cv2.imshow("YOLO Segmentation Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存图像
    if save_path:
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        print(f"可视化结果已保存至: {save_path}")

# def vis_json(image_path,label_path):
#
#     original_image, annotated_image = visualize_labels(image_path, label_path)
#     if original_image is None or annotated_image is None:
#         return
#     # 并排显示原始图像和标注图像
#     combined_image = np.hstack((original_image, annotated_image))
#
#     # 添加分隔线
#     height = combined_image.shape[0]
#     cv2.line(combined_image, (original_image.shape[1], 0), (original_image.shape[1], height), (0, 255, 255), 2)
#
#     # 添加标题
#     title_text = "左: 原始图像 | 右: 标注图像"
#     title_pos = (10, 30)
#     combined_image = put_chinese_text(combined_image, title_text, title_pos, size=24, color=(255, 255, 0))
#
#     # 显示图像
#     cv2.imshow("缺陷标签可视化", combined_image)
#
#     # 等待按键
#     key = cv2.waitKey(0)
#     if key == 27:  # ESC退出
#         print("退出程序")
#         break
#     elif key == 13:  # 回车下一张
#         continue
#
#     cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='从不同文件夹随机读取同名图像和标签并可视化（修复中文乱码）')
    parser.add_argument('--image_dir', type=str, required=True, help='图像文件夹路径')
    parser.add_argument('--label_dir', type=str, required=True, help='JSON标签文件夹路径')
    parser.add_argument('--yolo', action='store_true', help='yolo txt')
    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.isdir(args.image_dir):
        print(f"错误：图像文件夹 {args.image_dir} 不存在")
        return
    if not os.path.isdir(args.label_dir):
        print(f"错误：标签文件夹 {args.label_dir} 不存在")
        return

    # 获取所有同名图像-标签对
    image_label_pairs = get_matching_pairs(args.image_dir, args.label_dir)
    if not image_label_pairs:
        print("错误：未找到匹配的图像和标签文件")
        return

    print(f"找到 {len(image_label_pairs)} 组匹配的图像-标签对")
    print("按回车键显示下一张随机图像，按ESC键退出")

    # 创建窗口
    cv2.namedWindow("缺陷标签可视化", cv2.WINDOW_NORMAL)

    # 已显示的索引（避免重复）
    shown_indices = set()

    while True:
        # 重置已显示集合（全部显示过一遍后）
        if len(shown_indices) >= len(image_label_pairs):
            shown_indices.clear()

        # 随机选择未显示过的图像
        while True:
            idx = random.randint(0, len(image_label_pairs) - 1)
            if idx not in shown_indices:
                shown_indices.add(idx)
                break

        image_path, label_path = image_label_pairs[idx]
        print(f"\n显示: {os.path.basename(image_path)}")

        # 生成原始图像和带标签的图像
        if(args.yolo):
            visualize_yolo_seg(image_path, label_path)





if __name__ == "__main__":
    main()