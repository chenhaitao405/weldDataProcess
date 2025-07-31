import os
import cv2
import numpy as np
import argparse
from glob import glob


class ImageEnhancer:
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像增强处理

        Args:
            image: 输入图像（可能是16位）

        Returns:
            增强后的8位3通道图像
        """
        # 确保输入是单通道图像
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3:
            # 如果是多通道但不是3通道，取第一个通道
            image = image[:, :, 0]

        # 1. 对比度拉伸
        if image.dtype == np.uint16:
            # 16位图像的对比度拉伸
            p2, p98 = np.percentile(image, (2, 98))
            if p98 > p2:
                image_stretched = np.clip((image - p2) / (p98 - p2) * 65535, 0, 65535).astype(np.uint16)
            else:
                image_stretched = image

            # 转换为8位
            image_8bit = (image_stretched / 256).astype(np.uint8)
        else:
            # 8位图像的对比度拉伸
            p2, p98 = np.percentile(image, (2, 98))
            if p98 > p2:
                image_8bit = np.clip((image - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            else:
                image_8bit = image.astype(np.uint8)

        # 2. CLAHE处理
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_8bit)

        # 3. 转换为3通道图像
        image_3ch = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)

        return image_3ch

    def crop_center(self, image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """
        裁剪图像的中心区域为指定大小（默认640x640）

        Args:
            image: 输入图像
            target_size: 目标裁剪尺寸，默认640

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]

        # 如果图像小于目标尺寸，先放大到目标尺寸
        if h < target_size or w < target_size:
            scale = max(target_size / h, target_size / w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            h, w = image.shape[:2]

        # 计算中心区域的坐标
        center_h, center_w = h // 2, w // 2
        half_size = target_size // 2

        # 裁剪中心区域
        cropped = image[
                  center_h - half_size: center_h + half_size,
                  center_w - half_size: center_w + half_size
                  ]

        # 确保裁剪结果正好是目标尺寸（处理奇数尺寸情况）
        if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
            cropped = cv2.resize(cropped, (target_size, target_size))

        return cropped


def visualize_enhancement(image_path, enhancer):
    """显示裁剪后的原图和增强后的图像对比"""
    # 读取图像，对于16位图像使用IMREAD_ANYDEPTH
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False

    # 裁剪中心区域为640x640
    cropped_image = enhancer.crop_center(image)

    # 保存原图的显示版本（转换为8位以便显示）
    if cropped_image.dtype == np.uint16:
        # 16位转8位用于显示
        img_display = (cropped_image / 256).astype(np.uint8)
        if len(img_display.shape) == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    else:
        img_display = cropped_image.copy()
        if len(img_display.shape) == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    # 增强图像
    enhanced_img = enhancer.enhance_image(cropped_image)

    # 添加标题
    cv2.putText(img_display, "Original (Cropped)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(enhanced_img, "Enhanced", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 拼接图像
    combined = cv2.hconcat([img_display, enhanced_img])

    # 显示图像（640x2的宽度，适合查看）
    cv2.imshow("Image Enhancement Comparison (640x640 Crop) - Press Enter for next, ESC to exit", combined)

    # 等待用户按键
    key = cv2.waitKey(0)
    if key == 13:  # 回车键
        return True
    elif key == 27:  # ESC键退出
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='可视化图像增强效果（带中心裁剪）')
    parser.add_argument('--image_path', type=str, required=True,
                        help='图像文件路径或包含图像的目录')
    parser.add_argument('--crop_size', type=int, default=640,
                        help='裁剪尺寸，默认640')
    args = parser.parse_args()

    # 获取所有图像路径
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_paths = []

    if os.path.isdir(args.image_path):
        # 如果是目录，获取所有图像文件
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.image_path, ext)))
    elif os.path.isfile(args.image_path):
        # 如果是文件，直接使用
        image_paths.append(args.image_path)
    else:
        print(f"无效的路径: {args.image_path}")
        return

    if not image_paths:
        print(f"在 {args.image_path} 中未找到图像文件")
        return

    # 排序图像路径
    image_paths.sort()
    print(f"找到 {len(image_paths)} 张图像，按回车键查看下一张，按ESC键退出")

    # 创建增强器实例
    enhancer = ImageEnhancer()

    # 逐个显示图像
    for i, img_path in enumerate(image_paths):
        print(f"正在显示 {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
        continue_flag = visualize_enhancement(img_path, enhancer)
        if not continue_flag:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
