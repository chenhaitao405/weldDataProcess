"""
图像处理工具模块
提供图像增强、滑动窗口等图像处理功能
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def apply_window_level(image: np.ndarray, window_width: int,
                       window_level: int, output_bits: int = 8) -> np.ndarray:
    """
    应用窗宽窗位变换

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


def auto_window_level(image: np.ndarray) -> Tuple[int, int]:
    """
    自动计算窗宽窗位（基于统计信息）

    Args:
        image: 输入图像

    Returns:
        (window_width, window_level)
    """
    img_float = image.astype(np.float32)

    img_min = np.min(img_float)
    img_max = np.max(img_float)
    img_mean = np.mean(img_float)
    img_std = np.std(img_float)

    # 使用均值作为窗位，4倍标准差作为窗宽
    window_level = int(img_mean)
    window_width = int(min(4 * img_std, img_max - img_min))

    # 确保窗宽至少为1
    window_width = max(1, window_width)

    return window_width, window_level


def enhance_image_windowing(image: np.ndarray) -> np.ndarray:
    """
    使用窗宽窗位方法增强图像

    Args:
        image: 输入图像（可能是16位）

    Returns:
        增强后的8位3通道图像
    """
    # 如果是3通道图像，转换为单通道
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 确保图像是float32类型
    if image.dtype == np.uint16:
        img_float = image.astype(np.float32)
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float32)
    else:
        img_float = image

    # 自动计算窗宽窗位
    window_width, window_level = auto_window_level(img_float)

    # 应用窗宽窗位变换，输出8位图像
    enhanced_8bit = apply_window_level(img_float, window_width, window_level, 8)

    # CLAHE对比度增强
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # image_clahe = clahe.apply(enhanced_8bit)

    # 转换为3通道图像
    image_3ch = cv2.cvtColor(enhanced_8bit, cv2.COLOR_GRAY2BGR)

    return image_3ch


def enhance_image_original(image: np.ndarray) -> np.ndarray:
    """
    原始图像增强处理方法

    Args:
        image: 输入图像（可能是16位）

    Returns:
        增强后的8位3通道图像
    """
    # 如果是3通道图像，转换为单通道
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 转换为8位并进行直方图均衡
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image

    # 直方图均衡
    image_equalized = cv2.equalizeHist(image_8bit)

    # CLAHE处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_equalized)

    # 转换为3通道图像
    image_3ch = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)

    return image_3ch


def enhance_image(image: np.ndarray, mode: str = 'original') -> np.ndarray:
    """
    统一的图像增强接口

    Args:
        image: 输入图像
        mode: 增强模式 ('original' 或 'windowing')

    Returns:
        增强后的图像
    """
    if mode == 'windowing':
        return enhance_image_windowing(image)
    else:
        return enhance_image_original(image)


def sliding_window_crop(image: np.ndarray, window_size: Tuple[int, int],
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
        if not any(p['position'] == (x, y) for p in patches):
            patches.append(patch_info)

    return patches


def calculate_stride(window_size: Tuple[int, int], overlap_ratio: float) -> Tuple[int, int]:
    """
    根据窗口大小和重叠率计算步长

    Args:
        window_size: 窗口大小 (height, width)
        overlap_ratio: 重叠率 (0.0-1.0)

    Returns:
        步长 (y_stride, x_stride)
    """
    return (int(window_size[0] * (1 - overlap_ratio)),
            int(window_size[1] * (1 - overlap_ratio)))