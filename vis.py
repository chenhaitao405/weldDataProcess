import os

# 解决Qt冲突问题
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import cv2
import numpy as np
import matplotlib

# 使用非交互式后端避免Qt冲突
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# 添加上级目录到路径，以便导入WeldImagePreprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from WeldImagePreprocessor import WeldImagePreprocessor


class PreprocessingVisualizer:
    """焊缝图像预处理可视化工具"""

    def __init__(self):
        self.preprocessor = WeldImagePreprocessor(overlap_ratio=0.5)

        # 定义缺陷类型的颜色映射
        self.defect_colors = {
            '气孔': 'red',
            '裂纹': 'blue',
            '未焊透': 'green',
            '未熔合': 'yellow',
            '夹渣': 'purple',
            '伪缺陷': 'orange',
            '焊缝': 'cyan'
        }

    def draw_annotations(self, ax, annotations: Dict, scale: float = 1.0):
        """
        在matplotlib axis上绘制标注

        Args:
            ax: matplotlib axis对象
            annotations: 标注数据
            scale: 缩放比例
        """
        for shape in annotations.get('shapes', []):
            points = np.array(shape['points']) * scale
            label = shape['label']
            color = self.defect_colors.get(label, 'white')

            # 绘制多边形
            polygon = Polygon(points, closed=True, fill=False,
                              edgecolor=color, linewidth=2)
            ax.add_patch(polygon)

            # 添加标签
            center_x = points[:, 0].mean()
            center_y = points[:, 1].mean()
            ax.text(center_x, center_y, label, color=color,
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    def visualize_single_image(self, image_path: str, json_path: str,
                               save_path: str = None, show: bool = True):
        """
        可视化单张图像的预处理效果

        Args:
            image_path: 原始图像路径
            json_path: 原始标注路径
            save_path: 保存路径（可选）
            show: 是否显示图像（在Agg后端下此参数无效）
        """
        # 读取原始图像和标注
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            print(f"无法读取图像: {image_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            original_annotations = json.load(f)

        # 转换原始图像为显示格式
        if original_image.dtype == np.uint16:
            display_original = (original_image / 256).astype(np.uint8)
        else:
            display_original = original_image

        # 获取处理后的patches
        patches = self.preprocessor.process_single_image(
            image_path, json_path,
            './temp_output/images', './temp_output/labels',
            return_patches=True
        )

        if not patches:
            print("处理图像失败")
            return

        print(f"生成了 {len(patches)} 个patches")

        # 限制显示的patches数量，避免subplot索引超限
        max_patches_to_show = 20  # 最多显示20个patches
        if len(patches) > max_patches_to_show:
            print(f"由于patches数量过多({len(patches)}个)，只显示前{max_patches_to_show}个")
            patches_to_show = patches[:max_patches_to_show]
        else:
            patches_to_show = patches

        # 创建可视化图形
        # 原始图像占一整行，每个patch占两列（原始+增强）
        n_patches_to_show = len(patches_to_show)
        n_cols = 4  # 固定4列
        # 计算需要的行数：1行原始图像 + patches需要的行数
        n_patch_rows = (n_patches_to_show * 2 + n_cols - 1) // n_cols  # 每个patch占2个位置
        n_rows = 1 + n_patch_rows

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        # 显示原始图像（占据第一行的所有列）
        ax1 = plt.subplot(n_rows, 1, 1)  # 使用不同的网格规范
        ax1.imshow(display_original, cmap='gray')
        ax1.set_title(f'原始图像 - {Path(image_path).name}\n尺寸: {original_image.shape}, 总patches数: {len(patches)}',
                      fontsize=14)
        ax1.axis('off')

        # 绘制原始标注
        self.draw_annotations(ax1, original_annotations)

        # 在原始图像上绘制所有裁剪区域（包括未显示的）
        for i, patch_info in enumerate(patches):
            x, y = patch_info['position']
            w, h = patch_info['size']

            # 为显示的patches使用不同的颜色
            if i < len(patches_to_show):
                edgecolor = 'yellow'
                linewidth = 3
            else:
                edgecolor = 'gray'
                linewidth = 1

            rect = mpatches.Rectangle((x, y), w, h, linewidth=linewidth,
                                      edgecolor=edgecolor, facecolor='none',
                                      linestyle='--', alpha=0.8)
            ax1.add_patch(rect)
            ax1.text(x + w / 2, y + h / 2, f'{i}', color=edgecolor,
                     fontsize=12, ha='center', va='center', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # 显示每个处理后的patch（从第二行开始）
        for idx, patch_info in enumerate(patches_to_show):
            # 计算子图位置
            row = idx // (n_cols // 2) + 2  # 从第2行开始
            col_base = (idx % (n_cols // 2)) * 2 + 1  # 每个patch占2列

            # 原始patch
            subplot_idx_orig = (row - 1) * n_cols + col_base
            ax_orig = plt.subplot(n_rows, n_cols, subplot_idx_orig)

            if patch_info['original_patch'].dtype == np.uint16:
                display_patch = (patch_info['original_patch'] / 256).astype(np.uint8)
            else:
                display_patch = patch_info['original_patch']
            ax_orig.imshow(display_patch, cmap='gray')
            ax_orig.set_title(f'Patch {idx} - 原始\n位置: {patch_info["position"]}', fontsize=10)
            ax_orig.axis('off')

            # 绘制调整后的标注
            self.draw_annotations(ax_orig, patch_info['annotations'])

            # 增强后的patch
            subplot_idx_enh = subplot_idx_orig + 1
            ax_enh = plt.subplot(n_rows, n_cols, subplot_idx_enh)
            ax_enh.imshow(patch_info['enhanced_patch'])
            ax_enh.set_title(f'Patch {idx} - 增强后\n缺陷数: {len(patch_info["annotations"]["shapes"])}', fontsize=10)
            ax_enh.axis('off')

            # 绘制调整后的标注
            self.draw_annotations(ax_enh, patch_info['annotations'])

        # 添加图例
        legend_elements = [mpatches.Patch(facecolor=color, edgecolor=color, label=defect)
                           for defect, color in self.defect_colors.items()]
        fig.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(0.98, 0.98), fontsize=10)

        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            # 如果没有指定保存路径，自动保存
            auto_save_path = f"visualization_{Path(image_path).stem}.png"
            plt.savefig(auto_save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {auto_save_path}")

        plt.close(fig)  # 释放内存

        # 清理临时文件
        import shutil
        if os.path.exists('./temp_output'):
            shutil.rmtree('./temp_output')

        return patches

    def compare_enhancement(self, image_path: str, save_path: str = None, show: bool = True):
        """
        对比图像增强效果

        Args:
            image_path: 原始图像路径
            save_path: 保存路径（可选）
            show: 是否显示图像（在Agg后端下此参数无效）
        """
        # 读取原始图像
        original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if original is None:
            print(f"无法读取图像: {image_path}")
            return

        # 准备显示的原始图像
        if original.dtype == np.uint16:
            display_original = (original / 256).astype(np.uint8)
        else:
            display_original = original

        # 应用增强
        enhanced = self.preprocessor.enhance_image(original)

        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始图像
        axes[0, 0].imshow(display_original, cmap='gray')
        axes[0, 0].set_title('原始图像', fontsize=14)
        axes[0, 0].axis('off')

        # 直方图 - 原始
        axes[1, 0].hist(display_original.ravel(), bins=256, color='gray', alpha=0.7)
        axes[1, 0].set_title('原始图像直方图', fontsize=14)
        axes[1, 0].set_xlim([0, 255])
        axes[1, 0].set_ylabel('像素数量')
        axes[1, 0].set_xlabel('像素值')

        # 对比度拉伸后
        if original.dtype == np.uint16:
            p2, p98 = np.percentile(original, (2, 98))
            if p98 > p2:
                stretched = np.clip((original - p2) / (p98 - p2) * 65535, 0, 65535).astype(np.uint16)
                stretched_8bit = (stretched / 256).astype(np.uint8)
            else:
                stretched_8bit = display_original
        else:
            p2, p98 = np.percentile(original, (2, 98))
            if p98 > p2:
                stretched_8bit = np.clip((original - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            else:
                stretched_8bit = display_original

        axes[0, 1].imshow(stretched_8bit, cmap='gray')
        axes[0, 1].set_title('对比度拉伸后', fontsize=14)
        axes[0, 1].axis('off')

        # 直方图 - 对比度拉伸
        axes[1, 1].hist(stretched_8bit.ravel(), bins=256, color='blue', alpha=0.7)
        axes[1, 1].set_title('对比度拉伸后直方图', fontsize=14)
        axes[1, 1].set_xlim([0, 255])
        axes[1, 1].set_ylabel('像素数量')
        axes[1, 1].set_xlabel('像素值')

        # 增强后（CLAHE + 3通道）
        axes[0, 2].imshow(enhanced)
        axes[0, 2].set_title('完全增强后（CLAHE + 3通道）', fontsize=14)
        axes[0, 2].axis('off')

        # 直方图 - 增强后
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        axes[1, 2].hist(enhanced_gray.ravel(), bins=256, color='green', alpha=0.7)
        axes[1, 2].set_title('增强后直方图', fontsize=14)
        axes[1, 2].set_xlim([0, 255])
        axes[1, 2].set_ylabel('像素数量')
        axes[1, 2].set_xlabel('像素值')

        plt.suptitle(f'图像增强效果对比: {Path(image_path).name}', fontsize=16)
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"增强对比图已保存到: {save_path}")
        else:
            auto_save_path = f"enhancement_comparison_{Path(image_path).stem}.png"
            plt.savefig(auto_save_path, dpi=150, bbox_inches='tight')
            print(f"增强对比图已保存到: {auto_save_path}")

        plt.close(fig)  # 释放内存

    def visualize_sliding_window_process(self, image_path: str,
                                         save_path: str = None, show: bool = True):
        """
        可视化滑动窗口过程

        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            show: 是否显示图像（在Agg后端下此参数无效）
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return

        # 准备显示
        if image.dtype == np.uint16:
            display_image = (image / 256).astype(np.uint8)
        else:
            display_image = image

        # 计算窗口参数
        h, w = image.shape[:2]
        window_size = min(h, w) // 2
        stride = int(window_size * (1 - self.preprocessor.overlap_ratio))

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 显示原始图像
        ax.imshow(display_image, cmap='gray', alpha=0.8)
        ax.set_title(
            f'滑动窗口过程可视化\n窗口大小: {window_size}x{window_size}, 步长: {stride}, 重叠率: {self.preprocessor.overlap_ratio * 100}%',
            fontsize=14)

        # 获取所有窗口位置
        patches = self.preprocessor.sliding_window_crop(
            image, (window_size, window_size), (stride, stride)
        )

        # 使用不同颜色绘制窗口
        colors = plt.cm.rainbow(np.linspace(0, 1, len(patches)))

        for i, (patch_info, color) in enumerate(zip(patches, colors)):
            x, y = patch_info['position']
            w, h = patch_info['size']

            # 绘制矩形
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                      edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # 添加编号
            ax.text(x + w / 2, y + h / 2, f'{i}', color='white',
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('宽度（像素）')
        ax.set_ylabel('高度（像素）')

        # 添加统计信息
        info_text = f'图像尺寸: {w}x{h}\n窗口数量: {len(patches)}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=12)

        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"滑动窗口可视化已保存到: {save_path}")
        else:
            auto_save_path = f"sliding_window_{Path(image_path).stem}.png"
            plt.savefig(auto_save_path, dpi=150, bbox_inches='tight')
            print(f"滑动窗口可视化已保存到: {auto_save_path}")

        plt.close(fig)  # 释放内存


def test_single_image():
    """测试单张图像的预处理效果"""
    visualizer = PreprocessingVisualizer()

    # 设置测试图像路径（请根据实际情况修改）
    test_image = "/home/num2/datasets/Xray/crop_weld_data/crop_weld_images/L/1/A_bam5.tif"  # 示例路径
    test_json = "/home/num2/datasets/Xray/crop_weld_data/crop_weld_jsons/L/1/A_bam5.json"  # 示例路径

    # 检查文件是否存在
    if not Path(test_image).exists() or not Path(test_json).exists():
        print("请修改test_single_image()函数中的测试图像路径")
        print(f"当前路径: {test_image}")
        print(f"当前路径: {test_json}")
        # 尝试查找第一个可用的图像
        for weld_type in ['L', 'T']:
            for sub_type in ['1', '2']:
                img_dir = Path(f"./crop_weld_images/{weld_type}/{sub_type}")
                json_dir = Path(f"./crop_weld_jsons/{weld_type}/{sub_type}")
                if img_dir.exists():
                    img_files = list(img_dir.glob("*.tif"))
                    if img_files:
                        test_image = str(img_files[0])
                        test_json = str(json_dir / f"{img_files[0].stem}.json")
                        if Path(test_json).exists():
                            print(f"找到测试图像: {test_image}")
                            break
            else:
                continue
            break
        else:
            print("未找到任何可用的测试图像")
            return

    # 1. 可视化完整的预处理过程
    print("1. 可视化完整预处理过程...")
    visualizer.visualize_single_image(
        test_image, test_json,
        save_path="./visualization_complete.png"
    )

    # 2. 可视化图像增强效果
    print("\n2. 可视化图像增强效果...")
    visualizer.compare_enhancement(
        test_image,
        save_path="./visualization_enhancement.png"
    )

    # 3. 可视化滑动窗口过程
    print("\n3. 可视化滑动窗口过程...")
    visualizer.visualize_sliding_window_process(
        test_image,
        save_path="./visualization_sliding_window.png"
    )

    print("\n所有可视化结果已保存到当前目录")


def test_batch_visualization():
    """批量测试多张图像（可选）"""
    visualizer = PreprocessingVisualizer()

    # 设置输入目录
    base_dir = Path(".")
    output_dir = Path("./visualization_results")
    output_dir.mkdir(exist_ok=True)

    # 查找所有图像
    test_cases = []
    for weld_type in ['L', 'T']:
        for sub_type in ['1', '2']:
            image_dir = base_dir / 'crop_weld_images' / weld_type / sub_type
            json_dir = base_dir / 'crop_weld_jsons' / weld_type / sub_type

            if image_dir.exists():
                # 只取前2张图像作为示例
                image_files = sorted(list(image_dir.glob('*.tif')))[:2]

                for img_file in image_files:
                    json_file = json_dir / f"{img_file.stem}.json"
                    if json_file.exists():
                        test_cases.append({
                            'image': str(img_file),
                            'json': str(json_file),
                            'type': f"{weld_type}_{sub_type}",
                            'name': img_file.stem
                        })

    # 处理每个测试案例
    for i, case in enumerate(test_cases):
        print(f"\n处理测试案例 {i + 1}/{len(test_cases)}: {case['name']} ({case['type']})")

        # 创建输出子目录
        case_dir = output_dir / f"{case['type']}_{case['name']}"
        case_dir.mkdir(exist_ok=True)

        # 可视化
        visualizer.visualize_single_image(
            case['image'], case['json'],
            save_path=str(case_dir / "complete_process.png")
        )

        visualizer.compare_enhancement(
            case['image'],
            save_path=str(case_dir / "enhancement_comparison.png")
        )

        visualizer.visualize_sliding_window_process(
            case['image'],
            save_path=str(case_dir / "sliding_window.png")
        )

    print(f"\n所有可视化结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("焊缝图像预处理可视化测试")
    print("=" * 50)

    # 运行单张图像测试
    test_single_image()

    # 可选：运行批量测试
    # print("\n\n运行批量可视化测试...")
    # test_batch_visualization()


if __name__ == "__main__":
    main()