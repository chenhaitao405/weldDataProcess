import os
import json
from pathlib import Path


def analyze_image_dimensions(json_dir):
    # 初始化存储列表
    heights = []
    widths = []

    # 遍历目录下的所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)

            try:
                # 读取JSON文件
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # 提取高度和宽度
                height = data.get('imageHeight')
                width = data.get('imageWidth')

                if height is not None and width is not None:
                    heights.append(height)
                    widths.append(width)
                    print(f"文件: {filename} - 高度: {height}, 宽度: {width}")
                else:
                    print(f"文件: {filename} - 缺少imageHeight或imageWidth字段")

            except json.JSONDecodeError:
                print(f"文件: {filename} - 不是有效的JSON文件")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    # 计算统计数据
    if heights and widths:
        # 高度统计
        avg_height = sum(heights) / len(heights)
        max_height = max(heights)
        min_height = min(heights)

        # 宽度统计
        avg_width = sum(widths) / len(widths)
        max_width = max(widths)
        min_width = min(widths)

        # 打印统计结果
        print("\n===== 统计结果 =====")
        print(f"高度 - 平均值: {avg_height:.2f}, 最大值: {max_height}, 最小值: {min_height}")
        print(f"宽度 - 平均值: {avg_width:.2f}, 最大值: {max_width}, 最小值: {min_width}")
    else:
        print("\n没有找到有效的imageHeight和imageWidth数据")


if __name__ == "__main__":
    # 指定JSON文件目录
    json_directory = "/home/num2/datasets/Xray/crop_weld_data/crop_weld_jsons/L/1"

    # 检查目录是否存在
    if not Path(json_directory).exists():
        print(f"错误: 目录 {json_directory} 不存在")
    else:
        analyze_image_dimensions(json_directory)
