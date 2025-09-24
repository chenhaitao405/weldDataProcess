import os
import argparse
import shutil
from pathlib import Path


def seg_to_det_line(seg_line):
    """
    将一行分割标注转换为检测标注
    输入: class x1 y1 x2 y2 ... xn yn (归一化的多边形顶点)
    输出: class x_center y_center width height (归一化的边界框)
    """
    parts = seg_line.strip().split()
    if len(parts) < 3:  # 至少需要类别 + 一个点(2个坐标)
        return None

    class_id = parts[0]
    coords = list(map(float, parts[1:]))

    # 提取x和y坐标
    x_coords = []
    y_coords = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x_coords.append(coords[i])
            y_coords.append(coords[i + 1])

    if not x_coords or not y_coords:
        return None

    # 计算边界框
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # 计算中心点和宽高（都是归一化的）
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # 确保值在[0, 1]范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_label_file(input_label_path, output_label_path):
    """
    转换单个标注文件（分割到检测）
    """
    with open(input_label_path, 'r') as f:
        seg_lines = f.readlines()

    det_lines = []
    for seg_line in seg_lines:
        det_line = seg_to_det_line(seg_line)
        if det_line:
            det_lines.append(det_line + '\n')

    with open(output_label_path, 'w') as f:
        f.writelines(det_lines)


def copy_images(input_dir, output_dir):
    """
    复制图像文件夹结构和内容
    """
    if os.path.exists(output_dir):
        print(f"Copying images from {input_dir} to {output_dir}")
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)


def get_image_extensions():
    """
    获取支持的图像文件扩展名
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def find_corresponding_image(images_dir, label_name):
    """
    根据标签文件名查找对应的图像文件
    """
    base_name = Path(label_name).stem
    image_extensions = get_image_extensions()

    for ext in image_extensions:
        image_path = images_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path

    return None


def get_primary_class(label_file):
    """
    从标签文件中获取主要类别（出现次数最多的类别）
    如果文件为空或无有效标注，返回None
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            return None

        # 统计每个类别出现的次数
        class_counts = {}
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = parts[0]
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        if not class_counts:
            return None

        # 返回出现次数最多的类别
        return max(class_counts, key=class_counts.get)

    except Exception:
        return None


def convert_to_cls(input_dir, output_dir):
    """
    将数据集转换为分类格式
    根据标签文件将图像组织到不同的类别文件夹中
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 检查输入目录
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找图像和标签目录
    input_images_dir = input_path / 'images'
    input_labels_dir = input_path / 'labels'

    if not input_images_dir.exists():
        print(f"Error: Images directory not found in {input_dir}")
        return

    if not input_labels_dir.exists():
        print(f"Error: Labels directory not found in {input_dir}")
        return

    # 获取所有split（train, val, test等）
    image_splits = [d.name for d in input_images_dir.iterdir() if d.is_dir()]
    label_splits = [d.name for d in input_labels_dir.iterdir() if d.is_dir()]

    # 使用两者的交集
    splits = list(set(image_splits) & set(label_splits))

    if not splits:
        print("No matching splits found in both images and labels folders!")
        return

    print(f"Found splits: {splits}")

    total_processed = 0
    total_with_labels = 0
    total_without_labels = 0
    class_distribution = {}

    # 处理每个split
    for split in splits:
        print(f"\n📂 Processing {split} split...")

        split_images_dir = input_images_dir / split
        split_labels_dir = input_labels_dir / split
        split_output_dir = output_path / split

        # 获取所有图像文件
        image_extensions = get_image_extensions()
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_images_dir.glob(f'*{ext}'))

        print(f"  Found {len(image_files)} images")

        # 处理每个图像
        for image_file in image_files:
            # 查找对应的标签文件
            label_file = split_labels_dir / f"{image_file.stem}.txt"

            # 判断图像属于哪个类别
            if label_file.exists():
                primary_class = get_primary_class(label_file)
                if primary_class is not None:
                    # 有标签，放入对应的类别文件夹
                    class_folder = f"label{primary_class}"
                    total_with_labels += 1

                    # 统计类别分布
                    if split not in class_distribution:
                        class_distribution[split] = {}
                    class_distribution[split][class_folder] = \
                        class_distribution[split].get(class_folder, 0) + 1
                else:
                    # 标签文件存在但为空或无效
                    class_folder = "none"
                    total_without_labels += 1
            else:
                # 无标签文件
                class_folder = "none"
                total_without_labels += 1

            # 创建目标文件夹并复制图像
            target_dir = split_output_dir / class_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / image_file.name
            shutil.copy2(image_file, target_path)

            total_processed += 1

            # 每处理100个文件打印一次进度
            if total_processed % 100 == 0:
                print(f"    Processed {total_processed} files...")

        # 打印split统计
        print(f"  ✅ Completed {split}:")
        print(f"     - Total images: {len(image_files)}")
        if split in class_distribution:
            for class_name, count in sorted(class_distribution[split].items()):
                print(f"     - {class_name}: {count} images")

        # 统计none类别
        none_count = len(list((split_output_dir / "none").glob("*"))) \
            if (split_output_dir / "none").exists() else 0
        if none_count > 0:
            print(f"     - none: {none_count} images")

    # 打印总体统计
    print(f"\n" + "=" * 50)
    print(f"✅ Classification dataset created successfully!")
    print(f"📊 Overall Statistics:")
    print(f"  - Total images processed: {total_processed}")
    print(f"  - Images with labels: {total_with_labels}")
    print(f"  - Images without labels: {total_without_labels}")
    print(f"  - Output saved to: {output_dir}")

    # 打印类别分布总结
    print(f"\n📈 Class Distribution Summary:")
    all_classes = set()
    for split_classes in class_distribution.values():
        all_classes.update(split_classes.keys())

    if all_classes:
        for class_name in sorted(all_classes):
            total_in_class = sum(
                split_classes.get(class_name, 0)
                for split_classes in class_distribution.values()
            )
            print(f"  - {class_name}: {total_in_class} total images")


def convert_to_det(input_dir, output_dir, copy_imgs=True):
    """
    将分割数据集转换为检测数据集（原有功能）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 检查输入目录是否存在
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 处理图像文件夹
    if copy_imgs:
        input_images_dir = input_path / 'images'
        output_images_dir = output_path / 'images'

        if input_images_dir.exists():
            print("Copying images directory...")
            copy_images(input_images_dir, output_images_dir)
            print("Images copied successfully!")
        else:
            print(f"Warning: Images directory not found in {input_dir}")
    else:
        print("Skipping image copying (--no_copy_images flag set)")

    # 处理标注文件夹
    input_labels_dir = input_path / 'labels'
    output_labels_dir = output_path / 'labels'

    if not input_labels_dir.exists():
        print(f"Error: Labels directory not found in {input_dir}")
        return

    # 获取所有子目录（train, val, test等）
    splits = [d.name for d in input_labels_dir.iterdir() if d.is_dir()]

    if not splits:
        print("No subdirectories found in labels folder!")
        return

    print(f"Found splits: {splits}")

    # 转换每个split的标注文件
    total_files = 0
    for split in splits:
        input_split_dir = input_labels_dir / split
        output_split_dir = output_labels_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有txt文件
        txt_files = list(input_split_dir.glob('*.txt'))

        print(f"\nProcessing {split} split: {len(txt_files)} files")

        for txt_file in txt_files:
            input_label_path = txt_file
            output_label_path = output_split_dir / txt_file.name

            try:
                convert_label_file(input_label_path, output_label_path)
                total_files += 1

                # 每处理100个文件打印一次进度
                if total_files % 100 == 0:
                    print(f"  Processed {total_files} files...")

            except Exception as e:
                print(f"  Error processing {txt_file.name}: {e}")

        print(f"  Completed {split}: {len(txt_files)} files converted")

    print(f"\n✅ Conversion completed successfully!")
    print(f"Total files converted: {total_files}")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO dataset between different formats (segmentation/detection/classification)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert segmentation to detection format
  python convert.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det

  # Convert to classification format
  python convert.py --input_dir ./seg_dataset --output_dir ./cls_dataset --mode cls

  # Convert to detection without copying images
  python convert.py --input_dir ./seg_dataset --output_dir ./det_dataset --mode det --no_copy_images
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to the input dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output dataset directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['det', 'cls'],
        default='det',
        help='Conversion mode: "det" for detection format, "cls" for classification format (default: det)'
    )
    parser.add_argument(
        '--no_copy_images',
        action='store_true',
        help='Do not copy images to output directory (only for det mode, cls mode always copies images)'
    )

    args = parser.parse_args()

    # 根据模式选择转换函数
    if args.mode == 'cls':
        print(f"🔄 Converting to classification format...")
        if args.no_copy_images:
            print("⚠️  Warning: --no_copy_images is ignored in cls mode")
        convert_to_cls(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
    else:  # det mode
        print(f"🔄 Converting to detection format...")
        convert_to_det(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            copy_imgs=not args.no_copy_images
        )


if __name__ == "__main__":
    main()