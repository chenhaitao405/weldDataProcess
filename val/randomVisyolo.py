import os
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def denormalize_coordinates(norm_coords, img_width, img_height):
    """
    Denormalizes the coordinates to the image scale.
    norm_coords: List of normalized coordinates [x1, y1, x2, y2, ..., xn, yn]
    img_width: Image width
    img_height: Image height
    """
    coords = []
    for i in range(0, len(norm_coords), 2):
        x = int(norm_coords[i] * img_width)
        y = int(norm_coords[i + 1] * img_height)
        coords.append((x, y))
    return coords


def denormalize_bbox(xc, yc, w_norm, h_norm, img_width, img_height):
    """
    把 YOLO 归一化的 [x_center, y_center, w, h] 转换为像素 [x, y, w, h]
    (x, y) 是左上角坐标
    """
    w = w_norm * img_width
    h = h_norm * img_height
    x = (xc * img_width) - w / 2
    y = (yc * img_height) - h / 2
    return x, y, w, h



def visualize_segmentation(image, label_path, img_width, img_height, ax):
    """
    Visualizes segmentation annotations (polygons).
    """
    # Read the labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        coords = list(map(float, line.strip().split()[1:]))  # Ignore class label (first value)
        coords = denormalize_coordinates(coords, img_width, img_height)
        coords = np.array(coords)

        # Plot the polygon (closed shape)
        ax.fill(coords[:, 0], coords[:, 1], 'r', alpha=0.3, edgecolor='r', linewidth=2)


def visualize_detection(image, label_path, img_width, img_height, ax):
    """
    Visualizes YOLO detection annotations (bounding boxes).
    每行: class x_center y_center w h (全部归一化到0-1)
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # 应该有类别 + 4个数
            print(f"Warning: Invalid detection format in line: {line.strip()}")
            continue

        # 取归一化的中心点和宽高
        xc, yc, w_norm, h_norm = map(float, parts[1:5])

        # 转换成像素坐标
        x, y, w, h = denormalize_bbox(xc, yc, w_norm, h_norm, img_width, img_height)

        # 画矩形 (左上角(x, y)，宽w，高h)
        rect = plt.Rectangle((x, y), w, h,
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def visualize_image_and_labels(image_path, label_path, format_type='seg'):
    """
    Visualizes the image with its corresponding YOLO label annotations.
    format_type: 'seg' for segmentation, 'det' for detection
    """
    # Load image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Plot the image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Choose visualization based on format type
    if format_type == 'seg':
        visualize_segmentation(image, label_path, img_width, img_height, ax)
    elif format_type == 'det':
        visualize_detection(image, label_path, img_width, img_height, ax)
    else:
        print(f"Unknown format type: {format_type}")
        return

    # Add title
    ax.set_title(f"{os.path.basename(image_path)} - {format_type.upper()} Format", fontsize=14)
    ax.axis('off')
    plt.show()


def main(data_dir, format_type='seg'):
    """
    Main function to visualize YOLO dataset.
    data_dir: Path to the dataset directory
    format_type: 'seg' for segmentation, 'det' for detection
    """
    print(f"Visualizing dataset in {format_type.upper()} format")

    # Get list of subdirectories (train and val)
    splits = ['train', 'val']

    # Choose randomly between train and val
    split = random.choice(splits)
    print(f"Selected split: {split}")

    # Get list of images and corresponding label files in the selected split
    images_dir = os.path.join(data_dir, 'images', split)
    labels_dir = os.path.join(data_dir, 'labels', split)

    # List all images in the chosen split directory
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    if not images:
        print(f"No images found in {images_dir}")
        return

    while True:
        # Select a random image
        img_file = random.choice(images)
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace(img_file.split('.')[-1], 'txt')
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file for {img_file} not found. Skipping.")
            continue

        print(f"\nDisplaying: {img_file}")

        # Visualize image with labels
        visualize_image_and_labels(img_path, label_path, format_type)

        # Wait for the user to press enter to continue to the next image
        user_input = input("Press Enter to continue to the next image (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset annotations')
    parser.add_argument('--data_dir', type=str,
                        default='/home/lenovo/code/CHT/detect/dataprocess/preprocessed_data2/test/SWRDsize112',
                        help='Path to the dataset directory')
    parser.add_argument('--format', type=str, choices=['seg', 'det'], default='seg',
                        help='Format type: seg (segmentation) or det (detection)')

    args = parser.parse_args()

    # Run main function with specified parameters
    main(args.data_dir, args.format)