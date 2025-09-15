import os
import random
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


def visualize_image_and_labels(image_path, label_path):
    """
    Visualizes the image with its corresponding YOLO label annotations.
    """
    # Load image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Read the labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Plot the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for line in lines:
        coords = list(map(float, line.strip().split()[1:]))  # Ignore class label (first value)
        coords = denormalize_coordinates(coords, img_width, img_height)
        coords = np.array(coords)

        # Plot the polygon (closed shape)
        plt.fill(coords[:, 0], coords[:, 1], 'r', alpha=0.3)

    # Show plot with labels
    plt.axis('off')
    plt.show()


def main(data_dir):
    # Get list of subdirectories (train and val)
    splits = ['train', 'val']

    # Choose randomly between train and val
    split = random.choice(splits)

    # Get list of images and corresponding label files in the selected split
    images_dir = os.path.join(data_dir, 'images', split)
    labels_dir = os.path.join(data_dir, 'labels', split)

    # List all images in the chosen split directory
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    while True:
        # Select a random image
        img_file = random.choice(images)
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace(img_file.split('.')[-1], 'txt')
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file for {img_file} not found. Skipping.")
            continue

        # Visualize image with labels
        visualize_image_and_labels(img_path, label_path)

        # Wait for the user to press enter to continue to the next image
        input("Press Enter to continue to the next image...")


if __name__ == "__main__":
    # Set your data directory path
    data_dir = '/home/lenovo/code/CHT/detect/dataprocess/preprocessed_data/YOLODataset_seg'
    main(data_dir)
