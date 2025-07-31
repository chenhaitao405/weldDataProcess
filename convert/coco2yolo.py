from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/home/num2/CODE/detect/dataprocess/preprocessed_data/labels/L/1",
    save_dir="output_dir",
    use_keypoints=True,
)
