"""
Script for YOLOv8 model training and prediction.

This script automates the following tasks:

1. Creates a YOLOv8 dataset from original images and labels.
2. Generates a YOLOv8 configuration file.
3. Trains a YOLOv8 model using transfer learning.
4. Predicts bounding boxes on a validation set and saves results to a CSV.
5. Draws predicted bounding boxes on validation images.

Usage:
  python yolov8_train_predict.py -d <dataset_dir> -i <images_dir> -l <labels_dir> -c <config_file> [-n <model_name>] [-e <epochs>] [--classes <class1> <class2> ...]
"""

import argparse
from pathlib import Path
import os

# Import the YOLOv8 wrapper module (assumed to be available)
from yolo_tools.training_tools.YoloWrapper import YoloWrapper


def main():
    """Main function to handle command-line arguments and execute the workflow."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and predict with YOLOv8.")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Path to the directory where the YOLO dataset will be created.')
    parser.add_argument('-i', '--images', type=str, required=True,
                        help='Path to the directory containing the original images.')
    parser.add_argument('-l', '--labels', type=str, required=True,
                        help='Path to the directory containing the labels (annotations).')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the YOLO configuration file to be created.')
    parser.add_argument('-n', '--name', type=str, default='yolo_model',
                        help='Name for the trained model (default: yolo_model).')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200).')
    parser.add_argument('--classes', nargs='+', default=['arena', 'fly'],
                        help='List of class names in the order they appear in the training data (default: arena fly).')
    args = parser.parse_args()

    # Create Path objects for easier file/directory handling
    dataset_path = Path(args.dataset)
    large_field_images_path = Path(args.images)
    labels_path = Path(args.labels)
    config_path = args.config

    # Create the YOLO dataset
    print("Creating YOLO dataset...")
    YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
    print("YOLO dataset created.")

    # Create the YOLO configuration file
    print("Creating YOLO configuration file...")
    YoloWrapper.create_config_file(dataset_path, args.classes, config_path)
    print("YOLO configuration file created.")

    # Enable expandable memory for PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Train the YOLO model
    print("Training YOLO model...")
    model = YoloWrapper('small')  # Assuming 'small' refers to a model size/type
    model.train(config_path, epochs=args.epochs, name=args.name)
    print("YOLO model training completed.")

    # Make predictions on the validation set
    data_to_predict_path = dataset_path / 'images' / 'val'
    val_image_list = list(data_to_predict_path.glob('*.png'))  # Get all PNG images in the validation folder
    csv_path = f'{args.name}.csv'  # Construct the CSV filename based on the model name

    print("Making predictions and saving results to CSV...")
    model.predict_and_save_to_csv(val_image_list, path_to_save_csv=csv_path, minimum_size=100, threshold=0.25,
                                 only_most_conf=True)

    # Draw bounding boxes on the validation images
    print("Drawing bounding boxes on validation images...")
    for image in val_image_list:
        model.draw_bbox_from_csv(image, csv_path, image.stem) 

if __name__ == '__main__':
    main()
