#!/usr/bin/env python3

import os
import argparse
import shutil
import random
import xml.etree.ElementTree as ET
import yaml

"""
Example usage:
python voc_to_ultralytics.py \
  --xml_folder /home/geuba03p/label_test/labeledDataLena \
  --output_folder /home/geuba03p/label_test/uLytics_Lena\
  --split 0.8 \
  --classes arena fly

The result will be something like:
output_ultra/
  ├─ images/
  │   ├─ train/
  │   └─ val/
  ├─ labels/
  │   ├─ train/
  │   └─ val/
  └─ data.yaml
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Convert VOC XML to Ultralytics YOLO format.")
    parser.add_argument("--xml_folder", required=True, help="Folder containing PascalVOC .xml annotation files + images.")
    parser.add_argument("--output_folder", required=True, help="Where to put images/, labels/, data.yaml.")
    parser.add_argument("--classes", nargs='+', required=True, help="List of class names, e.g. --classes arena fly.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio (0.8 = 80% train).")
    parser.add_argument("--img_ext", default=".png",
                        help="Image extension if not .jpg, e.g. .png. Script will look for <xml_basename>.png.")
    parser.add_argument("--copy_images", action="store_true",
                        help="If set, copy images. Otherwise we move them (faster, but original is removed).")
    return parser.parse_args()

def voc_to_ultra(xml_folder, output_folder, classes, split_ratio=0.8, img_ext=".png", copy_images=False):
    # 1) Prepare folder structure
    images_train_dir = os.path.join(output_folder, "images", "train")
    images_val_dir   = os.path.join(output_folder, "images", "val")
    labels_train_dir = os.path.join(output_folder, "labels", "train")
    labels_val_dir   = os.path.join(output_folder, "labels", "val")

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    # 2) Gather all XML files
    xml_files = []
    for root, dirs, files in os.walk(xml_folder):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    random.shuffle(xml_files)

    # 3) Split
    split_index = int(len(xml_files) * split_ratio)
    train_xmls = xml_files[:split_index]
    val_xmls   = xml_files[split_index:]

    # 4) Helper to convert PascalVOC -> YOLO txt lines
    def voc_to_yolo_lines(xml_path):
        # parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            # skip if not in classes
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # YOLO format: class_id center_x center_y width height (all normalized 0-1)
            cx = (xmin + xmax) / 2.0 / w
            cy = (ymin + ymax) / 2.0 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return yolo_lines

    # 5) Convert & copy/move
    def process_set(xml_list, images_out_dir, labels_out_dir):
        for xml_path in xml_list:
            base = os.path.splitext(os.path.basename(xml_path))[0]  # e.g. "image_0001"
            image_src = os.path.join(os.path.dirname(xml_path), base + img_ext)
            # create YOLO label lines
            yolo_lines = voc_to_yolo_lines(xml_path)

            # if no lines, we can still produce an empty txt (some YOLO versions want that)
            label_txt_path = os.path.join(labels_out_dir, base + ".txt")
            with open(label_txt_path, "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")

            # copy or move the image
            if os.path.isfile(image_src):
                image_dst = os.path.join(images_out_dir, base + img_ext)
                if copy_images:
                    shutil.copyfile(image_src, image_dst)
                else:
                    shutil.move(image_src, image_dst)
            else:
                print(f"WARNING: image not found {image_src}, skipping.")

    # process train set
    process_set(train_xmls, images_train_dir, labels_train_dir)
    # process val set
    process_set(val_xmls, images_val_dir, labels_val_dir)

    # 6) Write data.yaml for Ultralytics
    data_yaml_path = os.path.join(output_folder, "data.yaml")
    data_dict = {
        "train": os.path.join(output_folder, "images", "train"),
        "val": os.path.join(output_folder, "images", "val"),
        # "test": ... # optional
        "names": classes
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_dict, f)
    print(f"Ultralytics data.yaml created at: {data_yaml_path}")
    print("Done.")

def main():
    args = parse_args()
    voc_to_ultra(
        xml_folder=args.xml_folder,
        output_folder=args.output_folder,
        classes=args.classes,
        split_ratio=args.split,
        img_ext=args.img_ext,
        copy_images=args.copy_images
    )

if __name__ == "__main__":
    main()
