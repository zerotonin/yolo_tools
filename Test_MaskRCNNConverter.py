import os
import json
import xml.etree.ElementTree as ET
import argparse
import shutil
from pathlib import Path

class AnnotationConverterMaskRCNN:
    """
    A class to convert XML annotations to Mask R-CNN (COCO) format.

    Attributes:
        output_file (str): The path to the output JSON file.
        class_dict (dict): A dictionary mapping class names to class IDs.
        image_id (int): An incremental ID for images.
        annotation_id (int): An incremental ID for annotations.
        images (list): A list to store image information.
        annotations (list): A list to store annotation information.
    """
    
    def __init__(self, output_file, class_dict):
        self.output_file = output_file
        self.class_dict = class_dict
        self.image_id = 0
        self.annotation_id = 0
        self.images = []
        self.annotations = []

    def convert_XML_to_MaskRCNN(self, xml_file):
        """
        Convert an XML annotation file to Mask R-CNN format and save the output.

        Args:
            xml_file (str): The path to the XML file.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        img_filename = root.find('filename').text

        # Add image info
        self.image_id += 1
        image_info = {
            "id": self.image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_filename
        }
        self.images.append(image_info)

        for obj in root.findall('object'):
            class_name = obj.find('name').text

            if class_name not in self.class_dict:
                continue  # Skip unknown classes

            class_id = self.class_dict[class_name]

            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            width = xmax - xmin
            height = ymax - ymin

            # Calculate segmentation (here, using bounding box as polygon)
            segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

            # Add annotation info
            self.annotation_id += 1
            annotation_info = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": class_id,
                "bbox": [xmin, ymin, width, height],
                "segmentation": segmentation,
                "area": width * height,
                "iscrowd": 0
            }
            self.annotations.append(annotation_info)

    def save_to_json(self):
        """
        Save the converted annotations to a JSON file in COCO format.
        """
        coco_format = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": v, "name": k} for k, v in self.class_dict.items()]
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(coco_format, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to Mask R-CNN format.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    parser.add_argument("--class_dict", type=str, help="Path to a JSON file containing the class dictionary.")

    args = parser.parse_args()

    if args.class_dict:
        import json
        with open(args.class_dict, "r") as f:
            class_dict = json.load(f)
    else:
        # Example class dictionary
        class_dict = {'arena': 0, 'fly': 1}

    converter = AnnotationConverterMaskRCNN(args.output_file, class_dict)

    for root, _, filenames in os.walk(args.xml_folder):
        for xml_filename in filenames:
            if xml_filename.endswith('.xml'):
                xml_path = os.path.join(root, xml_filename)
                converter.convert_XML_to_MaskRCNN(xml_path)

    converter.save_to_json()

if __name__ == "__main__":
    main()

    # python convert_to_mask_rcnn.py <xml_folder> <output_file> --class_dict <class_dict.json>
    # python convert_to_mask_rcnn.py annotations output_annotations.json