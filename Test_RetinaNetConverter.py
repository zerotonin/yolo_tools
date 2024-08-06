import os
import json
import shutil
import xml.etree.ElementTree as ET

class AnnotationConverter:
    def __init__(self, output_dir, class_dict, tag):
        self.output_dir = output_dir
        self.class_dict = class_dict
        self.tag = tag
        self.annotations = {"images": [], "annotations": [], "categories": []}
        self.image_id = 0
        self.annotation_id = 0
        self._create_output_folders()
        self._create_categories()

    def _create_output_folders(self):
        os.makedirs(os.path.join(self.output_dir, "original_images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)

    def _create_categories(self):
        for class_name, class_id in self.class_dict.items():
            self.annotations["categories"].append({"id": class_id, "name": class_name})

    def _generate_filenames(self, xml_filename):
        xml_base_name, xml_extension = os.path.splitext(xml_filename)
        new_img_filename = f"{self.tag}_{self.image_id:06d}{xml_extension}"
        return new_img_filename

    def _get_image_path(self, xml_file): 
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')

    def _copy_image(self, img_path, new_img_filename): 
        img_output_path = os.path.join(self.output_dir, "original_images", new_img_filename)
        print("image_path", img_path)
        print("img_output_path", img_output_path)
        shutil.copyfile(img_path, img_output_path)

    def _get_class_id(self, class_name):
        return self.class_dict.get(class_name, None)

    def _write_COCO_annotation(self, class_id, obj, img_width, img_height):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin

        self.annotations["annotations"].append({
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": class_id,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0
        })
        self.annotation_id += 1

    def convert_XML_to_RetinaNet(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        img_filename = root.find('filename').text

        new_img_filename = self._generate_filenames(img_filename)
        img_path = self._get_image_path(xml_file)

        self._copy_image(img_path, new_img_filename)

        self.annotations["images"].append({
            "id": self.image_id,
            "file_name": new_img_filename,
            "width": img_width,
            "height": img_height
        })

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = self._get_class_id(class_name)

            if class_id is not None:
                self._write_COCO_annotation(class_id, obj, img_width, img_height)

        self.image_id += 1

    def save_annotations(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)

def main():
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser(description="Convert XML annotations to COCO format for RetinaNet.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("--tag", help="Tag to prefix the output file names.", default="")
    parser.add_argument("--class_dict", type=str, help="Path to a JSON file containing the class dictionary.")

    args = parser.parse_args()

    if args.class_dict:
        with open(args.class_dict, "r") as f:
            class_dict = json.load(f)
    else:
        class_dict = {'arena': 0, 'fly': 1}  # Example class dictionary

    args.xml_folder = Path(args.xml_folder)
    args.output_folder = Path(args.output_folder)
    converter = AnnotationConverter(args.output_folder, class_dict, args.tag)

    for root, _, filenames in os.walk(args.xml_folder):
        for xml_filename in filenames:
            if xml_filename.endswith('.xml'):
                try:
                    xml_path = os.path.join(root, xml_filename)
                    converter.convert_XML_to_RetinaNet(xml_path)
                except Exception as e:
                    print(f'Error processing {xml_filename}: {e}')

    converter.save_annotations(os.path.join(args.output_folder, 'annotations.json'))

if __name__ == "__main__":
    main()
