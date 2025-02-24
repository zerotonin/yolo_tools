#!/usr/bin/env python3

import os
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz

#
# 1) LABEL NAME CLEANER
#
class LabelNameCleaner:
    """
    Handles unknown or misspelled class names using fuzzy matching.
    Maintains a reference to a class dictionary (class name -> ID).
    """

    def __init__(self, class_dict):
        self.class_dict = class_dict  # e.g. {'arena': 0, 'fly': 1}

    def _find_closest_match(self, unknown_name):
        # Just a minimal fuzzy search among known keys
        best_match, best_score = None, 0
        for known_name in self.class_dict.keys():
            score = fuzz.ratio(unknown_name, known_name)
            if score > best_score:
                best_match, best_score = known_name, score
        return best_match, best_score

    def handle_unknown_class(self, unknown_name):
        """
        Checks if the unknown_name is close to an existing known name.
        Returns the corrected name or None if we decide to ignore.
        """
        best_match, score = self._find_closest_match(unknown_name)
        if best_match is None:
            return None

        # Suppose we accept anything over 70% similarity automatically;
        # you could also prompt the user or handle differently.
        if score > 70:
            print(f"Detected unknown class '{unknown_name}'. Using fuzzy match '{best_match}' (score: {score}).")
            return best_match
        else:
            print(f"Detected unknown class '{unknown_name}', no good match found. Ignoring.")
            return None

    def get_class_id(self, class_name):
        """
        Returns the class ID if found. If not, tries to fix it or ignores it.
        """
        if class_name in self.class_dict:
            return self.class_dict[class_name]

        # Attempt to fix with fuzzy matching
        corrected_name = self.handle_unknown_class(class_name)
        if corrected_name is None:
            # Let's store a placeholder or just return None
            return None
        else:
            # If we adopt the new name, unify the dictionary:
            # e.g. class_dict[unknown_name] = class_dict[corrected_name]
            return self.class_dict[corrected_name]


#
# 2) FILE MANAGER
#
class FileManager:
    """
    Responsible for creating directories, copying files, generating consistent filenames, etc.
    """

    def __init__(self, output_folder, tag=""):
        self.output_folder = output_folder      # e.g. "/path/to/output"
        self.tag = tag                          # e.g. "exp1"

        self.images_dir = os.path.join(self.output_folder, "original_images")
        self.ann_dir = os.path.join(self.output_folder, "annotations")

        self.create_folders()

    def create_folders(self):
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.ann_dir, exist_ok=True)

    def generate_filenames(self, xml_filename, file_no):
        """
        Given the original XML filename and a file index, produce a new image/annotation filename.
        E.g., "image_<tag>_<0001>.png", "image_<tag>_<0001>.txt"
        """
        base, ext = os.path.splitext(xml_filename)  # e.g. "foobar", ".xml"
        new_img_name = f"image_{self.tag}_{str(file_no).zfill(4)}.png"
        new_txt_name = f"image_{self.tag}_{str(file_no).zfill(4)}.txt"
        return new_img_name, new_txt_name

    def copy_image(self, src_image_path, new_img_filename):
        """
        Copies the image file to `original_images/` with the new name.
        """
        dst_path = os.path.join(self.images_dir, new_img_filename)
        shutil.copyfile(src_image_path, dst_path)
        return dst_path

    def get_annotation_path(self, new_txt_filename):
        return os.path.join(self.ann_dir, new_txt_filename)


#
# 3) ANNOTATION CONVERTER
#
class AnnotationConverter:
    """
    Reads Pascal VOC XML, calls LabelNameCleaner for unknown classes, calls FileManager for I/O,
    and generates output for YOLO, RetinaNet, ResNet, MaskRCNN, etc.
    """

    def __init__(self, label_cleaner, file_manager, ai_converter="YOLO"):
        self.label_cleaner = label_cleaner    # instance of LabelNameCleaner
        self.file_manager = file_manager      # instance of FileManager
        self.ai_converter = ai_converter      # "YOLO", "RetinaNet", ...
        self.annotations_coco = {             # For COCO-based formats
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.image_id = 0
        self.annotation_id = 0

    def convert(self, xml_file, file_no):
        """
        The main entry point: decide which format to convert to.
        """
        if self.ai_converter == "YOLO":
            self._convert_to_yolo(xml_file, file_no)
        elif self.ai_converter == "RetinaNet":
            self._convert_to_retinanet(xml_file, file_no)
        elif self.ai_converter == "ResNet":
            self._convert_to_resnet(xml_file, file_no)
        elif self.ai_converter == "Mask_RCNN":
            self._convert_to_maskrcnn(xml_file, file_no)
        else:
            raise ValueError(f"Unknown converter: {self.ai_converter}")

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        filename = root.find("filename").text

        objects = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            objects.append((class_name, (xmin, ymin, xmax, ymax)))

        return filename, width, height, objects

    #
    # 3.1. YOLO
    #
    def _convert_to_yolo(self, xml_file, file_no):
        filename, w, h, objects = self._parse_xml(xml_file)
        # Generate new filenames
        new_img_name, new_txt_name = self.file_manager.generate_filenames(filename, file_no)

        # Copy image
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        src_image_path = os.path.join(os.path.dirname(xml_file), img_base_name + ".png")
        self.file_manager.copy_image(src_image_path, new_img_name)

        # YOLO annotation path
        txt_path = self.file_manager.get_annotation_path(new_txt_name)

        # Write bounding boxes
        with open(txt_path, "w") as f:
            for class_name, (xmin, ymin, xmax, ymax) in objects:
                class_id = self.label_cleaner.get_class_id(class_name)
                if class_id is None:
                    continue  # skip unknown classes
                box_w = (xmax - xmin) / float(w)
                box_h = (ymax - ymin) / float(h)
                cx = (xmin + xmax) / 2.0 / float(w)
                cy = (ymin + ymax) / 2.0 / float(h)

                f.write(f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n")

    #
    # 3.2. RetinaNet (COCO style) 
    #
    def _convert_to_retinanet(self, xml_file, file_no):
        filename, w, h, objects = self._parse_xml(xml_file)
        new_img_name, _ = self.file_manager.generate_filenames(filename, file_no)

        # Copy image
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        src_image_path = os.path.join(os.path.dirname(xml_file), img_base_name + ".png")
        self.file_manager.copy_image(src_image_path, new_img_name)

        # Add to self.annotations_coco["images"]
        image_info = {
            "id": self.image_id,
            "file_name": new_img_name,
            "width": w,
            "height": h
        }
        self.annotations_coco["images"].append(image_info)

        # For each bounding box, add to "annotations"
        for class_name, (xmin, ymin, xmax, ymax) in objects:
            class_id = self.label_cleaner.get_class_id(class_name)
            if class_id is None:
                continue
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue  # skip invalid boxes

            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": class_id,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            }
            self.annotations_coco["annotations"].append(annotation)
            self.annotation_id += 1

        self.image_id += 1

    #
    # 3.3. ResNet
    #
    def _convert_to_resnet(self, xml_file, file_no):
        """
        Could store a simple CSV mapping or some other simpler format.
        We'll just copy the image here for demonstration.
        """
        filename, w, h, objects = self._parse_xml(xml_file)
        new_img_name, _ = self.file_manager.generate_filenames(filename, file_no)

        # Copy image
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        src_image_path = os.path.join(os.path.dirname(xml_file), img_base_name + ".png")
        self.file_manager.copy_image(src_image_path, new_img_name)
        # e.g., we might do something else with bounding boxes or classes if needed...

    #
    # 3.4. Mask R-CNN (COCO style with segmentation)
    #
    def _convert_to_maskrcnn(self, xml_file, file_no):
        filename, w, h, objects = self._parse_xml(xml_file)
        new_img_name, _ = self.file_manager.generate_filenames(filename, file_no)

        # Copy image
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        src_image_path = os.path.join(os.path.dirname(xml_file), img_base_name + ".png")
        self.file_manager.copy_image(src_image_path, new_img_name)

        image_info = {
            "id": self.image_id,
            "file_name": new_img_name,
            "width": w,
            "height": h
        }
        self.annotations_coco["images"].append(image_info)

        for class_name, (xmin, ymin, xmax, ymax) in objects:
            class_id = self.label_cleaner.get_class_id(class_name)
            if class_id is None:
                continue
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            # for a naive bounding-box polygon
            seg = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": class_id,
                "bbox": [xmin, ymin, bw, bh],
                "segmentation": seg,
                "area": bw * bh,
                "iscrowd": 0
            }
            self.annotations_coco["annotations"].append(annotation)
            self.annotation_id += 1

        self.image_id += 1

    def finalize_coco_output(self, class_dict):
        """
        If using COCO-based formats (RetinaNet or Mask R-CNN), 
        finalize categories, then return the big dictionary.
        """
        # Build categories
        categories = []
        for cname, cid in class_dict.items():
            if cid is not None:
                categories.append({"id": cid, "name": cname})
        self.annotations_coco["categories"] = categories

        return self.annotations_coco


#
# 4) COMMAND-LINE ENTRY POINT
#
def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to a chosen format.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("ai_converter", help="One of: YOLO, RetinaNet, ResNet, Mask_RCNN")
    parser.add_argument("--tag", default="", help="Tag to prefix new filenames.")
    parser.add_argument("--class_dict", type=str, help="JSON file with class->ID mapping.")

    args = parser.parse_args()

    # Load or define a default dictionary
    if args.class_dict and os.path.exists(args.class_dict):
        with open(args.class_dict, "r") as f:
            class_dict = json.load(f)
    else:
        class_dict = {"arena": 0, "fly": 1}  # example fallback

    # Instantiate each piece
    label_cleaner = LabelNameCleaner(class_dict)
    file_manager = FileManager(args.output_folder, args.tag)
    converter = AnnotationConverter(label_cleaner, file_manager, ai_converter=args.ai_converter)

    # Process all .xml in the input folder
    file_count = 0
    for root_dir, _, files in os.walk(args.xml_folder):
        for fname in files:
            if fname.lower().endswith(".xml"):
                xml_path = os.path.join(root_dir, fname)
                try:
                    converter.convert(xml_path, file_count)
                    file_count += 1
                except Exception as e:
                    print(f"Error converting {xml_path}: {e}")

    # If we did RetinaNet or Mask R-CNN, we might want to save the COCO JSON
    if args.ai_converter in ["RetinaNet", "Mask_RCNN"]:
        coco_data = converter.finalize_coco_output(class_dict)
        out_path = os.path.join(args.output_folder, "annotations", "annotations.json")
        with open(out_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"COCO annotations saved to: {out_path}")

    print("Done!")

if __name__ == "__main__":
    main()
