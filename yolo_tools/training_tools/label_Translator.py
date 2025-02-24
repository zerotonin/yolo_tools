import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz  
import os
from tabulate import tabulate  # If you're using tabulate
import shutil
import argparse
import json



class AnnotationConverter:
    # definition for init method or constructor
    counter = 0 # Class variable???

     
    def __init__(self, output_dir, class_dict, tag, ai_converter): 

        # __init__: constructer (special method). Here Parameterised __init__ constructer (__init__(self, parameters,...))
        # automatically calld when a new instance (object) of a class is created.
        # It allows you to set up the initial state of an object by assigning values to its attributes or performing other setup tasks.
        # __init__ allows you to initialize tha attributes (variables) of an object.
        # self: default parameter. Always passed in its arguments. Represents the object of the class itself.
        # __: method is invoked and used internally in Python. No calling necessary.
        # __init__(self,...): takes arguments for object initialisation. Any other parameters to initialize the object.
        # This allows you to create objects with default attribute values if no arguments are provided for those parameters.

        # __init__ initializes the output_dir, class_dict and tag attributes of the Class AnnotationConverter.
        """
        Create the necessary output folders for images and annotations.

        The method ensures the existence of the 'original_images' and 'annotations'
        subdirectories within the specified output directory.
        """

        self.output_dir = output_dir # instance variable
        self.output_folder = os.path.join(output_dir, 'annotations')
        self.class_dict = class_dict
        self.tag = tag
        self.annotations = {"images": [], "annotations": [], "categories": []}
        self.image_id = 0
        self.images = []
        self.annotation_id = 0
        self._create_output_folders()
        self._create_categories()
        self._ai_converters = {
            "YOLO": self.convert_XML_to_YOLO,
            "RetinaNet": self.convert_XML_to_RetinaNet, 
            "ResNet": self.convert_XML_to_ResNet, 
            "Mask_RCNN": self.convert_XML_to_MaskRCNN,
        }
        self._chosen_ai_converter = self._ai_converters[ai_converter]



    def convert_XML_file(self, xml_file, file_no):
        self._chosen_ai_converter(xml_file, file_no)
        # print("do I get here?")

    def _create_output_folders(self):
        """
        Handle misspelled class names by suggesting possible matches and allowing user input.

        Args:
            misspelled_word (str): The misspelled class name.
            default_match (str): The default suggested match for the misspelled word.

        Returns:
            str or None: The corrected class name chosen by the user, or None if ignored.
        """
        os.makedirs(os.path.join(self.output_dir, "original_images"), exist_ok=True) 
        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)

    # Class method:
    def _create_categories(self):
        for class_name, class_id in self.class_dict.items():
            self.annotations["categories"].append({"id": class_id, "name": class_name})




    def _generate_filenames(self, xml_filename, file_no):
        """
        Get the image path corresponding to an XML file.

        Args:
            xml_file (str): The path to the XML file.

        Returns:
            str: The path to the corresponding image file.
        """
        file_no = str(file_no) # converts the int. file_no into a string.
        xml_base_name, xml_extension = os.path.splitext(xml_filename)  
        new_img_filename = f"image_{self.tag}_{file_no.zfill(4)}{xml_extension}"  
        new_txt_filename = f"image_{self.tag}_{file_no.zfill(4)}.txt"
        return new_img_filename, new_txt_filename, file_no
    



    def _get_image_path(self, xml_file):
        """
        Copy the image file to the output directory with a new filename.

        Args:
            img_path (str): The original path to the image file.
            new_img_filename (str): The new filename for the image file.
        """
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')  

    

    def _read_XML_file(self, xml_file, file_no, return_object_bndbox=False):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_width = int(root.find('size').find('width').text) # finds the size and width in root and saves it as .text (int) in img_widht
        img_height = int(root.find('size').find('height').text) # finds the size and height in root, saves it as .text (int) in img_height
        # bndbox = root.find('bndbox') # finds the bndbox in root and saves it in bndbox
        # xmin = int(bndbox.find('xmin').text) # finds xmin in bndbox, saves as .text (int) in xmin
        # print(bndbox.find('ymin').text)
        # ymin = int(bndbox.find('ymin').text) # see above
        # xmax = int(bndbox.find('xmax').text)
        # ymax = int(bndbox.find('ymax').text)
        img_filename = root.find('filename').text # finds filename in root, saves as .text in img_filename.

        image_dimensions = {
            "img_height": img_height,
            "img_width": img_width,
        }
        new_img_filename, new_txt_filename, file_no = self._generate_filenames(img_filename, file_no) 
        img_path = self._get_image_path(xml_file)
        txt_output_path = os.path.join(self.output_dir, "annotations", new_txt_filename)
        

        file_data = {
            "new_img_filename": new_img_filename,
            "new_txt_filename": new_txt_filename,
            "txt_output_path": txt_output_path,
            "img_path": img_path,
        }
        
        object_classes = {}
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = self._get_class_id(class_name)
              # Allow None for ignored entries
            if return_object_bndbox:
                obj_bndbox = obj.find('bndbox')
                obj_xmin = int(obj_bndbox.find('xmin').text)
                obj_ymin = int(obj_bndbox.find('ymin').text)
                obj_xmax = int(obj_bndbox.find('xmax').text)
                obj_ymax = int(obj_bndbox.find('ymax').text)
                object_classes[class_id] = [class_name, {"xmin": obj_xmin, "xmax": obj_xmax, "ymin": obj_ymin, "ymax": obj_ymax}]
            else:  
                object_classes[class_id] = class_name
                
            
            

        return object_classes, image_dimensions, file_data

    
    
    def _copy_image(self, img_path, new_img_filename):
        """
        Update the class dictionary with the correct class name or mark it as ignored.

        Args:
            misspelled_word (str): The misspelled class name.
            correct_name (str or None): The correct class name, or None if ignored.
        """
        img_output_path = os.path.join(self.output_dir, "original_images", new_img_filename)
        shutil.copyfile(img_path, img_output_path)


    def _get_class_id(self,class_name):
        if class_name not in self.class_dict:
            correct_name = self._handle_unknown_class(class_name)
        else:
            correct_name = class_name 

        class_id = self.class_dict.get(correct_name) 

        return class_id              
    


############### YOLO #######################

    def _write_YOLO_annotation(self, txt_output_path, class_id, obj_dims, img_dims):
        """
        Write the YOLO annotation for a given object.

        Args:
            txt_output_path (str): The path to the output text file.
            class_id (int): The class ID of the object.
            obj (xml.etree.ElementTree.Element): The XML element of the object.
            img_width (int): The width of the image.
            img_height (int): The height of the image.
        """
               

        box_w = (obj_dims["xmax"] - obj_dims["xmin"]) / img_dims["img_width"]
        box_h = (obj_dims["ymax"] - obj_dims["ymin"]) / img_dims["img_height"]
        box_cx = (obj_dims["xmin"] + obj_dims["xmax"]) / 2 / img_dims["img_width"]
        box_cy = (obj_dims["ymin"] + obj_dims["ymax"]) / 2 / img_dims["img_height"]

        with open(txt_output_path, 'a') as txt_file:  # Using append mode 
            txt_file.write(f"{class_id} {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f}\n")


    def convert_XML_to_YOLO(self, xml_file, file_no):
        """
        Convert an XML annotation file to YOLO format and save the output.

        Args:
            xml_file (str): The path to the XML file.
            file_no (int): The file number to be included in the new filename.
        """    
        object_classes, image_dimensions, file_data = self._read_XML_file(xml_file, file_no, return_object_bndbox=True)

        self._copy_image(file_data["img_path"], file_data["new_img_filename"])

        for (class_id, object_info) in object_classes.items():
            if class_id is not None:
                self._write_YOLO_annotation(file_data["txt_output_path"], class_id, object_info[1], image_dimensions)  


############## RetinaNet ################

    def _write_COCO_annotation(self, class_id, obj_dims, img_dim):

        width = obj_dims["xmax"] - obj_dims["xmin"]
        height = obj_dims["ymax"] - obj_dims["ymin"]
        
        self.annotations["annotations"].append({
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": class_id,
            "bbox": [obj_dims["xmin"], obj_dims["ymin"], width, height], # ich glaube das ist der Fehler!!!
            "area": width * height,
            "iscrowd": 0
        })
        self.annotation_id += 1


    def convert_XML_to_RetinaNet(self, xml_file, file_no):
        object_classes, image_dimensions, file_data = self._read_XML_file(xml_file, file_no, return_object_bndbox = True)

        self._copy_image(file_data["img_path"], file_data["new_img_filename"])
        
        self.annotations["images"].append({
            "id": self.image_id,
            "file_name": file_data["new_img_filename"],
            "width": image_dimensions["img_width"],
            "height": image_dimensions["img_height"],
        })

        for (class_id, object_info) in object_classes.items():
            if class_id is not None:
                self._write_COCO_annotation(class_id, object_info[1], image_dimensions)
        self.image_id += 1
        
    def save_annotations(self, output_file):
        output_file = os.path.join(self.output_folder, 'annotations.json')
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)  



################ ResNet #########################

    def convert_XML_to_ResNet(self, xml_file, file_no):
        """
        Convert an XML annotation file to a format compatible with ResNet.

        Args:
            xml_file (str): The path to the XML file.
            file_no (int): The file number to be included in the new filename.
        """

        object_classes, image_dimensions, file_data = self._read_XML_file(xml_file, file_no)

        self._copy_image(file_data["img_path"], file_data["new_img_filename"])

        # 1. Reading the XML File - calls the method _read_XML_file to extract information from an XML file.
        # 2. Copying the Image - copies the image file from its original location to a new location or renames it.


################ Mask R-CNN #######################

    def convert_XML_to_MaskRCNN(self, xml_file, file_no):
        """
        Convert an XML annotation file to Mask R-CNN format and save the output.

        Args:
            xml_file (str): The path to the XML file.
        """
        
        object_classes, image_dimensions, file_data = self._read_XML_file(xml_file, file_no, return_object_bndbox=True)
        
        self._copy_image(file_data["img_path"], file_data["new_img_filename"])

        # Add image info
        image_info = {
            "id": self.image_id,
            "file_name": file_data["new_img_filename"],
            "width": image_dimensions["img_width"],
            "height": image_dimensions["img_height"],
        }
        self.images.append(image_info)
        
        self.image_id += 1 

        
        for (class_name, bndbox) in object_classes.values():
            if class_name in self.class_dict:

                class_id = self.class_dict[class_name]

                # Get bounding box coordinates
                xmin = bndbox["xmin"]
                xmax = bndbox["xmax"]
                ymin = bndbox["ymin"]
                ymax = bndbox["ymax"]
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
                    "bbox": [bndbox["xmin"], bndbox["ymin"], width, height],
                    "segmentation": segmentation,
                    "area": width * height,
                    "iscrowd": 0
                }
                self.annotations["images"].append(annotation_info)

    def save_to_json(self):
        """
        Save the converted annotations to a JSON file in COCO format.
        """
        output_file = os.path.join(self.output_folder, 'coco_annotations.json')
        coco_format = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": v, "name": k} for k, v in self.class_dict.items()]
        }
        
        with open(output_file, 'w') as f:
            json.dump(coco_format, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to selected format.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("ai_converter", help="AI converter: YOLO, RetinaNet, ResNet, Mask_RCNN")
    parser.add_argument("--tag", help="Tag to prefix the output file names.", default="")
    parser.add_argument("--class_dict", type=str, help="Path to a JSON file containing the class dictionary.")

    args = parser.parse_args()

    if args.ai_converter not in ["YOLO", "RetinaNet", "ResNet", "Mask_RCNN"]:
        print("Converter not recognized. Try again.")
    else:

        if args.class_dict:
            import json
            with open(args.class_dict, "r") as f:
                class_dict = json.load(f)
        else:
            # Example class dictionary
            class_dict = {'arena': 0, 'fly': 1}  
            
        converter = AnnotationConverter(args.output_folder, class_dict, args.tag, args.ai_converter)

        file_count = 0
        for root, _, filenames in os.walk(args.xml_folder):
            for xml_filename in filenames:
                if xml_filename.endswith('.xml'):
                    try:
                        xml_path = os.path.join(root, xml_filename)   
                        converter.convert_XML_file(xml_path, file_count)
                        file_count += 1
                    except:
                        print(f'error transforming: {xml_filename}')
        if args.ai_converter in ["Mask_RCNN", "RetinaNet"]:
            converter.save_to_json()
            # converter.save_to_json(args.output_folder)


if __name__ == "__main__":
    main()      





