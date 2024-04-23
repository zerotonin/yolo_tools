import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz  
import os
from tabulate import tabulate  # If you're using tabulate
import shutil

class AnnotationConverter:
    def __init__(self, output_dir, class_dict, tag):
        self.output_dir = output_dir
        self.class_dict = class_dict
        self.tag = tag
        self._create_output_folders()

    def _create_output_folders(self):
        os.makedirs(os.path.join(self.output_dir, "original_images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)

    def _handle_misspelled_class(self, misspelled_word,default_match):
        best_match = default_match
        while True:
            print("Potential Misspelling Detected!")
            print(f"Word: {misspelled_word}")

            # Optional: Show some possible matches
            matches = [(class_name, fuzz.ratio(misspelled_word, class_name)) 
                    for class_name in class_dict.keys()]
            matches.sort(key=lambda item: item[1], reverse=True) 
            print(tabulate(matches[:3], headers=["Class Name", "Similarity Score"])) 

            print("\nOptions:")
            print("1. Use suggested match ({})".format(default_match))
            print("2. Choose a different match")
            print("3. Ignore this entry")

            choice = input("Enter your choice (1, 2, or 3): ")

            if choice == '1':            
                break  # Accept default
            elif choice == '2':
                new_match = input("Enter the correct class name: ")
                if new_match in class_dict:
                    best_match = new_match
                    break
                else:
                    print("Invalid class name.")
            elif choice == '3':
                return None  # Ignore
            else:
                print("Invalid choice. Please try again.")
        
        return best_match 

    def _find_closest_match(self, misspelled_word):
        ratios = [(class_name, fuzz.ratio(misspelled_word, class_name)) 
                for class_name in self.class_dict.keys()]
        best_match, best_score = max(ratios, key=lambda item: item[1])
        return best_match, best_score

    def _generate_filenames(self, xml_filename, file_no):
        file_no = str(file_no)
        xml_base_name, xml_extension = os.path.splitext(xml_filename)  
        new_img_filename = f"image_{self.tag}_{file_no.zfill(4)}{xml_extension}"  
        new_txt_filename = f"image_{self.tag}_{file_no.zfill(4)}.txt"
        return new_img_filename, new_txt_filename

    def _get_image_path(self, xml_file):
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')  

    def _copy_image(self, img_path, new_img_filename):
        img_output_path = os.path.join(self.output_dir, "original_images", new_img_filename)
        shutil.copyfile(img_path, img_output_path)

    def _update_class_dictionary(self, misspelled_word, correct_name):
        if correct_name == None:
            self.class_dict[misspelled_word] = None
        else:
            self.class_dict[misspelled_word] = self.class_dict[correct_name]

    def _handle_unknown_class(self, class_name):
        best_match, best_score = self._find_closest_match(class_name)
        correct_name = self._handle_misspelled_class(class_name, best_match)
        self._update_class_dictionary(class_name, correct_name)  
        return correct_name
    
    def _get_class_id(self,class_name):
        if class_name not in self.class_dict:
            correct_name = self._handle_unknown_class(class_name)
        else:
            correct_name = class_name 

        class_id = self.class_dict.get(correct_name) 
        return class_id 
    

    def _write_YOLO_annotation(self, txt_output_path, class_id, obj, img_width, img_height):
               
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        box_w = (xmax - xmin) / img_width
        box_h = (ymax - ymin) / img_height
        box_cx = (xmin + xmax) / 2 / img_width
        box_cy = (ymin + ymax) / 2 / img_height

        with open(txt_output_path, 'a') as txt_file:  # Using append mode 
            txt_file.write(f"{class_id} {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f}\n")

    def convert_XML_to_YOLO(self, xml_file, file_no):    
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        img_filename = root.find('filename').text

        new_img_filename, new_txt_filename = self._generate_filenames(img_filename, file_no)
        img_path = self._get_image_path(xml_file)
        txt_output_path = os.path.join(self.output_dir, "annotations", new_txt_filename)

        self._copy_image(img_path, new_img_filename)

        for obj in root.findall('object'):            
            class_name = obj.find('name').text
            class_id = self._get_class_id(class_name)  # Allow None for ignored entries

            if class_id is not None:
                self._write_YOLO_annotation(txt_output_path, class_id, obj, img_width, img_height)


# Example Usage:
xml_folder = '/home/geuba03p/label_test/labeldDataLena'
output_folder = '/home/geuba03p/output'  
class_dict = {'arena': 0, 'fly': 1}  
tag = 'food_2Choice'

converter = AnnotationConverter(output_folder, class_dict, tag)

file_count = 0
for root, _, filenames in os.walk(xml_folder):
    for xml_filename in filenames:
        if xml_filename.endswith('.xml'):
            try:
                xml_path = os.path.join(root, xml_filename)   
                converter.convert_XML_to_YOLO(xml_path, file_count)
                file_count += 1
            except:
                print(f'error transforming: {xml_filename}')
