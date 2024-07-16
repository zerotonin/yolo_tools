import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz  
import os
from tabulate import tabulate  # If you're using tabulate
import shutil
import argparse

class AnnotationConverter:
    """
    A class to convert XML annotations to YOLO format, handling potential misspellings and 
    organizing the output into a specified directory structure.

    This class provides methods to:
    - Create necessary output folders.
    - Handle misspelled class names by suggesting possible matches.
    - Generate filenames for images and annotations.
    - Retrieve image paths from XML files.
    - Copy images to the output directory.
    - Update class dictionary with correct or ignored class names.
    - Write YOLO annotations based on XML input.
    - Convert XML annotations to YOLO format.

    Attributes:
        output_dir (str): The directory where the output will be saved.
        class_dict (dict): A dictionary mapping class names to class IDs.
        tag (str): A tag to prefix the output file names.
    """
    def __init__(self, output_dir, class_dict, tag):
        """
        Create the necessary output folders for images and annotations.

        The method ensures the existence of the 'original_images' and 'annotations'
        subdirectories within the specified output directory.
        """
        self.output_dir = output_dir
        self.class_dict = class_dict
        self.tag = tag
        self._create_output_folders()

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

    def _handle_misspelled_class(self, misspelled_word,default_match):
        """
        Find the closest match for a misspelled class name using fuzzy matching.

        Args:
            misspelled_word (str): The misspelled class name.

        Returns:
            tuple: The best matching class name and its similarity score.
        """
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
        """
        Generate new filenames for the image and annotation files.

        Args:
            xml_filename (str): The original XML filename.
            file_no (int): The file number to be included in the new filename.

        Returns:
            tuple: The new image filename and the new annotation filename.
        """
        ratios = [(class_name, fuzz.ratio(misspelled_word, class_name)) 
                for class_name in self.class_dict.keys()]
        best_match, best_score = max(ratios, key=lambda item: item[1])
        return best_match, best_score

    def _generate_filenames(self, xml_filename, file_no):
        """
        Get the image path corresponding to an XML file.

        Args:
            xml_file (str): The path to the XML file.

        Returns:
            str: The path to the corresponding image file.
        """
        file_no = str(file_no)
        xml_base_name, xml_extension = os.path.splitext(xml_filename)  
        new_img_filename = f"image_{self.tag}_{file_no.zfill(4)}{xml_extension}"  
        new_txt_filename = f"image_{self.tag}_{file_no.zfill(4)}.txt"
        return new_img_filename, new_txt_filename

    def _get_image_path(self, xml_file):
        """
        Copy the image file to the output directory with a new filename.

        Args:
            img_path (str): The original path to the image file.
            new_img_filename (str): The new filename for the image file.
        """
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')  

    def _copy_image(self, img_path, new_img_filename):
        """
        Update the class dictionary with the correct class name or mark it as ignored.

        Args:
            misspelled_word (str): The misspelled class name.
            correct_name (str or None): The correct class name, or None if ignored.
        """
        img_output_path = os.path.join(self.output_dir, "original_images", new_img_filename)
        shutil.copyfile(img_path, img_output_path)

    def _update_class_dictionary(self, misspelled_word, correct_name):
        """
        Handle unknown class names by finding the closest match and updating the dictionary.

        Args:
            class_name (str): The unknown class name.

        Returns:
            str: The corrected class name.
        """
        if correct_name == None:
            self.class_dict[misspelled_word] = None
        else:
            self.class_dict[misspelled_word] = self.class_dict[correct_name]

    def _handle_unknown_class(self, class_name):
        """
        Get the class ID for a given class name, handling unknown classes if necessary.

        Args:
            class_name (str): The class name.

        Returns:
            int or None: The class ID, or None if the class is ignored.
        """
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
        """
        Write the YOLO annotation for a given object.

        Args:
            txt_output_path (str): The path to the output text file.
            class_id (int): The class ID of the object.
            obj (xml.etree.ElementTree.Element): The XML element of the object.
            img_width (int): The width of the image.
            img_height (int): The height of the image.
        """
               
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
        """
        Convert an XML annotation file to YOLO format and save the output.

        Args:
            xml_file (str): The path to the XML file.
            file_no (int): The file number to be included in the new filename.
        """    
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



def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO format.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("--tag", help="Tag to prefix the output file names.", default="")
    parser.add_argument("--class_dict", type=str, help="Path to a JSON file containing the class dictionary.")

    args = parser.parse_args()

    if args.class_dict:
        import json
        with open(args.class_dict, "r") as f:
            class_dict = json.load(f)
    else:
        # Example class dictionary
        class_dict = {'arena': 0, 'fly': 1}  
        
    converter = AnnotationConverter(args.output_folder, class_dict, args.tag)

    file_count = 0
    for root, _, filenames in os.walk(args.xml_folder):
        for xml_filename in filenames:
            if xml_filename.endswith('.xml'):
                try:
                    xml_path = os.path.join(root, xml_filename)   
                    converter.convert_XML_to_YOLO(xml_path, file_count)
                    file_count += 1
                except:
                    print(f'error transforming: {xml_filename}')

if __name__ == "__main__":
    main()