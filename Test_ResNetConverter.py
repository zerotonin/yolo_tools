import os
import shutil
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

class AnnotationConverterForResNet:
    """
    A class to convert XML annotations to a format compatible with ResNet, 
    organizing images into class-specific directories.
    """

    def __init__(self, output_dir, tag):
        """
        Initializes the AnnotationConverter with the output directory and tag.

        Args:
            output_dir (str): The directory where the output will be saved.
            tag (str): A tag to prefix the output file names.
        """
        self.output_dir = output_dir
        self.tag = tag
        self._create_output_folder()

    def _create_output_folder(self):
        """
        Create the main output folder if it does not exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_filename(self, xml_filename, file_no):
        """
        Generate a new filename for the image file.

        Args:
            xml_filename (str): The original XML filename.
            file_no (int): The file number to be included in the new filename.

        Returns:
            str: The new image filename.
        """
        file_no = str(file_no)
        xml_base_name, xml_extension = os.path.splitext(xml_filename)
        new_img_filename = f"image_{self.tag}_{file_no.zfill(4)}{xml_extension}"
        return new_img_filename

    def _get_image_path(self, xml_file):
        """
        Get the image path corresponding to an XML file.

        Args:
            xml_file (str): The path to the XML file.

        Returns:
            str: The path to the corresponding image file.
        """
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')

    def _copy_image(self, img_path, class_name, new_img_filename):
        """
        Copy the image file to the class-specific directory.

        Args:
            img_path (str): The original path to the image file.
            class_name (str): The class name for the image.
            new_img_filename (str): The new filename for the image file.
        """
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_output_path = os.path.join(class_dir, new_img_filename)
        shutil.copyfile(img_path, img_output_path)

    def convert_XML_to_ResNet(self, xml_file, file_no):
        """
        Convert an XML annotation file to a format compatible with ResNet.

        Args:
            xml_file (str): The path to the XML file.
            file_no (int): The file number to be included in the new filename.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_filename = root.find('filename').text
        new_img_filename = self._generate_filename(img_filename, file_no)
        img_path = self._get_image_path(xml_file)

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            self._copy_image(img_path, class_name, new_img_filename)


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to ResNet format.")
    parser.add_argument("xml_folder", help="Path to the folder containing XML files.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("--tag", help="Tag to prefix the output file names.", default="")

    args = parser.parse_args()

    converter = AnnotationConverterForResNet(args.output_folder, args.tag)

    file_count = 0
    for root, _, filenames in os.walk(args.xml_folder):
        for xml_filename in filenames:
            if xml_filename.endswith('.xml'):
                try:
                    xml_path = os.path.join(root, xml_filename)
                    converter.convert_XML_to_ResNet(xml_path, file_count)
                    file_count += 1
                except:
                    print(f'Error transforming: {xml_filename}')

if __name__ == "__main__":
    main()
