import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz  
import os
from tabulate import tabulate  # If you're using tabulate
import shutil
import argparse
import json

# import ClassCategoryReader


class AnnotationConverter:
    # definition for init method or constructor
    counter = 0 # Class variable???

    # Class variables: 
    # shared among all instances of a class.
    # They are defined within the class but outside of any methods, typically near the top of the class definition.
    # Class variables store data that is common to all instances, making them a powerful tool for managing shared state and settings
    
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
        # Instance variables:
        # unique to each instance of a class.
        # They are defined within methods and are prefixed with the self keyword.
        # These variables store data that is specific to an instance.

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

        # 1. self._ai_converters: instnce variable that stores a directory.
        #   The dictionary maps the names of AI models (as strings) to corresponding methods of the class.

        # 2. Dictionary Keys: These are strings representing the names of different AI models. (YOLO,...)

        # 3. Dictionary Values (e.g., self.convert_XML_to_YOLO):
        #   These are method references. For example, self.convert_XML_to_YOLO refers to a method of the class that converts XML data
        #   into a format suitable for the YOLO model.

        # 2. Line:
        # 1. ai_converter: This is a variable that holds the name of the desired AI model as a string.
        # 2. self._ai_converters[ai_converter]: 
        #   This line looks up the dictionary self._ai_converters using the key ai_converter.
        #   If ai_converter is "YOLO", for instance, this would access self.convert_XML_to_YOLO
        # 3. self._chosen_ai_converter:
        # The result of the dictionary lookup is then stored in another instance variable, self._chosen_ai_converter.


    def convert_XML_file(self, xml_file, file_no):
        self._chosen_ai_converter(xml_file, file_no)
        # print("do I get here?")

        # Class method:
        # def: This keyword defines a new function or method.
        # convert_XML_file: The name of the method. It suggests that the method's purpose is to convert an XML file.
        # self: This is a reference to the instance of the class. It allows the method to access instance variables 
        # and other methods of the class.
        # xml_file: A parameter that likely represents the path to the XML file or the XML content itself that needs to be converted.
        # file_no: A second parameter, probably representing a file number, index, or identifier associated with the XML file. 
        # This might be used to distinguish between different files or to handle them uniquely during the conversion process.

        # The line self._chosen_ai_converter(xml_file, file_no) calls the method stored in self._chosen_ai_converter, 
        # passing xml_file and file_no as arguments.

    # Class method:
    # Same inside the class method, we use the cls (self?) keyword as a first parameter to access class variables. 
    # The class method can only access the class attributes, not the instance attributes
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

        # os.makedirs(path, mode = Oo777, exist_ok = False): 
        # Erstellt ein Directory recursevely. Wenn ein directry dazwischen fehlt wird es automatisch erstellt. 
        # path: path-like object (string). File system path
        # mode: optional. Integer value.
        # exist_ok: default False. Target directory exist -> Error. True: leaves directory unaltered.

        # os.path.join(path, *paths):
        # joins one or more path components intelligently. 
        # Concatenates various path components with exactly one directory seperator ('/') following each non-empty part.
        # path: path-like object (string). File system path.
        # *paths: A path-like object representing a file system path. It represents the path components to be joined.
        # *args (here *paths): in function definitions in python is used to pass a variable number of arguments to a funtion.

        # Was passiert:
        # Erstellt ein neues Directory aus dem self.output_dir und dem path component "oiginal images"/"annotations".

    # Class method:
    def _create_categories(self):
        for class_name, class_id in self.class_dict.items():
            self.annotations["categories"].append({"id": class_id, "name": class_name})

        # populate a categories list within an annotations dictionary. 
        # Each item in this list is a dictionary containing an id and a name, corresponding to different classes/categories.

        # 1. Method Definition: The function is defined as _create_categories(self). 
        #    The underscore prefix in the method name indicates that this method is intended for internal use within the class.

        # 2. The function iterates through the items of self.class_dict.
        #    self.class_dict is expected to be a dictionary where:
        #       The keys (class_name) are the names of the categories.
        #       The values (class_id) are the identifiers for those categories

        # 3. Accessing annotations: 
        #    The function assumes that self.annotations is a dictionary with a key "categories" that holds a list.

        # 4. Appending to Categories: For each class_name and class_id in the class_dict:
        #    The function creates a dictionary with two keys:
        #       "id": This is set to class_id.
        #       "name": This is set to class_name.
        #    This dictionary is appended to the self.annotations["categories"] list


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
    
    # os.path module: 
    # is submodule of OS module in Python used for common pathname manipulation.
    # os.path.splitext() method: is used to split the path name into a pair root and ext. ext stands for extension and has the extension portion of the specified path while root is everything except ext part.

    # formatted string (f-string):
    # used to create a string with dynamic content, combining literal text with variables and expressions.
    # 1. f-string: begins with 'f'. 
    #   This allows you to include expressions inside curly braces {} within the string, which will be evaluated and replaced with 
    #   their  values.
    # 2. Literal text: '"image_", "_", and "{xml_extension}"'. will be included in the final string as they are written.
    # 3. Variable self.tag:
    #   The value of self.tag is inserted at this position in the string.
    # 4. Method file_no.zfill(4):
    #   method call on the file_no variable, which is expected to be a string or a number.
    #   The .zfill(4) method pads the string file_no with zeros on the left to ensure that its total length is 4 characters.
    #   e.g.: 23 -> 0023.
    # 5. Variable {xml_extension}:
    #   value of the variable xml_extension is included at this position in the string.

    # Example: 
    # self.tag is "sample"
    # file_no is "45"
    # xml_extension is ".xml"
    # Output: "image_sample_0045.xml"

    # Class method returns the new_img_filename and new_text_filename


    def _get_image_path(self, xml_file):
        """
        Copy the image file to the output directory with a new filename.

        Args:
            img_path (str): The original path to the image file.
            new_img_filename (str): The new filename for the image file.
        """
        img_base_name, _ = os.path.splitext(os.path.basename(xml_file))
        return os.path.join(os.path.dirname(xml_file), img_base_name + '.png')  
    
    # os.path.basename():
    # internally use os.path.split() method to split the specified path into a pair (head, tail).
    # returns the tail part after splitting the specified path into (head, tail) pair.
    # Example: '/home/User/Documents/file.txt'-> returns: 'file.txt'

    # img_base_name (str):
    # os.path.splitext splits the pathname into root and ext. os.path.basename returns the tail of the pathname.
    # Example: path= Dokuments/filename.xml -> Output: filename

    # os.path.dirname():
    # used to get the directory name from the specified path.
    # Example: '/home/User/Documents/file.txt' -> returns: /home/User/Documents/

    # Class method returns:
    # A string containing the directory name and the image basename of the xml file and attends the extension .png.
    # Output example:
    # /home/User/Documents/file.png
    

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

    
    #1. xml.etree.ElementTree module:
    # xml file processin routine.
    # 1. ET.parse(xml_file): It reads and parses an XML file.
    #       This function reads the content of the XML file and constructs a tree structure (an ElementTree object) 
    #       in memory that represents the hierarchical structure of the XML document.
    #       Result:
    #           is stored in the variable tree
    #           tree is an instance of ElementTree, which allows you to navigate and manipulate the XML data.

    # 2. tree.getroot(): getroot method is called on the tree object
    #   method retrieves the root element of the XML tree.
    #   It accesses the top-level element in the XML document
    #   root element is the parent of all other elements in the XML hierarchy
    #   root element serves as the starting point
    #   Result:
    #       root element serves as the starting point
    #       root is an instance of Element, representing the root of the XML tree.
    #       Through root, you can access all the child elements, attributes, and text data contained within the XML file.

    # Example: see notes.

    # 3. Dictionary image_dimensions: 
    #   The dictionary is initialized with the following example key-value pairs:
    #   "img_height": img_height
    #   Key: "img_height" — a string representing the height of the image.
    #   Value: img_height — a variable that holds the actual height value of the image.

    # Data Storage: By storing this information in a dictionary, you can easily pass this data around, 
    # access individual elements by their keys, and maintain a clear structure for the image's dimensions and bounding box coordinates.

    # 4. new_img_filename, new_txt_filename = self._generate_filenames(img_filename, file_no):
    #   calls the method _generate_filenames and passes img-filename and file_no. Saves the result in new_img:filename and
    #   new_txt_filename.
    # 5. img_path = self._get_image_path(xml_file):
    #   calls the method _get_image_path and passes xml_file. Saves the result in the variable img_path.
    # 6. txt_output_path = os.path.join(self.output_dir, "annotations", new_txt_filename)
    #   os.path.join: joins the path of self.output:dir, "annotations" and new_txt_filename to one path.
    #   saves path in variable txt_output_path

    # 7. Dictionary file_data

    # 8. dictionary object_classes = {} ...:
    #   - loop iterates over each object (obj) element in root (and finds all objects).
    #       root.findall('object') is a method that finds all elements in the XML with the tag 'object'.
    #   - finds the 'name' for each object and saves it as .text in the variable class_name.
    #   - calls the method _get_class_id and passes class_name. Returns class_id and saves it in the variable class_id.
    #   - checking if bounding box coordinates should be returned:
    #       return_object_bndbox is likely a boolean parameter indicating 
    #       whether the bounding box coordinates should be included in the object_classes dictionary.
    #       Extracting Bounding Box Coordinates:
    #           obj.find('bndbox') locates the <bndbox> element within the <object> element, which contains the bounding box coordinates.
    #           extract text content of xmin,..., converts them to integers and stores them in the corresponding variable.
    #       Storing Class and Bounding Box Information:
    #           If return_object_bndbox is True, this line stores a list as the value in object_classes for the given class_id.
    #           The list contains the class_name and a dictionary of bounding box coordinates 
    #           with keys "xmin", "xmax", "ymin", and "ymax".
    #       Storing Only the Class Name:
    #           If return_object_bndbox is False, only the class_name is stored in object_classes for the given class_id.

    # Method returns three values:
    #   object_classes: The dictionary containing all the objects' class names and possibly their bounding box coordinates.
    #   image_dimensions: Likely a dictionary containing metadata about the image (like height, width, etc.).
    #   file_data: Another variable that likely contains additional metadata or information about the file.

    
    def _copy_image(self, img_path, new_img_filename):
        """
        Update the class dictionary with the correct class name or mark it as ignored.

        Args:
            misspelled_word (str): The misspelled class name.
            correct_name (str or None): The correct class name, or None if ignored.
        """
        img_output_path = os.path.join(self.output_dir, "original_images", new_img_filename)
        shutil.copyfile(img_path, img_output_path)

        # 1. os.path.join:
        #   joins the paths self.output_dir, "oringinal_images" and new_img_filename to one path.
        #   Saves the path in the variable img_output_path

        # 2. shutil.copyfile():
        #   shutil.copyfile(source, destination, *, follow_symlinks = True) (last two are optional)
        #   - used to copy the content of the source file to the destination file
        #   - The metadata of the file is not copied.
        #   - The source and destination must represent a file and the destination must be writable
        #   - If the destination already exists then it will be replaced with the source file otherwise a new file will be created.
        #   - source: A string representing the path of the source file. 
        #     destination: A string representing the path of the destination file.
        # returns a string that represents the path of the newly created file.

    def _get_class_id(self,class_name):
        if class_name not in self.class_dict:
            correct_name = self._handle_unknown_class(class_name)
        else:
            correct_name = class_name 

        class_id = self.class_dict.get(correct_name) 

        return class_id              
    
    # 1. Method: 
    # self refers to the instance of the class, and class_name is the parameter that takes the name of the class 
    # whose ID you want to retrieve.

    # 2. Checking class name:
    # checks whether the provided class_name exists as a key in the dictionary self.class_dict
    # self.class_dict is assumed to be a dictionary where class names are keys and their corresponding IDs are values.

    # 3. Handling Unknown Class Name:
    # If class_name is not found in self.class_dict, this line calls another method _handle_unknown_class with class_name as an argument.
    # The result of _handle_unknown_class is stored in correct_name.

    # 4. Handling Known Class Name:
    # If class_name is found in self.class_dict, class_name is assigned to correct_name.

    # 5. Retrieving Class ID:
    # attempts to retrieve the class ID associated with correct_name from self.class_dict using the get method.
    # get() method:
    # The get method of a dictionary returns the value for the specified key if the key is in the dictionary; 
    # otherwise, it returns None (or a default value if provided).

    # 6. Returning the Class ID:
    # returns the retrieved class_id. 
    # If correct_name was not found in the dictionary, this will return None.



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


        # 1. Calculating the Relative Width and Height of the Bounding Box:
        #   - box_w: Calculates the width of the bounding box in pixel units, normalized by dividing the 
        #     pixel width by the total width of the image (obj_dims["img_width"]).
        #   - box_h: calculates the height of the bounding box in pixel units, normalized by dividing the 
        #     pixel height by the total height of the image (obj_dims["img_height"]).

        # These calculations convert the bounding box dimensions from absolute pixel values to relative values (ranging between 0 and 1),
        # which are independent of the image size. 
        # This is a common practice in object detection tasks 
        # to make the model training more generalizable to images of different sizes.

        # 2. Calculating the Relative Center of the Bounding Box
        #   - box_cx: calculates the x-coordinate of the center of the bounding box in pixel units,  
        #     normalized by dividing by the total width of the image.
        #   - box_cy: calculates the y-coordinate of the center of the bounding box in pixel units, 
        #     normalized by dividing by the total height of the image.

        # These calculations determine the position of the center of the bounding box as relative coordinates, 
        # making the bounding box location independent of the image size.

        # opens (or creates) a text file at the path txt_output_path in append mode.
        # It then writes a line containing the class_id followed by four floating-point values 
        # (box_cx, box_cy, box_w, box_h), each formatted to 6 decimal places. The values are separated by spaces,
        # and a newline is added at the end so that the next write operation will be on a new line.
        # writes the bounding box coordinates.


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

        # 1. Reading the xml file:
        #   This line is calling a method named _read_XML_file to extract information from an XML file.
        #   Return values: object_classes, image_dimensions and file_data

        # 2. Copying the Image:
        #   This line is calling a method named _copy_image to copy the image from its original location to a new location.

        # 3. Writing YOLO Annotations
        #   This loop iterates over each object class ID found in the object_classes dictionary 
        #   and writes the corresponding YOLO annotation to a text file.
        #   The _write_YOLO_annotation method likely writes the class ID 
        #   and the corresponding bounding box information (in YOLO format) to a text file located at txt_output_path.


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

    # 1. Width and Hight of the bounding box:
    #   Calculates width and hight of the bounding box in pixel units and stores them in these thwo variables.

    # 2. Appending Annotation Data to the Annotations List:
    #    appends a new annotation (in dictionary form) to the list of annotations stored 
    #    under the "annotations" key in the self.annotations dictionary.
    #    Annotation fields:
    #       - "id": self.annotation_id: assigns a unique identifier (annotation_id) to the annotation
    #       - "image_id": self.image_id: links the annotation to a specific image by using an image 
    #               identifier (image_id).
    #       - "category_id": class_id: associates the annotation with a specific object class using the class_id.
    #               class_id likely corresponds to the ID of the category or class (e.g., "fly", "arena") 
    #               in the dataset.
    #       - "bbox": [obj_dims["xmin"], obj_dims["ymin"], width, height]:
    #               defines the bounding box for the annotated object using four values:
    #               obj_dims["xmin"], obj_dims["ymin"], width, height
    #               This is typically in the format [x_min, y_min, width, height].
    #       - "area": width * height: calculates and stores the area of the bounding box by multiplying its 
    #               width and height.
    #               This is useful for various purposes, such as filtering out very small bounding boxes
    #       - "iscrowd": 0: This flag indicates whether the annotation represents a single object or 
    #               a crowd of objects.

    # 3. Incrementing the Annotation ID:
    #   increments the annotation_id counter by 1.
    #   This ensures that the next annotation appended to self.annotations["annotations"] will have a unique ID.

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

    # 1. Reading the XML File - calls the method _read_XML_file to extract information from an XML file.
    # 2. Copying the Image - copies the image file from its original location to a new location or renames it.
    # 3. Appending Image Metadata to the Annotations:
    #       This block of code appends image metadata to the images list within the self.annotations dictionary.
    #       "id": self.image_id: Assigns a unique ID to the image.
    #       "file_name": file_data["new_img_filename"]: Records the filename of the image after copying/renaming.
    #       "width": image_dimensions["img_width"]: Records the width of the image.
    #       "height": image_dimensions["img_height"]: Records the height of the image.
    #       This step ensures that the image's metadata is stored in the annotations structure, 
    #       which is being formatted according to the COCO dataset format.
    
    # 4. Writing COCO Annotations for Each Object:
    #       This loop iterates through each object class in the object_classes dictionary and 
    #       writes a corresponding COCO annotation.
    #       The _write_COCO_annotation method likely adds the bounding box and 
    #       other relevant data for each object into the annotations dictionary under a COCO-compatible format.

    # 5. Incrementing the Image ID - increments the image_id counter to ensure that the next image processed has a unique ID.

    # 6. Saving Annotations to a JSON File:
    #       saves all the accumulated annotations to a JSON file.
    #       Saving Process:
    #       with open(output_file, 'w') as f:: Opens the specified file in write mode ('w').
    #       json.dump(self.annotations, f, indent=4): Converts the self.annotations dictionary 
    #       into a JSON-formatted string and writes it to the file. 
    #       The indent=4 parameter makes the JSON file more readable by formatting it with indentation.


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

        
    # 1. Reading xml file: calls a method to read the contents of an XML file that contains annotations 
    #    (such as object classes and bounding boxes).
    #    return_object_bndbox=True: bounding box information should be included in the returned data.

    # 2.  copies the image from its original path (img_path) to a new location (new_img_filename). 

    # 3. Adding Image Info:
    #       A dictionary named image_info is created, which contains metadata about the image, including:
    #           id: A unique identifier for this image, tracked by self.image_id
    #           file_name: The new filename of the image
    #           width and height: The dimensions of the image, extracted from image_dimensions.
    #       This image information is added to a list (self.images), which stores details about all images.
    #       self.image_id += 1: The image_id counter is incremented for the next image.

    # 4. Loop Over Object Classes:
    #   This loop iterates over all object annotations in the image. Each object has a class name and 
    #   a bounding box (bndbox), which contains the coordinates (xmin, xmax, ymin, ymax).
    #   if class_name in self.class_dict::
    #       This checks if the class name exists in self.class_dict, 
    #       which likely maps class names to numeric category IDs.
    
    # 5. Bounding Box Calculation:
    #   The bounding box dimensions are calculated:
    #       Extract the coordinates of the bounding box from bndbox.
    #   Calculate the width and height of the bounding box.

    # 6. Segmentation Calculation:
    #   A simple segmentation mask is created based on the bounding box coordinates. 
    #   For Mask R-CNN, this represents the outline of the object as a polygon, 
    #   here using the four corners of the bounding box.
    #   More sophisticated segmentations (beyond bounding boxes) can have multiple points.

    # 7. Annotation Info Block:
    #   Each annotation needs a unique ID, so the annotation_id is incremented.
    #   A dictionary named annotation_info is created to store the information of this annotation, including:
    #       id: The unique annotation ID.
    #       image_id: The ID of the image to which this annotation belongs.
    #       category_id: The class ID.
    #       bbox: The bounding box in the form [xmin, ymin, width, height].
    #       segmentation: The segmentation polygon calculated earlier.
    #       area: The area of the bounding box, calculated as width * height.
    #       iscrowd: Whether the object is "crowded" (0 means it's not crowded).
    #   This annotation is added to the list of annotations, which is a key part of the Mask R-CNN format.

    # save_to_json(self)
    #   function saves the annotations collected in the above process into a JSON file in COCO format.

    # 1. Path to JSON file:
    #   defines the path where the output JSON file will be saved, 
    #   under self.output_folder with the filename coco_annotations.json.

    # 2. COCO Format Structure:
    #   A dictionary named coco_format is created in the COCO annotation format, containing:
    #       images: A list of all images and their associated metadata (e.g., filename, dimensions, ID).
    #       annotations: A list of all annotations (i.e., objects detected in the images), 
    #                    which were added in the convert_XML_to_MaskRCNN function.
    # categories: A list of categories (classes) in the dataset, with each category having an id and name. 
    #             The category list is generated by iterating over the self.class_dict dictionary.
    #               "categories": [{"id": v, "name": k} for k, v in self.class_dict.items()]:
    #                   This creates a list of dictionaries, where each dictionary contains the class id (from v)
    #                   and the class name (from k).

    # 3. Writing to JSON:
    #   The output_file is opened in write mode ('w').
    #   json.dump(coco_format, f, indent=4):
    #       The json.dump() method is used to write the coco_format dictionary to the file f in JSON format.
    #       The argument indent=4 ensures the JSON is written in a human-readable way, 
    #       with an indentation level of 4 spaces.


########## Main ###################

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


# Command Line Structure:

# python script.py /path/to/xml/folder /path/to/output AI converter --tag mydata --class_dict /path/to/class_dict.json

# /path/to/xml/folder: This is the folder containing the XML annotation files.
# /path/to/output: This is the folder where the converted annotation files will be saved.
# AI converter: Specifies which converter the script should use ( YOLO, RetinaNet, ResNet, Mask_RCNN).
# --tag mydata: (Optional) This prefixes the output files with "mydata".
# --class_dict /path/to/class_dict.json: (Optional) Specifies the path to a JSON file that contains the class dictionary.


# 1. Argument Parsing:
#   The function uses argparse.ArgumentParser to define command-line arguments for this script.
#   The following arguments are expected when running the script from the command line:
#       xml_folder: Path to the folder containing XML files.
#       output_folder: Path where the output (converted annotations) will be saved.
#       ai_converter: Specifies the AI converter model (YOLO, RetinaNet, ResNet, Mask R-CNN).
#       --tag (optional): A tag that will be prefixed to the output file names.
#       --class_dict (optional): Path to a JSON file that contains the mapping of class names to class IDs.

# 2. Parsing the Arguments:
#       args = parser.parse_args() 
#           parses the arguments passed from the command line into the args object. 
#           This allows easy access to these arguments throughout the script 
#           (e.g., args.xml_folder, args.output_folder, args.ai_converter).

# 3. Checking for Valid Converter:
#   ensures that the ai_converter argument is one of the accepted values: YOLO, RetinaNet, ResNet, or Mask R-CNN.
#   If an invalid option is passed, it will print an error message.

# 4. Loading the Class Dictionary:
#   If the user provides a --class_dict argument (a JSON file containing the class dictionary), 
#   the script loads it using json.load(). This dictionary maps class names (like "dog", "cat") to numerical IDs.

#   If --class_dict is not provided, it defaults to a simple predefined class dictionary ({'arena': 0, 'fly': 1}).

# 5. Initializing the Converter:
#   converter object is created using the AnnotationConverter class. It takes the following arguments:
#       output_folder: The folder where converted annotations will be saved.
#       class_dict: The class dictionary, mapping object classes to numerical IDs.
#       tag: The optional tag to prefix output files.
#       ai_converter: The AI model type (YOLO, RetinaNet, ResNet, Mask R-CNN).

# 6. Processing XML Files:
#   script walks through the xml_folder directory (and its subdirectories) to find XML files using os.walk(). 
#   For each XML file it finds, it attempts to convert it using converter.convert_XML_file().

#   The variable file_count tracks the number of processed files, incrementing after each successful conversion.
#   If an error occurs while converting an XML file, the script prints an error message showing which file failed:
#   print(f'error transforming: {xml_filename}').

# 7. Saving the Results:
#   If the chosen converter is either "Mask_RCNN" or "RetinaNet", 
#   the script saves the annotations to a JSON file using converter.save_to_json()


