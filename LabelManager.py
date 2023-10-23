from fuzzywuzzy import fuzz, process
import xml.etree.ElementTree as ET

class LabelManager:
    """Manages label corrections in LabelImg XML annotations.

    This class is an evolution of the old Charon Label Manager, with enhancements
    to correct label inconsistencies or errors in LabelImg XML annotations.
    It inherits class labels from an instance of the YoloMancer class and 
    uses those as a reference for validation.

    One of the key improvements is the integration of a Levenshtein auto-assigner,
    which uses fuzzy string matching to automatically suggest the most likely
    correct label based on existing class labels. This feature is particularly
    useful for correcting human errors in label names, like typos or variations
    in spelling or casing.

    Attributes:
        yolo_mancer (YoloMancer): An instance of the YoloMancer class.
        labelsAll (set): All unique labels found in the XML files.
        labelChanger (dict): Dictionary to map old labels to new labels or actions.
        
    Author: Dr. Bart Geurten
    Date: 23rd October 2023
    """

    def __init__(self, yolo_mancer):
        """Initializes the LabelManager class.

        Args:
            yolo_mancer (YoloMancer): An instance of the YoloMancer class.
        """
        self.yolo_mancer = yolo_mancer
        self.labelsAll = set()  # All unique labels found in the XML files
        self.labelChanger = {}  # Dictionary to map old labels to new labels or actions

    def getLabelsFromXML(self, file):
        """Retrieves all unique labels from a given LabelImg XML file."""
        nameList = set()
        root = ET.parse(file).getroot()
        for nameText in root.iter('name'):
            nameList.add(nameText.text)
        return nameList

    def renameLabelsVerbose(self):
        """Interactive CLI tool to manually correct or confirm labels."""
        
        # Only proceed if there are labels not in yolo_mancer.class_dict
        unknown_labels = self.labelsAll - set(self.yolo_mancer.class_dict.keys())
        if not unknown_labels:
            print("All labels are known. No action required.")
            return

        print('These unknown labels were entered:')
        print('==========================')
        for i, label in enumerate(unknown_labels, 1):
            print(f" {i}. {label}")
        print('')

        for label in unknown_labels:
            print('============================================')
            # Automatic guess using fuzzywuzzy
            closest_match, score = process.extractOne(label, self.yolo_mancer.class_dict.keys())
            if score > 80:  # You can adjust the score threshold as needed
                print(f"Did you mean {closest_match}?")
                correct = input('Is this correct? [y/n] ').strip().lower()
                if correct == 'y':
                    self.labelChanger[label] = closest_match
                    continue

            # If the fuzzy match was incorrect or not close enough, proceed to manual correction
            self.enterNewLabel(label)

        print('============================================')
        print('Here are the rules for relabelling:')
        print(self.labelChanger)

    def enterNewLabel(self, label):
        """Sub-function to rename or delete a single label.

        Args:
            label (str): The label to be renamed or deleted.
        """
        print(f"Old label: {label}")
        correct = input('Do you want to change the label [y/n] or delete it [d]? ').strip().lower()
        while correct not in ('y', 'n', 'd'):
            correct = input('Do you want to change the label [y/n] or delete it [d]? ').strip().lower()

        if correct == 'n':
            self.labelChanger[label] = label
        elif correct == 'd':
            self.handleLabelDeletion(label)
        else:
            self.handleLabelRenaming(label)

    def handleLabelDeletion(self, label):
        """Handles label deletion after user confirmation.

        Args:
            label (str): The label to be deleted.
        """
        correct = input(f'Are you sure to delete this label: {label}? [y/n] ').strip().lower()
        while correct not in ('y', 'n'):
            correct = input(f'Are you sure to delete this label: {label}? [y/n] ').strip().lower()
        
        if correct == 'y':
            self.labelChanger[label] = '!deleteThisLabel!'
        else:
            self.enterNewLabel(label)

    def handleLabelRenaming(self, label):
        """Handles label renaming after user confirmation.

        Args:
            label (str): The label to be renamed.
        """
        newLabel = input('Enter new label: ').strip()
        correct = input(f'{label} -> {newLabel}. Is this correct? [y/n] ').strip().lower()
        while correct not in ('y', 'n'):
            correct = input(f'{label} -> {newLabel}. Is this correct? [y/n] ').strip().lower()

        if correct == 'y':
            self.labelChanger[label] = newLabel
        else:
            self.enterNewLabel(label)

# Usage example
yolo_mancer = YoloMancer()
yolo_mancer.add_class("Penguin", 0)
yolo_mancer.add_class("Chick", 1)
yolo_mancer.add_class("Creche", 2)

label_manager = LabelManager(yolo_mancer)
label_manager.labelsAll = label_manager.getLabelsFromXML('/path/to/xml/file')
label_manager.renameLabelsVerbose()
