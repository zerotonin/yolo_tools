import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

class AnalysisFileManager:
    def __init__(self):
        self.file_dict = {}
        self.folder_dict = {}

    def create_subfolders(self):
        """
        Creates predefined subdirectories within the base output folder.
        Does not raise an error if the directories already exist.
        """
        subfolders = [
            "preprocessed_single_videos",
            "slurm_scripts",
            "presets",
            "meta_data",
            "results"
        ]

        for folder in subfolders:
            subfolder_path = self.base_path / folder
            subfolder_path.mkdir(parents=True, exist_ok=True)
            self.folder_dict[folder] = subfolder_path
        
        print("Subfolders created or verified.")

    def use_tkinter(self):
        return os.environ.get('DISPLAY', None) is not None

    def get_file_path(self, file_types, title):
        """
        Opens a file dialog to select a file with specified types.
        """
        if self.use_tkinter():
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title=title, filetypes=file_types, initialdir="/")
            root.destroy()
        else:
            print(f"{title}:")
            for i, ft in enumerate(file_types, 1):
                print(f"{i}. {ft[1]} files")
            file_path = input("Please enter the path to your file: ")
        return file_path
    
    def get_folder_path(self, title):
        """
        Opens a directory dialog to select or create a directory.
        """
        if self.use_tkinter():
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory(title=title, mustexist=False, initialdir="/")
            root.destroy()
        else:
            print(f"{title}: Please enter the path to your directory")
            folder_path = input("Directory path: ")
        return folder_path

    def check_and_get_paths(self, file_path, mode):
        """
        Validates and potentially updates paths for files and folders.
        """
        if file_path:
            path = Path(file_path)
            if path.exists():
                return path
        
        title = f"Select a {mode}"
        if mode in ['database','python_interpreter', 'yolo_weights']:
            extensions = {
                'video': [("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("SEQ files", "*.seq"), ("All files", "*.*")],
                'database': [("SQLite 3", "*.db")],
                'python_interpreter': [("Python", '*.*')],
                'yolo_weights': [("Yolo Weights", '*.*')]
            }
            file_path = self.get_file_path(extensions[mode], title)
        elif mode == 'output_folder':
            file_path = self.get_folder_path(title)

        if file_path:
            self.file_dict[mode] = file_path
            print(f"Selected {mode} file: {file_path}")
            return file_path
        else:
            raise ValueError(f"No {mode} file was selected.")

    def setup_experiment_paths(self, base_output_path, db_file, video_file,python_interpreter,yolo_weights):
        """
        Sets up and verifies all paths necessary for an experiment, including the database file,
        video file, and output directory. It initializes the experiment's directory structure
        by creating subfolders for various data types.

        Args:
            base_output_path (str): The base path where the experiment's data will be stored.
            db_file (str): The path to the experiment's database file.
            video_file (str): The path to the experiment's video file.
            python_interpreter (str): The path to the python interpreter you want to use.

        Raises:
            ValueError: If any required file or directory is not selected or does not exist.

        The function updates internal dictionaries to manage paths efficiently:
        - `file_dict` to keep track of file locations like the database and video files.
        - `path_dict` to store paths to important directories and subdirectories, ensuring
        that all components of the experiment can reference these locations easily.

        This method is typically called at the start of an experiment setup process to ensure
        all necessary files and folders are properly configured and exist.
        """
        db_file = self.check_and_get_paths(db_file, 'database')
        video_file = self.check_and_get_paths(video_file, 'video')
        output_folder = self.check_and_get_paths(base_output_path, 'output_folder')
        python_interpreter = self.check_and_get_paths(python_interpreter, 'python_interpreter')
        yolo_weights = self.check_and_get_paths(yolo_weights, 'yolo_weights')
        self.base_path = Path(output_folder)
        self.create_subfolders()
        
        self.file_dict = {
            'db_file_position': db_file,
            'video_file_position': video_file,
            'python_interpreter': python_interpreter,
            'yolo_weights': yolo_weights
        }
        
        self.path_dict = {
            'output_file_path': output_folder,
            'preprocessed_single_videos': os.path.join(output_folder, 'preprocessed_single_videos'),
            'slurm_scripts': os.path.join(output_folder, 'slurm_scripts'),
            'presets': os.path.join(output_folder, 'presets'),
            'meta_data': os.path.join(output_folder, 'meta_data'),
            'results': os.path.join(output_folder, 'results')
        }
