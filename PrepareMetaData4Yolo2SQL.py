import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from movie_preprocessing.FrameSplitter import FrameSplitter
from database.FlyChoiceDatabase import DatabaseHandler
from AnalysisFileManager.PresetManager import PresetManager
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def create_subfolders(output_folder):
    """
    Creates multiple subdirectories within the specified output folder.
    Does not raise an error if the directories already exist.

    Args:
        output_folder (str): The path to the output directory where subdirectories will be created.
    """
    # Define the list of subdirectories to create
    subfolders = [
        "preprocessed_single_videos",
        "slurm_scripts",
        "presets",
        "meta_data",
        "results"
    ]
    
    # Convert the output folder path to a Path object
    base_path = Path(output_folder)
    
    # Iterate over the subfolders list and create each as a subdirectory of the base path
    for folder in subfolders:
        # Create the full path for the subfolder
        subfolder_path = base_path / folder
        
        # Use mkdir with exist_ok=True to avoid throwing an error if the folder already exists
        subfolder_path.mkdir(parents=True, exist_ok=True)

def get_file_path(file_types,mode_str):
    # Create a new Tkinter root instance
    root = tk.Tk()
    root.withdraw()  # Hide the main window to only show the dialog

    # Open the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title=f"Select a {mode_str} file",
        filetypes=file_types,
        initialdir="/"  # Start directory for the dialog
    )

    # Close the Tkinter instance
    root.destroy()

    return file_path

def get_folder_path(mode_str):
    # Create a new Tkinter root instance
    root = tk.Tk()
    root.withdraw()  # Hide the main window to only show the dialog

    # Open the directory selection dialog
    folder_path = filedialog.askdirectory(
        title=f"Select a {mode_str} folder",
        mustexist=False,  # Set to False to allow user to create new folders
        initialdir="/"  # Start directory for the dialog
    )

    # Close the Tkinter instance
    root.destroy()

    return folder_path

def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')

def check_and_get_paths(file_path,mode):

    if file_path:
        if os.path.isdir(file_path) or os.path.isfile(file_path):
            return file_path
        
    match mode:
        case 'video':
            file_path = get_file_path([("MP4 files", "*.mp4"),("AVI files", "*.avi"),("SEQ files", "*.seq"),("All files", "*.*")],mode)
        case 'database':
            file_path = get_file_path([("SQLite 3", "*.db")],mode)
        case 'output_folder':
            file_path = get_folder_path(mode)
        case _:
            raise ValueError(f"check_and_get_paths: Modus {mode} unknown.")
            
    if file_path:
        print(f"Selected file: {file_path}")
        return file_path

    else:
        raise ValueError(f"No {mode} file was selected.")

# file and folder paths
db_file_position = '/home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
video_file_position = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
output_file_path = '/home/geuba03p/food_example_video/output'

check_and_get_paths(db_file_position,'database')
check_and_get_paths(video_file_position,'video')
check_and_get_paths(output_file_path,'output_folder') 

# Make subfolders in output path 
create_subfolders(output_file_path)

# Make file_dict
file_dict = { 
'db_file_position' : db_file_position,
'video_file_position' : video_file_position,
'output_file_path' : output_file_path,
'preprocessed_single_videos' : os.path.join(output_file_path,'preprocessed_single_videos'),
'slurm_scripts' : os.path.join(output_file_path,'slurm_scripts'),
'presets' : os.path.join(output_file_path,'presets'),
'meta_data' : os.path.join(output_file_path,'meta_data'),
'results' : os.path.join(output_file_path,'results'),
}



# Instantiate database handler
db_handler = DatabaseHandler(f'sqlite:///{db_file_position}')
        

# Initialize managers
experiment_manager = ExperimentManager(db_handler)
arena_manager = ArenaManager(db_handler)
fly_manager = FlyManager(db_handler)
fly_distribution_manager = FlyDistributionManager(db_handler,{})
cli_file_manager = CLIFileManager()

clear_screen()
experiment_preset = cli_file_manager.load_dialogue('whole experiment',('*.csv', '*.*'))

# Start the process
experimenter_id, experiment_type_id = experiment_manager.manage_experiments()

# Get arena layout and other configurations
arena_layout = arena_manager.enter_arenas_for_experiment()
flies = fly_manager.enter_flies_for_experiment()
fly_distribution_manager.enter_flies_for_experiment(total_arenas)
arenas = fly_distribution_manager.distribute_flies(9, 6)  # Example layout with rows and cols
fly_distribution_manager.show_arena_assignments()

# Ask for the movie file location
video_file_position = filedialog.askopenfilename(
    title="Select the video file",
    filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"))
)

# Ask for the output folder location
output_folder = filedialog.askdirectory(title="Select the output folder")
split_vids_folder = os.path.join(output_folder, "split_vids")
presets_folder = os.path.join(output_folder, "presets")

# Create necessary folders
os.makedirs(split_vids_folder, exist_ok=True)
os.makedirs(presets_folder, exist_ok=True)

# Save presets
# Assuming you have functions or methods to save configuration presets
save_preset(fly_configurations, os.path.join(presets_folder, 'fly_presets.json'))
save_preset(arena_layout, os.path.join(presets_folder, 'arena_presets.json'))

# Process video
splitter = FrameSplitter(video_file_position, split_vids_folder, arena_layout)
splitter.split_video()

print("Experiment setup complete. Processed videos are saved in:", split_vids_folder)

# Optionally, start the main event loop
root.mainloop()
