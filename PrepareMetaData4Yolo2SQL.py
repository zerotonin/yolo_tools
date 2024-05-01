import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from movie_preprocessing.FrameSplitter import FrameSplitter
from database.FlyChoiceDatabase import DatabaseHandler
from CLI_tools.CLIFileManager import CLIFileManager
import tkinter as tk
from tkinter import filedialog

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
db_url = '/home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
video_file_path = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
output_file_path = '/home/geuba03p/food_example_video/output'

check_and_get_paths(db_url,'database')
check_and_get_paths(video_file_path,'video')
check_and_get_paths(output_file_path,'output_folder')

# Instantiate database handler
db_handler = DatabaseHandler(f'sqlite:///{db_url}')
        

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
video_file_path = filedialog.askopenfilename(
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
splitter = FrameSplitter(video_file_path, split_vids_folder, arena_layout)
splitter.split_video()

print("Experiment setup complete. Processed videos are saved in:", split_vids_folder)

# Optionally, start the main event loop
root.mainloop()
