import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from movie_preprocessing.FrameSplitter import FrameSplitter
from database.FlyChoiceDatabase import DatabaseHandler
from CLI_tools.CLIFileManager import CLIFileManager


def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')


# Instantiate database handler
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)

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
