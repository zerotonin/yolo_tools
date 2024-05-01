import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from movie_preprocessing.VideoInfoExtractor import VideoInfoExtractor
from database.FlyChoiceDatabase import DatabaseHandler
from AnalysisFileManager.PresetManager import PresetManager
from AnalysisFileManager.AnalysisFileManager import AnalysisFileManager



def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')


file_manager = AnalysisFileManager()


# file and folder paths
db_file_position = '/home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
video_file_position = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
base_output_path = '/home/geuba03p/food_example_video/output'
python_interp = '/home/geuba03p/miniconda3/envs/yolov8/bin/python'
file_manager.setup_experiment_paths(base_output_path,db_file_position,video_file_position,python_interp)



# Instantiate database handler
db_handler = DatabaseHandler(f'sqlite:///{db_file_position}')

# Initialise meta data  managers
experiment_manager = ExperimentManager(db_handler)
arena_manager = ArenaManager(db_handler)
fly_manager = FlyManager(db_handler)
fly_distribution_manager = FlyDistributionManager(db_handler,{})
preset_file_manager = PresetManager()
        


# Get all info for posthoc run
clear_screen()
experiment_preset = preset_file_manager.load_dialogue('whole experiment',('*.csv', '*.*'))

# Get automated info from posthoc run

video_info_extractor = VideoInfoExtractor(file_manager.file_dict['video_file_position'])
video_info = video_info_extractor.get_video_info()



# Start the process
experimenter_id, experiment_type_id = experiment_manager.manage_experiments()

# Get arena layout and other configurations
arena_layout = arena_manager.enter_arenas_for_experiment()
flies = fly_manager.enter_flies_for_experiment()
fly_distribution_manager.enter_flies_for_experiment(total_arenas)
arenas = fly_distribution_manager.distribute_flies(9, 6)  # Example layout with rows and cols
fly_distribution_manager.show_arena_assignments()

