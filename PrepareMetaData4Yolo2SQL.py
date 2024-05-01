import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from database.StimulusManager import StimulusManager
from movie_preprocessing.VideoInfoExtractor import VideoInfoExtractor
from database.FlyChoiceDatabase import DatabaseHandler
from AnalysisFileManager.PresetManager import PresetManager
from AnalysisFileManager.AnalysisFileManager import AnalysisFileManager

def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_arena_configuration():
    """
    Prompts the user via CLI to input the number of arenas and their arrangement in rows and columns.
    
    Returns:
        tuple: A tuple containing three integers: 
               1. The total number of arenas.
               2. The number of rows.
               3. The number of columns.
    
    Raises:
        ValueError: If the input values are not integers or if the product of rows and columns
                    does not equal the total number of arenas.
    """
    print("Please enter the details for arena configuration.")

    try:
        total_arenas = int(input("Enter the total number of arenas in the experiment: "))
        rows = int(input("Enter the number of rows of arenas: "))
        columns = int(input("Enter the number of columns of arenas: "))

        # Validate that rows and columns correctly represent the total number of arenas
        if rows * columns != total_arenas:
            raise ValueError("The product of rows and columns must equal the total number of arenas.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        return get_arena_configuration()  # Recursively call the function again if there's an error

    return {'arena_num' : total_arenas, 'arena_rows' : rows, 'arena_cols' : columns}


def check_loadable_preset(file_manager,file_type):
    return file_type in list(file_manager.file_dict.keys())

def save_preset(file_manager,preset_file_manager,file_type,data):
    preset_path = file_manager.folder_dict['presets']
    new_filename = os.path.join(preset_path,file_type+'.json')
    preset_file_manager.save_file(data,new_filename)
    file_manager.file_dict[file_type] = new_filename
    



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
stimulus_manager = StimulusManager(db_handler)
preset_file_manager = PresetManager()
        

if  check_loadable_preset(file_manager,'arena_layout'):
    arena_layout = preset_file_manager.load_json(file_manager.file_dict['arena_layout'])
else:

    # Get all info for posthoc run
    clear_screen()
    arena_layout = get_arena_configuration()
    print(f"Configuration: {arena_layout['arena_num']} arenas arranged in {arena_layout['arena_rows']} rows and {arena_layout['arena_cols']} columns.")
    arena_layout = arena_manager.enter_arenas_for_experiment(arena_layout['arena_num'], arena_layout['arena_rows'], arena_layout['arena_cols'])
    save_preset(file_manager,preset_file_manager,'arena_layout',arena_layout)


if  check_loadable_preset(file_manager,'arena_layout'):
    arena_layout = preset_file_manager.load_json(file_manager.file_dict['arena_layout'])

clear_screen()

stim_layout = stimulus_manager.enter_stimuli_for_experiment(arena_layout['arena_num'], arena_layout['arena_rows'], arena_layout['arena_cols'])




experiment_preset = preset_file_manager.load_dialogue('whole experiment',('*.csv', '*.*'))

# Get automated info from posthoc run

video_info_extractor = VideoInfoExtractor(file_manager.file_dict['video_file_position'])
video_info = video_info_extractor.get_video_info()



# Start the process
experiment_info = experiment_manager.manage_experiments()

# Get arena layout and other configurations
flies = fly_manager.enter_flies_for_experiment()
fly_distribution_manager.enter_flies_for_experiment(total_arenas)
arenas = fly_distribution_manager.distribute_flies(9, 6)  # Example layout with rows and cols
fly_distribution_manager.show_arena_assignments()


