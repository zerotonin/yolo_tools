import os
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from database.StimulusManager import StimulusManager
from movie_preprocessing.VideoInfoExtractor import VideoInfoExtractor
from database.FlyChoiceDatabase import DatabaseHandler
from AnalysisFileManager.PresetManager import PresetManager
from AnalysisFileManager.AnalysisFileManager import AnalysisFileManager


class ExperimentSetupManager:
    def __init__(self, base_output_path, db_file_path, video_file_path, python_env_path):
        self.file_manager = AnalysisFileManager()
        self.file_manager.setup_experiment_paths(base_output_path,db_file_path, video_file_path, python_env_path)
        self.db_handler = DatabaseHandler(f'sqlite:///{db_file_path}')
        self.experiment_manager = ExperimentManager(self.db_handler)
        self.arena_manager = ArenaManager(self.db_handler)
        self.fly_manager = FlyManager(self.db_handler)
        self.fly_distribution_manager = FlyDistributionManager(self.db_handler, {})
        self.stimulus_manager = StimulusManager(self.db_handler)
        self.preset_manager = PresetManager()
        self.load_existing_presets()

    def clear_screen(self):
        """ Clears the terminal screen for better readability. """
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_arena_configuration(self):
        """ Prompts user to input number of arenas and their arrangement. """
        self.clear_screen()
        print("Please enter the details for arena configuration.")
        try:
            total_arenas = int(input("Enter the total number of arenas in the experiment: "))
            rows = int(input("Enter the number of rows of arenas: "))
            columns = int(input("Enter the number of columns of arenas: "))
            if rows * columns != total_arenas:
                raise ValueError("The product of rows and columns must equal the total number of arenas.")
        except ValueError as e:
            print(f"Invalid input: {e}")
            return self.get_arena_configuration()
        return {'arena_num': total_arenas, 'arena_rows': rows, 'arena_cols': columns}

    def check_loadable_preset(self, file_type):
        """ Checks if a preset can be loaded based on the file type. """
        return file_type in list(self.file_manager.file_dict.keys())

    def save_preset(self, file_type, data):
        """ Saves a preset for a given file type. """
        preset_path = self.file_manager.folder_dict['presets']
        new_filename = os.path.join(preset_path, file_type + '.json')
        self.preset_manager.save_file(data, new_filename)
        self.file_manager.file_dict[file_type] = new_filename

    def load_preset(self, file_type):
        return self.preset_manager.load_json(self.file_manager.file_dict[file_type])
    
    def load_existing_presets(self):
        """
        Scans the preset folder for any existing files and updates the file_dict with these files.
        Uses the basename of each file as the key and the full path as the value.
        """
        preset_folder = self.file_manager.folder_dict['presets']
        try:
            # List all files in the preset folder
            for filename in os.listdir(preset_folder):
                file_path = os.path.join(preset_folder, filename)
                if os.path.isfile(file_path):
                    # Use the file's basename without extension as the key
                    key = os.path.splitext(filename)[0]
                    self.file_manager.file_dict[key] = file_path
                    print(f"Preset Found: {key} -> {file_path}")
        except FileNotFoundError:
            print("Preset folder not found.")
        except Exception as e:
            print(f"An error occurred: {e}")




# file and folder paths
db_file_position = '/home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
video_file_position = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
base_output_path = '/home/geuba03p/food_example_video/output'
python_interp = '/home/geuba03p/miniconda3/envs/yolov8/bin/python'

experiment = ExperimentSetupManager(base_output_path,db_file_position,video_file_position,python_interp)

if experiment.check_loadable_preset('experiment_info'):
    experiment_info = experiment.load_preset('experiment_info')
else:
    # Start the process
    experiment_info = experiment.experiment_manager.manage_experiments()
    experiment.save_preset('experiment_info',experiment_info)

if  experiment.check_loadable_preset('arena_layout'):
    arena_layout = experiment.load_preset('arena_layout')
    arena_info = experiment.load_preset('arena_info')
else:
    experiment.clear_screen()
    arena_info =  experiment.get_arena_configuration()
    print(f"Configuration: {arena_info['arena_num']} arenas arranged in {arena_info['arena_rows']} rows and {arena_info['arena_cols']} columns.")
    arena_layout =  experiment.arena_manager.enter_arenas_for_experiment(arena_info['arena_num'], arena_info['arena_rows'], arena_info['arena_cols'])
    experiment.save_preset('arena_layout',arena_layout)
    experiment.save_preset('arena_info',arena_info)


if  experiment.check_loadable_preset('stim_layout'):
    stim_layout = experiment.load_preset('stim_layout')
else:
    experiment.clear_screen()
    stim_layout =  experiment.stimulus_manager.enter_stimuli_for_experiment(arena_info['arena_num'], arena_info['arena_rows'], arena_info['arena_cols'])
    experiment.save_preset('stim_layout',stim_layout)



if  experiment.check_loadable_preset('flies'):
    flies = experiment.load_preset('fly_layout')
else:
    flies = experiment.fly_manager.enter_flies_for_experiment()
    experiment.save_preset('flies',flies)

    
if  experiment.check_loadable_preset('fly_layout'):
    fly_layout = experiment.load_preset('fly_layout')
else:
    experiment.fly_distribution_manager.fly_data = flies
    experiment.fly_distribution_manager.enter_flies_for_experiment(arena_info['arena_num'])
    fly_layout = experiment.fly_distribution_manager.distribute_flies(arena_info['arena_rows'], arena_info['arena_cols'])  # Example layout with rows and cols
    experiment.fly_distribution_manager.show_arena_assignments()
    experiment.save_preset('fly_layout',fly_layout)





# Get automated info from posthoc run

video_info_extractor = VideoInfoExtractor(file_manager.file_dict['video_file_position'])
video_info = video_info_extractor.get_video_info()




# Get arena layout and other configurations

