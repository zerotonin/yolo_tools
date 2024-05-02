import os
from prettytable import PrettyTable
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

    def manage_preset(self, key, creation_func, *args, **kwargs):
        """
        Manages loading or creating a preset. Checks if a preset is available and loads it;
        if not, it runs the specified creation function and saves the result as a new preset.

        Args:
            key (str): The key for the preset to check or create.
            creation_func (callable): Function to call to create the preset if it doesn't exist.
            *args: Arguments to pass to the creation function.
            **kwargs: Keyword arguments to pass to the creation function.

        Returns:
            The loaded or newly created data for the preset.
        """
        if self.check_loadable_preset(key):
            return self.load_preset(key)
        else:
            data = creation_func(*args, **kwargs)
            self.save_preset(key, data)
            return data

    def setup_experiments(self):
        """
        Setup and manage the entire experiment configuration using manage_preset method.
        """
        self.experiment_info = self.manage_preset('experiment_info', self.experiment_manager.manage_experiments)
        self.arena_info = self.manage_preset('arena_info', self.get_arena_configuration)
        self.arena_layout = self.manage_preset('arena_layout', self.arena_manager.enter_arenas_for_experiment,
                                          self.arena_info['arena_num'], self.arena_info['arena_rows'], self.arena_info['arena_cols'])
        self.stim_layout = self.manage_preset('stim_layout', self.stimulus_manager.enter_stimuli_for_experiment,
                                         self.arena_info['arena_num'], self.arena_info['arena_rows'], self.arena_info['arena_cols'])
        self.flies = self.manage_preset('flies', self.fly_manager.enter_flies_for_experiment)
        if self.check_loadable_preset('fly_layout'):
            self.fly_layout = self.load_preset('fly_layout')
        else:
            self.fly_distribution_manager.fly_data = self.flies
            self.fly_distribution_manager.enter_flies_for_experiment(self.arena_info['arena_num'])
            self.fly_layout =  self.fly_distribution_manager.distribute_flies(self.arena_info['arena_rows'], self.arena_info['arena_cols'])

    def display_experiment_overview(self):
        fly_legend = list()
        for fly in self.flies:
            fly_legend.append(self.fly_manager.get_human_readable_fly_details(fly))

        stim_legend = list()
        for stimulus in set(stim for arena in self.stim_layout for stim in arena):
            stim_legend.append(self.stimulus_manager.get_human_readable_stimulus_details(stimulus))
            
        pass

# Example usage of the setup_experiments method
if __name__ == "__main__":
    base_output_path = '/home/geuba03p/food_example_video/output'
    db_file_path = '/home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
    video_file_path = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
    python_interp = '/home/geuba03p/miniconda3/envs/yolov8/bin/python'

    experiment_setup = ExperimentSetupManager(base_output_path, db_file_path, video_file_path, python_interp)
    experiment_setup.setup_experiments()
    experiment_setup.display_experiment_overview()





# Get automated info from posthoc run

video_info_extractor = VideoInfoExtractor(file_manager.file_dict['video_file_position'])
video_info = video_info_extractor.get_video_info()




# Get arena layout and other configurations

