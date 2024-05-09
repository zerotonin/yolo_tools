import os
from prettytable import PrettyTable
from database.ExperimentManager import ExperimentManager
from database.ArenaManager import ArenaManager
from database.FlyManager import FlyManager,FlyDistributionManager
from database.StimulusManager import StimulusManager
from movie_preprocessing.VideoInfoExtractor import VideoInfoExtractor
from database.FlyChoiceDatabase import DatabaseHandler
from analysis_file_manager.PresetManager import PresetManager
from analysis_file_manager.AnalysisFileManager import AnalysisFileManager


class ExperimentSetupManager:
    def __init__(self, base_output_path, db_file_path, video_file_path, python_env_path,yolo_weights):
        self.file_manager = AnalysisFileManager()
        self.file_manager.setup_experiment_paths(base_output_path,db_file_path, video_file_path, python_env_path,yolo_weights)
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
            self.save_preset('fly_layout', self.fly_layout)


    
    def generate_legends_and_maps(self):
        """Generate legends and maps for easy reference."""
        fly_legend = [(i, self.fly_manager.get_human_readable_fly_details(fly)) for i, fly in enumerate(self.flies)]
        stim_legend = [(stimulus, self.stimulus_manager.get_human_readable_stimulus_details(stimulus)) for stimulus in set(stim for arena in self.stim_layout for stim in arena)]
        arena_legend = [(arena, self.arena_manager.get_human_readable_arena_details(arena)) for arena in set(self.arena_layout)]

        stim_index_map = {item[0]: chr(65 + i) for i, item in enumerate(stim_legend)}  # 'A' to 'Z'
        arena_index_map = {item[0]: chr(97 + i) for i, item in enumerate(arena_legend)}  # 'a' to 'z'
        return fly_legend, stim_legend, arena_legend, stim_index_map, arena_index_map

    def prepare_data_for_display(self, rows, cols, stim_index_map, arena_index_map):
        """Prepare data for table display."""
        data = []
        for row_index in range(rows):
            for col_index in range(cols):
                fly_index = self.fly_layout[row_index][col_index]
                fly_info = f"F:{fly_index+1}" if fly_index is not None else "F:-"
                stim_ids = self.stim_layout[row_index * cols + col_index]
                stim_info = f"S:{''.join(stim_index_map.get(id, '-') for id in stim_ids)}" if stim_ids else "S:[]"
                arena_type = self.arena_layout[row_index * cols + col_index]
                arena_info = f"A:{arena_index_map.get(arena_type)}"
                data.append([row_index * cols + col_index + 1, stim_info, fly_info, arena_info])
        return data
    
    def print_legend(self,fly_legend,stim_legend,arena_legend,arena_index_map,stim_index_map):

        # Print legend
        print("\nLegend:")
        print("Flies:")
        for i, details in fly_legend:
            print(f"{i+1}: {details}")
        print("Stimuli:")
        for id, details in stim_legend:
            print(f"{stim_index_map.get(id, '-')}: {details}")
        print("Arenas:")
        for id, details in arena_legend:
            print(f"{arena_index_map.get(id, '-')}: {details}")

    def display_experiment_overview_arena_wise(self):
        """Main function to display experiment setup and optionally save to CSV."""
        rows, cols = self.arena_info['arena_rows'], self.arena_info['arena_cols']
        fly_legend, stim_legend, arena_legend, stim_index_map, arena_index_map = self.generate_legends_and_maps()
        data = self.prepare_data_for_display(rows, cols, stim_index_map, arena_index_map)
        
        # Displaying in pretty table for user interface
        table = PrettyTable()
        table.field_names = ['Arena Number', 'Stimulus', 'Fly', 'Arena Type']
        for row in data:
            table.add_row(row[:1]+[entry[2::] for entry in row[1::]])
        print(table)
        self.print_legend(fly_legend,stim_legend,arena_legend,arena_index_map,stim_index_map)

    def display_experiment_overview(self):
        """
        Displays an overview of the experiment setup in a grid format,
        where each cell contains a simple encoded format for flies, stimuli, and arenas.
        """

        # Table setup
        rows, cols = self.arena_info['arena_rows'], self.arena_info['arena_cols']
        fly_legend, stim_legend, arena_legend, stim_index_map, arena_index_map = self.generate_legends_and_maps()
        data = self.prepare_data_for_display(rows, cols, stim_index_map, arena_index_map)
        table = PrettyTable()
        headers = ["Row/Col"] + [f"Col {i+1}" for i in range(cols)]
        table.field_names = headers

        for row_index in range(rows):
            display_row = [f"Row {row_index + 1}"]
            for col_index in range(cols):
                arena=    data[row_index * cols + col_index]
                display_row.append(f"{arena[2]} {arena[1]} {arena[3]}")
            table.add_row(display_row)

        self.clear_screen()
        print(table)
        self.print_legend(fly_legend,stim_legend,arena_legend,arena_index_map,stim_index_map)

    def get_video_info(self):

        # Get automated info from posthoc run

        self.video_info_extractor = VideoInfoExtractor(self.file_manager.file_dict['video_file_position'])
        self.experiment_info = self.manage_preset('video_info', self.video_info_extractor.get_video_info)
    def display_experiment_overview(self):
        """
        Displays an overview of the experiment setup in a grid format,
        where each cell contains a simple encoded format for flies, stimuli, and arenas.
        """

        # Table setup
        rows, cols = self.arena_info['arena_rows'], self.arena_info['arena_cols']
        fly_legend, stim_legend, arena_legend, stim_index_map, arena_index_map = self.generate_legends_and_maps()
        data = self.prepare_data_for_display(rows, cols, stim_index_map, arena_index_map)
        table = PrettyTable()
        headers = ["Row/Col"] + [f"Col {i+1}" for i in range(cols)]
        table.field_names = headers

        for row_index in range(rows):
            display_row = [f"Row {row_index + 1}"]
            for col_index in range(cols):
                arena=    data[row_index * cols + col_index]
                display_row.append(f"{arena[2]} {arena[1]} {arena[3]}")
            table.add_row(display_row)

        self.clear_screen()
        print(table)
        self.print_legend(fly_legend,stim_legend,arena_legend,arena_index_map,stim_index_map)

    def get_video_info(self):

        # Get automated info from posthoc run

        self.video_info_extractor = VideoInfoExtractor(self.file_manager.file_dict['video_file_position'])
        self.experiment_info = self.manage_preset('video_info', self.video_info_extractor.get_video_info)