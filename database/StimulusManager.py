from FlyChoiceDatabase import *
from prettytable import PrettyTable
import os 
class StimulusManager:
    def __init__(self, db_handler):
        """
        Initializes the StimulusManager with a DatabaseHandler instance.

        Args:
            db_handler (DatabaseHandler): The database handler instance to manage database operations.
        """
        self.db_handler = db_handler
    

    def _clear_screen(self):
        """
        Clears the terminal screen.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_stimuli(self):
        """
        Prints all stimuli with their IDs for selection.
        """
        with self.db_handler as db:
            stimuli = db.get_records(Stimulus)
            table = PrettyTable()
            table.field_names = ["ID", "Name", "Type", "Amplitude", "Unit"]
            for stimulus in stimuli:
                table.add_row([stimulus.id, stimulus.name, stimulus.type, stimulus.amplitude, stimulus.amplitude_unit])
            print(table)

    def enter_new_stimulus(self):
        """
        Guides the user to enter a new stimulus into the databasun.
        """
        self.show_stimuli()
        name = input("Enter the name of the new stimulus: ")
        type_ = input("Enter the type of the new stimulus (e.g., 'chemical', 'light'): ") 
        amplitude = float(input("Enter the amplitude of the new stimulus: "))
        amplitude_unit = input("Enter the unit of amplitude (e.g., 'mV', 'lux'): ")

        new_stimulus = Stimulus(name=name, type=type_, amplitude=amplitude, amplitude_unit=amplitude_unit)
        with self.db_handler as db:
            db.add_record(new_stimulus)
        print("New stimulus added successfully.")

    def enter_stimulus_list_for_arena(self):
        """
        Collects a list of stimuli for an arena, adding new stimuli if necessary.
        """
        stimuli_list = []
        while len(stimuli_list) < 10:
            self._clear_screen()
            self.show_stimuli()
            stimulus_id = input("Enter stimulus ID or 'new' to add a new stimulus: ")
            if stimulus_id.lower() == 'new':
                self.enter_new_stimulus()
            else:
                with self.db_handler as db:
                    stimulus = db.get_records(Stimulus, filters={'id': int(stimulus_id)})
                    if stimulus:
                        stimuli_list.append(stimulus[0])
                    else:
                        print("Stimulus not found.")
            if input("Add more? (y/n): ").lower() != 'y':
                break
        return stimuli_list

    def pattern_uniform(self, number_of_arenas):
        """
        Assigns a uniform pattern of stimuli across all arenas.
        """
        stimulus_list = self.enter_stimulus_list_for_arena()
        assignments = [(stimulus.id for stimulus in stimulus_list)] * number_of_arenas
        return assignments

    def enter_stimuli_for_experiment(self, number_of_arenas, number_of_arena_rows, number_of_arena_columns):
        """
        Main method to start the stimuli assignment for an experiment.
        """
        pattern = input("Choose a pattern (uniform, checkerboard, horizontal, vertical, individual): ").lower()
        if pattern == 'uniform':
            return self.pattern_uniform(number_of_arenas)
        # Add elif branches for other patterns and implement the corresponding methods.
        else:
            print("Pattern not recognized.")
        return None

# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
stimulus_manager = StimulusManager(db_handler)

# To start entering stimuli for an experiment:
assignments = stimulus_manager.enter_stimuli_for_experiment(54, 9, 6)
print(assignments)