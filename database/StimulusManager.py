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
                        stimuli_list.append(stimulus[0].id)
                    else:
                        print("Stimulus not found.")
            if input("Add more? (y/n): ").lower() != 'y':
                break
        return stimuli_list

    def enter_stimuli_for_experiment(self, number_of_arenas, number_of_arena_rows, number_of_arena_columns):
        """
        Main method to start the stimuli assignment for an experiment.
        Asks the user for the pattern type and generates a formatted table of stimuli assignments.
        """
        pattern = input("Choose a pattern (uniform, checkerboard, horizontal, vertical, individual): ").lower()
        pattern_functions = {
            'uniform': self.pattern_uniform,
            # other patterns mapped to their respective functions
        }

        if pattern in pattern_functions:
            assignments = pattern_functions[pattern](number_of_arenas)
            self.display_stimuli_grid(assignments, number_of_arena_rows, number_of_arena_columns)
            return assignments
        else:
            print("Pattern not recognized.")

    def display_stimuli_grid(self, assignments, rows, cols):
        """
        Displays the stimuli assignments in a grid format based on the rows and columns,
        and provides a detailed legend with information about each used stimulus.
        """
        # Extract unique stimulus IDs from assignments
        unique_stimulus_ids = set()
        for assignment in assignments:
            unique_stimulus_ids.update(assignment)

        # Fetch details only for the stimuli used in the grid
        stimuli_details = {}
        with self.db_handler as db:
            # Filters to fetch only relevant stimuli
            filtered_stimuli = db.get_records(Stimulus, filters={'id': unique_stimulus_ids})
            for stimulus in filtered_stimuli:
                stimuli_details[stimulus.id] = (stimulus.name, stimulus.type, stimulus.amplitude, stimulus.amplitude_unit)

        table = PrettyTable()
        table.field_names = [f"Column {i+1}" for i in range(cols)]

        index = 0
        for _ in range(rows):
            row_entries = []
            for _ in range(cols):
                if index < len(assignments):
                    row_entries.append(assignments[index])
                    index += 1
                else:
                    row_entries.append("None")
            table.add_row(row_entries)

        self._clear_screen()
        print(table)
        
        # Print the legend with detailed stimulus information
        print("\nLegend:")
        print("Each cell represents the stimulus IDs assigned to an arena. 'None' indicates no assignment.")
        for stimulus_id, details in stimuli_details.items():
            name, type_, amplitude, unit = details
            print(f"ID: {stimulus_id}, Name: {name}, Type: {type_}, Amplitude: {amplitude} {unit}")
        print('')
        _=input("Press Enter to continue...")



    def pattern_uniform(self, number_of_arenas):
        """
        Assigns a uniform pattern of stimuli across all arenas.
        """
        stimulus_list = self.enter_stimulus_list_for_arena()
        assignments = [stimulus_list for i in range(number_of_arenas)]
        return assignments

# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
stimulus_manager = StimulusManager(db_handler)

# To start entering stimuli for an experiment:
assignments = stimulus_manager.enter_stimuli_for_experiment(54, 9, 6)
print(assignments)