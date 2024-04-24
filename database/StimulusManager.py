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
        Prints all stimuli with their IDs and associated attributes for selection.
        """
        with self.db_handler as db:
            stimuli = db.get_records(Stimulus)
            table = PrettyTable()
            table.field_names = ["ID", "Name", "Type", "Amplitude", "Unit", "Attributes"]
            for stimulus in stimuli:
                # Fetching attributes for each stimulus
                attributes = ', '.join([attr.name for attr in stimulus.attributes])
                table.add_row([stimulus.id, stimulus.name, stimulus.type, stimulus.amplitude, stimulus.amplitude_unit, attributes])
            print(table)

    def enter_new_stimulus(self):
        """
        Guides the user to enter a new stimulus into the database, including attributes.
        """
        self._clear_screen()
        self.show_stimuli()
        name = input("Enter the name of the new stimulus: ")
        type_ = input("Enter the type of the new stimulus (e.g., 'chemical', 'light'): ")
        amplitude = float(input("Enter the amplitude of the new stimulus: "))
        amplitude_unit = input("Enter the unit of amplitude (e.g., 'mV', 'lux'): ")

        attribute_manager = StimulusAttributeManager(self.db_handler)
        attributes = []
        for _ in range(5):  # Allows entry for up to 5 attributes
            attribute = attribute_manager.select_or_create_attribute()
            if attribute:  # Ensure attribute is not None
                attributes.append(attribute)
            if input("Add more? (y/n): ").lower() != 'y' or not attribute:
                break

        new_stimulus = Stimulus(name=name, type=type_, amplitude=amplitude, amplitude_unit=amplitude_unit, attributes=attributes)
        with self.db_handler as db:
            db.add_record(new_stimulus)
        print("New stimulus added successfully.")

    def enter_stimulus_list_for_arena(self):
        """
        Collects a list of stimuli for an arena, adding new stimuli if necessary, and shows chosen IDs.
        """
        stimuli_list = []
        while len(stimuli_list) < 10:
            self._clear_screen()
            self.show_stimuli()
            # Display currently selected stimulus IDs
            if stimuli_list:
                print("Currently selected Stimulus IDs:", ', '.join(map(str, stimuli_list)))
            
            stimulus_id = input("Enter stimulus ID or 'new' to add a new stimulus: ")
            if stimulus_id.lower() == 'new':
                self.enter_new_stimulus()
                continue  # After adding new stimulus, show updated list
            else:
                try:
                    stimulus_id = int(stimulus_id)  # Ensure it's an integer
                    with self.db_handler as db:
                        stimulus = db.get_records(Stimulus, filters={'id': stimulus_id})
                        if stimulus:
                            stimuli_list.append(stimulus[0].id)
                            print(f"Added Stimulus ID {stimulus[0].id}.")
                        else:
                            print("Stimulus not found.")
                except ValueError:
                    print("Please enter a valid numeric ID.")
            
            if len(stimuli_list) >= 10:
                print("Maximum of 10 stimuli reached.")
                break
            
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
                # Fetch attribute names and create a comma-separated string
                attribute_names = ', '.join([attr.name for attr in stimulus.attributes])
                stimuli_details[stimulus.id] = (stimulus.name, stimulus.type, stimulus.amplitude, stimulus.amplitude_unit, attribute_names)

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
        
        # Print the legend with detailed stimulus information including attributes
        print("\nLegend:")
        print("Each cell represents the stimulus IDs assigned to an arena. 'None' indicates no assignment.")
        for stimulus_id, details in stimuli_details.items():
            name, type_, amplitude, unit, attributes = details
            print(f"ID: {stimulus_id}, Name: {name}, Type: {type_}, Amplitude: {amplitude} {unit}, Attributes: {attributes}")
        print('')
        _=input("Press Enter to continue...")



    def pattern_uniform(self, number_of_arenas):
        """
        Assigns a uniform pattern of stimuli across all arenas.
        """
        stimulus_list = self.enter_stimulus_list_for_arena()
        assignments = [stimulus_list for i in range(number_of_arenas)]
        return assignments


class StimulusAttributeManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def select_or_create_attribute(self):
        """
        Allows the user to select an existing attribute or create a new one.
        Returns the selected or newly created attribute.
        """
        with self.db_handler as db:
            attributes = db.get_records(StimuliAttribute)
            print("\nExisting Attributes:")
            for attr in attributes:
                print(f"ID: {attr.id}, Name: {attr.name}")

            choice = input("Enter attribute ID to select or 'new' to create a new attribute: ")
            if choice.lower() == 'new':
                return self.create_new_attribute()
            else:
                # Make sure to return a single StimuliAttribute object
                attribute = db.get_records(StimuliAttribute, filters={'id': int(choice)})
                return attribute[0] if attribute else None
            
    def create_new_attribute(self):
        """
        Guides the user to create a new stimulus attribute.
        """
        name = input("Enter the name of the new attribute: ")
        new_attribute = StimuliAttribute(name=name)
        with self.db_handler as db:
            db.add_record(new_attribute)
        return new_attribute


# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
stimulus_manager = StimulusManager(db_handler)

# To start entering stimuli for an experiment:
assignments = stimulus_manager.enter_stimuli_for_experiment(54, 9, 6)
print(assignments)