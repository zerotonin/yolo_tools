from .FlyChoiceDatabase import *
from prettytable import PrettyTable
import os

class ExperimentManager:
    """
    Manages the creation and management of experiments, including managing experimenters and
    experiment types.

    Attributes:
        db_handler (DatabaseHandler): The database handler for accessing experiment-related data.
    """

    def __init__(self, db_handler):
        """
        Initializes the ExperimentManager with a database handler.

        Args:
            db_handler (DatabaseHandler): The database handler for accessing experiment-related data.
        """
        self.db_handler = db_handler

    def _clear_screen(self):
        """
        Clears the terminal screen for better readability.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_experimenters(self):
        """
        Retrieves and formats a PrettyTable of all experimenters in the database.
        Returns:
            PrettyTable: Table containing the list of experimenters.
        """
        with self.db_handler as db:
            experimenters = db.get_records(Experimenter)
            table = PrettyTable()
            table.field_names = ["ID", "Name"]
            for experimenter in experimenters:
                table.add_row([experimenter.id, experimenter.name])
        return table

    def enter_new_experimenter(self):
        """
        Interactively creates a new experimenter and adds them to the database after user confirmation.
        """
        name = input("Enter the name of the new experimenter: ")
        if self.confirm_action("Are you sure you want to add this new experimenter?"):
            new_experimenter = Experimenter(name=name)
            with self.db_handler as db:
                db.add_record(new_experimenter)
            print("New experimenter added successfully.")
        else:
            print("Action canceled.")
        

    def show_experiment_types(self):
        """
        Retrieves and formats a PrettyTable of all experiment types in the database.
        Returns:
            PrettyTable: Table containing the list of experiment types.
        """
        with self.db_handler as db:
            types = db.get_records(ChoiceExperimentType)
            table = PrettyTable()
            table.field_names = ["ID", "Name"]
            for exp_type in types:
                table.add_row([exp_type.id, exp_type.name])
        return table
    
    def confirm_action(self, prompt_message):
        """
        Asks the user for confirmation before proceeding with an action.

        Args:
            prompt_message (str): The message to display to the user asking for confirmation.

        Returns:
            bool: True if the user confirms the action, False otherwise.
        """
        confirmation = input(prompt_message + " Type 'yes' to confirm: ").strip().lower()
        return confirmation == 'yes'


    def enter_new_experiment_type(self):
        """
        Interactively creates a new experiment type and adds it to the database after user confirmation.
        """
        name = input("Enter the name of the new experiment type: ")
        if self.confirm_action("Are you sure you want to add this new experiment type?"):
            new_type = ChoiceExperimentType(name=name)
            with self.db_handler as db:
                db.add_record(new_type)
            print("New experiment type added successfully.")
        else:
            print("Action canceled.")

    def display_tables_side_by_side(self):
        """
        Displays experimenters and experiment types side by side for easy comparison and selection.
        Uses padding to ensure both tables are aligned even if they have different numbers of rows.
        """
        self._clear_screen()
        experimenter_table = self.show_experimenters()
        type_table = self.show_experiment_types()

        # Split tables into lines
        exp_lines = experimenter_table.get_string().split('\n')
        type_lines = type_table.get_string().split('\n')
        
        # Calculate the maximum number of lines either table can have
        max_lines = max(len(exp_lines), len(type_lines))

        # Extend both lists to have the same number of lines
        exp_lines.extend([''] * (max_lines - len(exp_lines)))
        type_lines.extend([''] * (max_lines - len(type_lines)))

        # Format the lines with adequate spacing
        combined_lines = []
        for exp_line, type_line in zip(exp_lines, type_lines):
            # Ensure both lines have the same vertical alignment
            formatted_line = f"{exp_line.ljust(50)}{type_line}"
            combined_lines.append(formatted_line)

        # Print the combined lines for side-by-side display
        for line in combined_lines:
            print(line)



    def select_experimenter(self):
        """
        Allows the user to select an experimenter by ID from a displayed list.
        Returns:
            int: The ID of the selected experimenter, or None if selection is invalid.
        """
        self.show_experimenters()
        try:
            experimenter_id = int(input("Enter the ID of the experimenter you choose: "))
            with self.db_handler as db:
                experimenter = db.get_records(Experimenter, filters={'id': experimenter_id})
                if experimenter:
                    return experimenter_id
        except ValueError:
            pass
        print("Invalid ID. Please enter a valid numeric ID.")
        return None

    def select_experiment_type(self):
        """
        Allows the user to select an experiment type by ID from a displayed list.
        Returns:
            int: The ID of the selected experiment type, or None if selection is invalid.
        """
        self.show_experiment_types()
        try:
            experiment_type_id = int(input("Enter the ID of the experiment type you choose: "))
            with self.db_handler as db:
                experiment_type = db.get_records(ChoiceExperimentType, filters={'id': experiment_type_id})
                if experiment_type:
                    return experiment_type_id
        except ValueError:
            pass
        print("Invalid ID. Please enter a valid numeric ID.")
        return None

    def display_selected_options(self, experimenter_id, experiment_type_id):
        """
        Displays the names of the selected experimenter and experiment type.
        Args:
            experimenter_id (int): ID of the selected experimenter.
            experiment_type_id (int): ID of the selected experiment type.
        """
        with self.db_handler as db:
            experimenter = db.get_records(Experimenter, filters={'id': experimenter_id})[0].name if experimenter_id else "None"
            experiment_type = db.get_records(ChoiceExperimentType, filters={'id': experiment_type_id})[0].name if experiment_type_id else "None"
        print(f"Selected Experimenter: {experimenter}")
        print(f"Selected Experiment Type: {experiment_type}")

    def manage_experiments(self):
        """
        Main method to start the experiment management process. Allows the user to add, select
        experimenters and experiment types, showing both tables side by side for easy reference.
        Returns a dictionary with the selected experimenter and experiment type IDs.
        """
        experimenter_id = None
        experiment_type_id = None

        while True:
            self.display_tables_side_by_side()
            self.display_selected_options(experimenter_id, experiment_type_id)

            print("\nChoose an option:")
            print("1. Add Experimenter")
            print("2. Add Experiment Type")
            print("3. Select Experimenter")
            print("4. Select Experiment Type")
            print("5. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.enter_new_experimenter()
            elif choice == '2':
                self.enter_new_experiment_type()
            elif choice == '3':
                experimenter_id = self.select_experimenter()
            elif choice == '4':
                experiment_type_id = self.select_experiment_type()
            elif choice == '5':
                if experimenter_id and experiment_type_id:
                    return {'experimenter_id': experimenter_id, 'experiment_type_id': experiment_type_id}
                else:
                    print("Please ensure both experimenter and experiment type are selected before exiting.")
            else:
                print("Invalid choice. Please select a valid option.")


