from database.FlyChoiceDatabase import *
from prettytable import PrettyTable
from collections import defaultdict
import os,json,csv


def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')

def save_flies_to_json(filename,fly_data):
    with open(filename, 'w') as f:
        json.dump(fly_data, f, indent=4)

def load_flies_from_json( filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def show_flies(self):
    """
    Displays all flies in the flies list with human-readable details.
    """
    clear_screen()
    table = PrettyTable()
    table.field_names = ["ID", "Is Female", "Genotype Name", "Age (days)", "Attributes"]
    for index, fly in enumerate(self.flies):
        genotype_name = self.genotype_manager.get_genotype_name(fly['genotype_id'])
        attribute_names = self.attribute_manager.get_attribute_names(fly['attribute_ids'])
        table.add_row([
            index + 1,
            "Yes" if fly['is_female'] else "No",
            genotype_name,
            fly['age_day_after_eclosion'],
            ', '.join(attribute_names)
        ])
    print(table)

class FlyAttributeManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def select_or_create_attribute(self):
        if input("Would you like to enter an attribute for the fly? (yes/no): ").lower() != 'yes':
            return None

        with self.db_handler as db:
            attributes = db.get_records(FlyAttribute)
            self.display_attributes(attributes)

            choice = input("Enter attribute ID to select or 'new' to create a new attribute: ")
            if choice.lower() == 'new':
                return self.create_new_attribute(db)
            else:
                attribute = db.get_records(FlyAttribute, filters={'id': int(choice)})
                return attribute[0].id if attribute else None

    def create_new_attribute(self, db):
        name = input("Enter the name of the new attribute: ")
        new_attribute = FlyAttribute(name=name)
        db.add_record(new_attribute)
        return new_attribute.id

    def display_attributes(self, attributes):
        clear_screen()
        table = PrettyTable()
        table.field_names = ["ID", "Name"]
        for attribute in attributes:
            table.add_row([attribute.id, attribute.name])
        print(table)
    
    def get_attribute_names(self, attribute_ids):
        names = []
        with self.db_handler as db:
            for attribute_id in attribute_ids:
                attribute = db.get_records(FlyAttribute, {'id': attribute_id})
                if attribute:
                    names.append(attribute[0].name)
        return names
    
class GenotypeManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def select_or_create_genotype(self):
        with self.db_handler as db:
            genotypes = db.get_records(Genotype)
            self.display_genotypes(genotypes)

            choice = input("Enter genotype ID to select or 'new' to create a new genotype: ")
            if choice.lower() == 'new':
                return self.create_new_genotype(db)
            else:
                genotype = db.get_records(Genotype, filters={'id': int(choice)})
                return genotype[0].id if genotype else None

    def create_new_genotype(self, db):
        shortname = input("Enter the short name of the new genotype: ")
        genotype_desc = input("Enter the detailed genotype description: ")
        new_genotype = Genotype(shortname=shortname, genotype=genotype_desc)
        db.add_record(new_genotype)
        return new_genotype.id

    
    def get_genotype_name(self, genotype_id):
        with self.db_handler as db:
            genotype = db.get_records(Genotype, {'id': genotype_id})
            return genotype[0].shortname if genotype else 'Unknown'
        

    def display_genotypes(self,genotypes):
        clear_screen()
        table = PrettyTable()
        table.field_names = ["ID", "Shortname", "Description"]
        for genotype in genotypes:
            table.add_row([genotype.id, genotype.shortname, genotype.genotype])
        print(table)

class FlyManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.attribute_manager = FlyAttributeManager(db_handler)
        self.genotype_manager = GenotypeManager(db_handler)
        self.flies = []  # List to store fly data temporarily

    def enter_new_fly(self):
        is_female = input("Is the fly female? (yes/no): ").lower() == 'yes'
        age = float(input("Enter the age of the fly in days after eclosion: "))
        genotype_id = self.genotype_manager.select_or_create_genotype()

        attributes = []
        print("You can enter up to 5 attributes for the fly.")
        for _ in range(5):
            attribute_id = self.attribute_manager.select_or_create_attribute()
            if attribute_id:
                attributes.append(attribute_id)
            if input("Add more attributes? (y/n): ").lower() != 'y':
                break

        fly_data = {
            'is_female': is_female,
            'age_day_after_eclosion': age,
            'genotype_id': genotype_id,
            'attribute_ids': attributes
        }
        self.flies.append(fly_data)
        print("Fly details added successfully.")

    def show_flies(self):
        """
        Displays all flies in the flies list with human-readable details.
        """
        clear_screen()
        table = PrettyTable()
        table.field_names = ["ID", "Is Female", "Genotype Name", "Age (days)", "Attributes"]
        for index, fly in enumerate(self.flies):
            genotype_name = self.genotype_manager.get_genotype_name(fly['genotype_id'])
            attribute_names = self.attribute_manager.get_attribute_names(fly['attribute_ids'])
            table.add_row([
                index + 1,
                "Yes" if fly['is_female'] else "No",
                genotype_name,
                fly['age_day_after_eclosion'],
                ', '.join(attribute_names)
            ])
        print(table)

    def enter_flies_for_experiment(self):
        while True:
            clear_screen()
            if self.flies:
                print("Current Flies in Session:\n")
                self.show_flies()
            else:
                print("No flies entered yet.\n")

            user_input = input("Would you like to enter a new fly? (yes/no): ").lower()
            if user_input == 'yes':
                self.enter_new_fly()
            elif user_input == 'no':
                print("Exiting program...")
                return self.flies
            else:
                print("Invalid input, please type 'yes' or 'no'.")

class FlyDistributionManager:
    """
    Manages the distribution of flies across various experimental arenas, handling
    fly data initialization, display, and adjustments for experiments.

    Attributes:
        fly_data (list of dicts): Stores the details of each fly involved in the experiment.
        arenas (list of list of ints): Represents a 2D grid of arenas where each cell contains a fly index.
        rows (int): Number of rows in the arena grid.
        cols (int): Number of columns in the arena grid.
        fly_type_to_number (dict): Maps unique fly type configurations to display numbers for legends.
        genotype_manager (GenotypeManager): Manages genotype related operations.
        attribute_manager (FlyAttributeManager): Manages fly attribute related operations.
    """
    def __init__(self,db_handler,fly_dict_list):
        """
        Initializes the FlyDistributionManager with a database handler and a list of fly dictionaries.

        Args:
            db_handler (DatabaseHandler): The database handler for accessing fly-related data.
            fly_dict_list (list of dict): List of dictionaries, each containing details of a fly.
        """
        self.fly_data = fly_dict_list  # List to store fly data dictionaries
        self.initialize_participation_counts()
        self.arenas = None #init
        self.rows = None
        self.cols = None
        self.fly_type_to_number = dict()
        self.genotype_manager = GenotypeManager(db_handler)  
        self.attribute_manager = FlyAttributeManager(db_handler)

    def initialize_participation_counts(self):
        """
        Ensures that each fly dictionary has a 'participation_count' field initialized properly.
        Sets the count to 0 if it's not already present or if it's set to None.
        """
        for fly in self.fly_data:
            if 'participation_count' not in fly or fly['participation_count'] is None:
                fly['participation_count'] = 0
    
    def display_fly_data(self):
        """
        Displays the current configuration of flies using a pretty table format. Shows details
        like ID, gender, genotype, age, attributes, and participation count for each fly.
        """
        table = PrettyTable()
        table.field_names = ["ID", "Is Female", "Genotype", "Age (days)", "Attributes", "Participation Count"]
        for index, fly in enumerate(self.fly_data):
            self.initialize_participation_counts()  # Ensure all data is initialized

            # Fetch human-readable details from managers
            genotype_name = self.genotype_manager.get_genotype_name(fly['genotype_id'])
            attribute_names = self.attribute_manager.get_attribute_names(fly['attribute_ids'])
            attributes_display = ', '.join(attribute_names)

            table.add_row([
                index + 1,
                "Yes" if fly['is_female'] else "No",
                genotype_name,
                fly['age_day_after_eclosion'],
                attributes_display,
                fly['participation_count']
            ])
        print(table)


    def enter_flies_for_experiment(self, total_arenas):
        """
        Manages the interactive process of configuring flies for an experiment. Allows the user
        to adjust the number of participating flies to match the number of available arenas.

        Args:
            total_arenas (int): The total number of available arenas for the experiment.
        """

        while True:
            self.initialize_participation_counts()  # Ensure participation count is initialized
            clear_screen()
            print("Current fly configurations:")
            self.display_fly_data()
            num_flies = sum(fly['participation_count'] for fly in self.fly_data)
            print(f"Total flies configured: {num_flies}")
            print(f"Total arenas available: {total_arenas}")

            if num_flies > total_arenas:
                print("There are more flies than arenas. Please reduce the number of flies.")
            elif num_flies < total_arenas:
                print("There are fewer flies than arenas. Distribution will try to spread them evenly.")
            else:
                break

            modification = input("Would you like to adjust the flies? (adjust/done): ").lower()
            if modification == 'adjust':
                self.adjust_fly_numbers()
            elif modification == 'done':
                break

    def distribute_flies(self, rows, cols):
        """
        Distributes flies into a specified grid of arenas based on the fly type and participation count.
        Attempts to place flies into the grid while respecting their specified participation counts.

        Args:
            rows (int): Number of rows in the grid of arenas.
            cols (int): Number of columns in the grid of arenas.
        """
        self.cols = cols
        self.rows = rows
        arenas = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        grouped_flies = {}  # Group flies by type

        # Group flies by genotype and attributes, including participation count
        for i, fly in enumerate(self.fly_data):
            key = (fly['genotype_id'], tuple(fly.get('attribute_ids', [])), fly['is_female'])
            if key not in grouped_flies:
                grouped_flies[key] = []
            for _ in range(fly['participation_count']):  # Respect participation count
                grouped_flies[key].append(i)

        # Sort and place flies in arenas
        current_row, current_col = 0, 0
        for flies in grouped_flies.values():
            for fly_index in flies:
                if current_col >= self.cols:
                    current_row += 1
                    current_col = 0
                if current_row >= self.rows:
                    print("Not enough space in arenas to place all flies.")
                    return arenas
                arenas[current_row][current_col] = fly_index
                current_col += 1

        self.arenas = arenas
    
    def generate_legend(self):
        """
        Generates a legend that maps fly type numbers to their descriptions for easier identification
        in the display table. Includes genotype, sex, and attributes of each fly type.

        Returns:
            str: A string representing the legend for fly types in the table.
        """
        legend = "Legend:\n"
        legend += "0: Empty arena (-)\n"  # Represent empty arenas
        

        for fly_type_key, number in self.fly_type_to_number.items():
            # Fetch genotype name from genotype ID
            genotype_name = self.genotype_manager.get_genotype_name(fly_type_key[0])
            
            # Fetch attribute names from attribute IDs
            attribute_names = self.attribute_manager.get_attribute_names(fly_type_key[2])

            # Prepare attributes description
            attributes = ", ".join(attribute_names)

            # Append details to the legend
            legend += f"{number}: Genotype {genotype_name}, {'Female' if fly_type_key[1] else 'Male'}, Attributes: [{attributes}]\n"

        return legend

    def prepare_arena_table(self):
        """
        Prepares a PrettyTable to display the current fly assignments in the arenas with a row and
        column format. Each cell in the table represents an arena and contains the number representing
        a fly type as defined in the legend.

        Returns:
            PrettyTable: A table object with fly assignments across the configured grid of arenas.
        """
        table = PrettyTable()
        headers = ["Row/Col"] + [f"Col {i+1}" for i in range(len(self.arenas[0]))]
        table.field_names = headers

        display_arenas = []
        for row_index, row in enumerate(self.arenas):
            display_row = [f"Row {row_index + 1}"]
            for index in row:
                if index is None:
                    display_row.append("-")
                else:
                    fly = self.fly_data[index]
                    fly_type_key = (
                        fly['genotype_id'],
                        'Female' if fly['is_female'] else 'Male',
                        tuple(fly.get('attribute_ids', []))
                    )
                    if fly_type_key not in self.fly_type_to_number:
                        self.fly_type_to_number[fly_type_key] = len(self.fly_type_to_number) + 1
                    display_row.append(self.fly_type_to_number[fly_type_key])
            display_arenas.append(display_row)

        for row in display_arenas:
            table.add_row(row)
        
        return table

    def show_arena_assignments(self):
        """
        Displays the current assignments of flies to arenas if any are set up. It uses a PrettyTable
        to represent the layout of arenas and includes a generated legend to help identify the flies
        in each arena based on their types.

        Uses:
            prepare_arena_table(): To get the table layout of the arenas.
            generate_legend(): To provide the legend corresponding to the fly types in the table.
        """
        if not self.arenas:  # Check if arenas is None or empty
            print("No arena data available to display.")
            return
        
        clear_screen()

        # Prepare the table with arena assignments
        table = self.prepare_arena_table()

        # Generate the legend
        legend = self.generate_legend()

        print(table)
        print(legend)

    def display_fly_details(self, fly_index):
        """
        Displays detailed information for a specific fly based on its index in the fly_data list.
        The details include the fly's genotype, whether it is female, age, attributes, and participation count.

        Args:
            fly_index (int): The index of the fly in the fly_data list.
        """
        fly = self.fly_data[fly_index]

        # Fetch human-readable details from managers
        genotype_name = self.genotype_manager.get_genotype_name(fly['genotype_id'])
        attribute_names = self.attribute_manager.get_attribute_names(fly['attribute_ids'])
        attributes_display = ', '.join(attribute_names)

        details = (
            f"Genotype: {genotype_name}, "
            f"Is Female: {'Yes' if fly['is_female'] else 'No'}, "
            f"Age (days): {fly['age_day_after_eclosion']}, "
            f"Attributes: [{attributes_display}], "
            f"Participation Count: {fly['participation_count']}"
        )
        print(f"Current configuration for Fly ID {fly_index + 1}:")
        print(details)

    def adjust_fly_numbers(self):
        """
        Allows the user to adjust the participation count of a specific fly. The function prompts the user
        to select a fly by its ID, displays the current details of the selected fly, and then allows the
        user to input a new participation count.

        This function uses:
            display_fly_data(): To display the list of all flies for selection.
            display_fly_details(fly_index): To show details of the selected fly.
        """
        clear_screen()
        # Display fly data to choose from
        self.display_fly_data()

        try:
            # Ask user for fly ID to adjust
            fly_id = int(input("Enter the ID of the fly to adjust (choose by ID number): "))
            if fly_id < 1 or fly_id > len(self.fly_data):
                print("Invalid ID. Please enter a correct ID.")
                return

            # Adjust index to match list indexing (list is 0-indexed)
            fly_index = fly_id - 1

            # Display current fly information using the subfunction
            self.display_fly_details(fly_index)

            # Prompt for new participation count
            new_count = input("Enter new participation count for this fly: ")
            if new_count.isdigit() and int(new_count) >= 0:
                self.fly_data[fly_index]['participation_count'] = int(new_count)
                print("Participation count updated successfully.")
            else:
                print("Invalid input. Participation count must be a non-negative integer.")

        except ValueError:
            print("Invalid input. Please enter a numerical ID.")

    def save_pretty_table_to_text(self, filename):
        """
        Saves the current arena assignment table and the corresponding legend to a text file.

        Args:
            filename (str): The path and filename where the table and legend should be saved.
        """

        table = self.prepare_arena_table()
        legend = self.generate_legend()
        # Write to text file
        with open(filename, 'w') as file:
            file.write(str(table))
            file.write("\n\n" + legend)

    def save_sorted_csv(self, filename):
        """
        Exports the arena assignments to a CSV file. Each row in the CSV corresponds to a fly in an arena,
        with detailed fly information and the arena number.

        Args:
            filename (str): The path and filename where the CSV file should be saved.
        """
        # Prepare headers for CSV file
        headers = [
            "Arena Number", "Is Female", "Genotype ID", "Age (days)", 
            "Attribute 1", "Attribute 2", "Attribute 3", "Attribute 4", "Attribute 5"
        ]

        # Prepare data for CSV
        csv_data = []
        for row_idx, row in enumerate(self.arenas):
            for col_idx, fly_index in enumerate(row):
                if fly_index is not None:
                    fly = self.fly_data[fly_index]
                    attributes = fly.get('attribute_ids', [])
                    
                    # Ensure there are exactly five attribute columns
                    attributes += [None] * (5 - len(attributes))  # Fill missing attributes with None
                    
                    # Convert (row, col) into a 1-indexed arena number
                    arena_number = row_idx * len(row) + col_idx + 1

                    csv_data.append([
                        arena_number,
                        True if fly['is_female'] else False,
                        fly['genotype_id'],
                        fly['age_day_after_eclosion']
                    ] + attributes[:5])  # Ensure only the first five attributes are included

        # Sort data by the arena number
        csv_data.sort(key=lambda x: x[0])  # Sorting by arena number

        # Write data to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for row in csv_data:
                # Convert None to an empty string for better CSV formatting
                formatted_row = ["" if item is None else item for item in row]
                writer.writerow(formatted_row)
            
    def export_fly_data(self, text_filename,csv_filename):
        """
        Exports all fly data and arena assignments both as a text file with a pretty table and as a CSV file.
        The text file includes a pretty table of the arena assignments along with a legend,
        and the CSV file contains detailed fly data sorted by arena number.

        Args:
            text_filename (str): The file path and name where the pretty table and legend should be saved.
            csv_filename (str): The file path and name where the CSV file should be saved.

        Note:
            The function assumes that the directories for both filenames already exist.
            Ensure directories are created before calling this function if unsure.
        """
       
        # Call the function to save PrettyTable and legend as text
        self.save_pretty_table_to_text(text_filename)

        # Call the function to save CSV sorted by arenas
        self.save_sorted_csv(csv_filename)

