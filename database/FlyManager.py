from FlyChoiceDatabase import *
from prettytable import PrettyTable
from collections import defaultdict
import os,json


def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')

def save_flies_to_json(filename,fly_data):
    with open(filename, 'w') as f:
        json.dump(fly_data, f, indent=4)

def load_flies_from_json( filename):
    with open(filename, 'r') as f:
        return json.load(f)

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
    def __init__(self,fly_dict_list):
        self.fly_data = fly_dict_list  # List to store fly data dictionaries
        self.initialize_participation_counts()

    def initialize_participation_counts(self):
        for fly in self.fly_data:
            if 'participation_count' not in fly or fly['participation_count'] is None:
                fly['participation_count'] = 0
    
    def display_fly_data(self):
        table = PrettyTable()
        table.field_names = ["ID", "Is Female", "Genotype ID", "Age (days)", "Attribute IDs", "Participation Count"]
        for index, fly in enumerate(self.fly_data):
            self.initialize_participation_counts()  # Ensure all data is initialized
            attributes = ', '.join(map(str, fly.get('attribute_ids', [])))  # Handle if attribute_ids key is missing
            table.add_row([
                index + 1,
                "Yes" if fly['is_female'] else "No",
                fly['genotype_id'],
                fly['age_day_after_eclosion'],
                attributes,
                fly['participation_count']
            ])
        print(table)

    def enter_flies_for_experiment(self, total_arenas):

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
        arenas = [[None for _ in range(cols)] for _ in range(rows)]
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
                if current_col >= cols:
                    current_row += 1
                    current_col = 0
                if current_row >= rows:
                    print("Not enough space in arenas to place all flies.")
                    return arenas
                arenas[current_row][current_col] = fly_index
                current_col += 1

        return arenas


    def show_arena_assignments(self, arenas):
        if not arenas:  # Check if arenas is None or empty
            print("No arena data available to display.")
            return
        clear_screen()
        table = PrettyTable()
        headers = ["Row/Col"] + [f"Col {i+1}" for i in range(len(arenas[0]))]
        table.field_names = headers

        # Fly type to number mapping (legend)
        fly_type_to_number = {}
        legend = "Legend:\n"
        legend += "0: Empty arena (-)\n"  # Adding empty arena representation to the legend

        # Preparing display rows and updating the legend
        display_arenas = []
        for row_index, row in enumerate(arenas):
            display_row = [f"Row {row_index + 1}"]  # Row label
            for index in row:
                if index is None:
                    display_row.append("-")  # Use dash for empty arenas
                else:
                    fly = self.fly_data[index]  # Get fly data using the index stored in arenas
                    fly_type_key = (
                        fly['genotype_id'],
                        'Female' if fly['is_female'] else 'Male',
                        tuple(fly.get('attribute_ids', []))  # Handle if attribute_ids key is missing
                    )
                    if fly_type_key not in fly_type_to_number:
                        fly_type_to_number[fly_type_key] = len(fly_type_to_number) + 1
                        attributes = ", ".join(map(str, fly_type_key[2]))
                        legend += f"{fly_type_to_number[fly_type_key]}: Genotype {fly_type_key[0]}, {fly_type_key[1]}, Attributes: [{attributes}]\n"
                    display_row.append(fly_type_to_number[fly_type_key])
            display_arenas.append(display_row)

        # Adding rows to the table
        for row in display_arenas:
            table.add_row(row)

        print(table)
        print(legend)

    def display_fly_details(self, fly_index):
        fly = self.fly_data[fly_index]
        attributes = ', '.join(map(str, fly.get('attribute_ids', [])))  # Ensure attribute IDs are displayed correctly
        details = (
            f"Genotype ID: {fly['genotype_id']}, "
            f"Is Female: {'Yes' if fly['is_female'] else 'No'}, "
            f"Age (days): {fly['age_day_after_eclosion']}, "
            f"Attributes: [{attributes}], "
            f"Participation Count: {fly['participation_count']}"
        )
        print(f"Current configuration for Fly ID {fly_index + 1}:")
        print(details)
    def adjust_fly_numbers(self):
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

    




# Usage would involve creating an instance of FlyManager and using it to manage fly data efficiently.

# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
# fly_manager = FlyManager(db_handler)

# # To start entering stimuli for an experiment:
# assignments = fly_manager.enter_flies_for_experiment()
# save_flies_to_json('./test_flies.json',assignments)

# # Example usage:
flies = load_flies_from_json('./test_flySet.json')
fly_distribution_manager = FlyDistributionManager(flies)
total_arenas = 54  # Example arena count
fly_distribution_manager.enter_flies_for_experiment(total_arenas)
arenas = fly_distribution_manager.distribute_flies(9, 6)  # Example layout with rows and cols
fly_distribution_manager.show_arena_assignments(arenas)