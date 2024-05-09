from database.FlyChoiceDatabase import *
from prettytable import PrettyTable
import os 

class ArenaAttributeManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler


    def _clear_screen(self):
        """
        Clears the terminal screen.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def select_or_create_attribute(self):
        with self.db_handler as db:
            self._clear_screen()
            attributes = db.get_records(ArenaAttribute)
            print("\nExisting Arena Attributes:")
            for attr in attributes:
                print(f"ID: {attr.id}, Name: {attr.name}")

            choice = input("Enter attribute ID to select or 'new' to create a new attribute: ")
            if choice.lower() == 'new':
                return self.create_new_attribute()
            else:
                attribute = db.get_records(ArenaAttribute, filters={'id': int(choice)})
                return attribute[0] if attribute else None

    def create_new_attribute(self):
        name = input("Enter the name of the new attribute: ")
        new_attribute = ArenaAttribute(name=name)
        with self.db_handler as db:
            db.add_record(new_attribute)
        return new_attribute

class ArenaManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler


    def _clear_screen(self):
        """
        Clears the terminal screen.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_arenas(self):
        self._clear_screen()
        with self.db_handler as db:
            arenas = db.get_records(Arena)
            table = PrettyTable()
            table.field_names = ["ID", "Name", "Width (mm)", "Height (mm)", "Radius (mm)", "Attributes"]
            for arena in arenas:
                attributes = ', '.join([attr.name for attr in arena.attributes])
                table.add_row([arena.id, arena.name, arena.size_width_mm, arena.size_height_mm, arena.size_radius_mm, attributes])
            print(table)

    def enter_new_arena(self):
        self.show_arenas()
        name = input("Enter the name of the new arena: ")
        width = float(input("Enter the width of the arena in mm: "))
        height = float(input("Enter the height of the arena in mm: "))
        radius = float(input("Enter the radius of the arena in mm (if applicable), else enter 0: "))

        attribute_manager = ArenaAttributeManager(self.db_handler)
        attributes = []
        for _ in range(5):  # Allows entry for up to 5 attributes
            attribute = attribute_manager.select_or_create_attribute()
            if attribute:
                attributes.append(attribute)
            if input("Add more? (y/n): ").lower() != 'y' or not attribute:
                break

        new_arena = Arena(name=name, size_width_mm=width, size_height_mm=height, size_radius_mm=radius if radius > 0 else None, attributes=attributes)
        with self.db_handler as db:
            db.add_record(new_arena)
        print("New arena added successfully.")

    def enter_arena_list_for_experiment(self, number_of_arenas):
        arenas_list = []
        print("Select arenas for the experiment:")
        while len(arenas_list) < number_of_arenas:
            self.show_arenas()
            if arenas_list:
                print("Currently selected Arena IDs:", ', '.join(str(arena) for arena in arenas_list))

            arena_id = input("Enter arena ID or 'new' to add a new arena: ")
            if arena_id.lower() == 'new':
                self.enter_new_arena()
                continue
            else:
                try:
                    arena_id = int(arena_id)
                    with self.db_handler as db:
                        arena = db.get_records(Arena, filters={'id': arena_id})
                        if arena:
                            arenas_list.append(arena[0].id)
                            print(f"Added Arena ID {arena[0].id}.")
                        else:
                            print("Arena not found.")
                except ValueError:
                    print("Please enter a valid numeric ID.")
            if input("Add more? (y/n): ").lower() != 'y':
                break
        return arenas_list

    def pattern_uniform(self, number_of_arenas):
        """
        Assigns a uniform pattern of arenas across all experiments.
        """
        arenas_list = self.enter_arena_list_for_experiment(1)  # Get one arena and replicate it
        if arenas_list:
            return [arenas_list[0] for _ in range(number_of_arenas)]  # Replicate the selected arena uniformly
        return []

    def pattern_individual(self, number_of_arenas):
        """
        Each arena can be individually customized.
        """
        return self.enter_arena_list_for_experiment(number_of_arenas)

    def enter_arenas_for_experiment(self, number_of_arenas,rows,cols):
        """
        Main method to start the arena assignment for an experiment.
        """
        print("Choose a pattern (uniform, individual):")
        pattern = input().lower()
        pattern_functions = {
            'uniform': self.pattern_uniform,
            'individual': self.pattern_individual
        }

        if pattern in pattern_functions:
            assignments = pattern_functions[pattern](number_of_arenas)
        else:
            print("Pattern not recognized.")
            return []
        
        self.display_arenas_grid(assignments,rows,cols)
        return assignments
        
    def display_arenas_grid(self, arena_ids, rows, cols):
        """
        Displays the arena IDs in a grid format based on the specified number of rows and columns.
        
        Args:
            arena_ids (list): List of arena IDs.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
        """
        if len(arena_ids) != rows * cols:
            raise ValueError("The number of arena IDs does not match the product of rows and columns.")

        table = PrettyTable()
        table.field_names = [f"Column {i+1}" for i in range(cols)]

        index = 0
        for _ in range(rows):
            row_entries = []
            for _ in range(cols):
                if index < len(arena_ids):
                    row_entries.append(arena_ids[index])
                    index += 1
                else:
                    row_entries.append("None")  # In case there are fewer IDs than slots
            table.add_row(row_entries)
        self._clear_screen()
        print(table)
        _=input("Press Enter to continue...")

    def get_human_readable_arena_details(self, arena_id):
        """
        Returns a human-readable string of the arena's details, including its name,
        dimensions, and associated attributes.

        Args:
            arena_id (int): The ID of the arena to retrieve details for.

        Returns:
            str: Human-readable string of the arena's details, or an error message if not found.
        """
        with self.db_handler as db:
            # Fetch the arena information
            arena = db.get_records(Arena, {'id': arena_id})
            if not arena:
                return "Arena not found."
            arena = arena[0]

            # Collect attributes names
            attributes = ', '.join([attr.name for attr in arena.attributes])

            # Compile the details into a human-readable string
            dimensions = f"Width: {arena.size_width_mm} mm, Height: {arena.size_height_mm} mm"
            if arena.size_radius_mm:
                dimensions += f", Radius: {arena.size_radius_mm} mm"
            details = f"Name: {arena.name}, {dimensions}, Attributes: [{attributes}]"
            return details


