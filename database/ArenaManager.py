from FlyChoiceDatabase import *
from prettytable import PrettyTable
import os 

class ArenaAttributeManager:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def select_or_create_attribute(self):
        with self.db_handler as db:
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

    def show_arenas(self):
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
            return [arenas_list for _ in range(number_of_arenas)]  # Replicate the selected arena uniformly
        return []

    def pattern_individual(self, number_of_arenas):
        """
        Each arena can be individually customized.
        """
        return self.enter_arena_list_for_experiment(number_of_arenas)

    def enter_arenas_for_experiment(self, number_of_arenas):
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
            return pattern_functions[pattern](number_of_arenas)
        else:
            print("Pattern not recognized.")
            return []




# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
stimulus_manager = ArenaManager(db_handler)

# To start entering stimuli for an experiment:
assignments = stimulus_manager.enter_arenas_for_experiment(54)
print(assignments)