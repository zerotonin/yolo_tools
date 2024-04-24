from FlyChoiceDatabase import *
from prettytable import PrettyTable
import os 


def clear_screen():
    """Clears the terminal screen for better readability."""
    os.system('cls' if os.name == 'nt' else 'clear')

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
                break
            else:
                print("Invalid input, please type 'yes' or 'no'.")

# Usage would involve creating an instance of FlyManager and using it to manage fly data efficiently.

# Usage example:
db_url = 'sqlite:////home/geuba03p/PyProjects/yolo_tools/fly_choice.db'
db_handler = DatabaseHandler(db_url)
fly_manager = FlyManager(db_handler)

# To start entering stimuli for an experiment:
assignments = fly_manager.enter_flies_for_experiment()
print(assignments)