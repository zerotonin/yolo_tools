import tkinter as tk
from tkinter import filedialog
import pandas as pd
import json

class PresetManager:
    def __init__(self):
        """
        Initializes the CLIFileManager with a hidden root Tkinter window.
        """
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the Tkinter root window

    def query_load_presets(self, topic="presets"):
        """
        Asks the user if they want to load presets for a specified topic, handling the response via CLI.

        Args:
            topic (str): The topic for which presets might be loaded (e.g., "experiment settings").

        Returns:
            bool: True if the user confirms the action, False otherwise.
        """
        response = input(f"Do you want to load {topic}? (yes/no): ").strip().lower()
        return response == 'yes'

    def open_file_dialog(self, file_extensions):
        """
        Opens a file dialog using Tkinter, allowing the user to select a file of specified types.

        Args:
            file_extensions (tuple): Tuple of file types that can be selected, e.g., 
                                     (("JSON files", "*.json"), ("CSV files", "*.csv"))

        Returns:
            str: The file path to the selected file.
        """
        file_types = [(f"{ext} files", f"*{ext}") for ext in file_extensions]
        filename = filedialog.askopenfilename(title="Select a file", filetypes=file_types)
        return filename


    def load_csv(self, filepath):
        """
        Loads data from a CSV file using pandas for enhanced performance and ease of use.

        Args:
            filepath (str): Path to the CSV file to be loaded.

        Returns:
            pandas.DataFrame: A DataFrame object containing the loaded data.
        """
        df = pd.read_csv(filepath)
        return df


    def load_json(self, filepath):
        """
        Loads data from a JSON file.

        Args:
            filepath (str): Path to the JSON file to be loaded.

        Returns:
            dict or list: The JSON content loaded from the file.
        """
        with open(filepath, 'r') as jsonfile:
            data = json.load(jsonfile)
        return data

    def close(self):
        """
        Closes the Tkinter application.
        """
        self.root.destroy()

    def load_dialogue(self,topic,file_extensions):
        if self.query_load_presets(topic):
            file_path = self.open_file_dialog(file_extensions)
            if file_path:
                if file_path.endswith('.json'):
                    data = self.load_json(file_path)
                elif file_path.endswith('.csv'):
                    data = self.load_csv(file_path)
                return data
            else:
                print(f"No file selected for {topic} preset.")
                return None

