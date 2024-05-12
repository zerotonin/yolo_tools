import pandas as pd
from yolo_tools.database.FlyChoiceDatabase import *

class ResultManager:
    def __init__(self, db_url,meta_data_csv_path,result_base_path):
        """
        Initialize the ExperimentManager with a database URL and the path to the metadata CSV.

        Args:
            db_url (str): Connection string for the database.
            csv_path (str): Path to the metadata CSV file.
        """
        self.db_handler = DatabaseHandler(db_url)
        self.meta_data_csv_path = meta_data_csv_path
        self.result_base_path = result_base_path
        self.metadata_df = None
        self.experiment_id = None

    def read_metadata(self):
        """
        Read metadata from a CSV file and store it in a DataFrame.
        """
        self.metadata_df = pd.read_csv(self.meta_data_csv_path)
        return self.metadata_df
    
    def update_metadata_file(self):
        """
        Save the updated metadata DataFrame back to the CSV.
        """
        self.metadata_df.to_csv(self.csmeta_data_csv_pathv_path, index=False)


    def insert_experiment(self, data):
        """
        Insert an experiment into the database and update metadata DataFrame with the experiment ID.
        """
        experiment = Experiment(
            date_time               = self.metadata_df.date_time[0],
            fps                     = self.metadata_df.fps[0],
            video_file_path         = self.metadata_df.video_file_path.fps[0],
            experiment_type         = self.metadata_df.experiment_type[0],
            experimenter_id         = self.metadata_df.experimenter_id[0],
            number_of_arenas        = self.metadata_df.number_of_arenas[0],
            number_of_arena_rows    = self.metadata_df.number_of_arena_rows[0],
            number_of_arena_columns = self.metadata_df.number_of_arena_columns[0])
        with self.db_handler as db:
            db.add_record(experiment)
        self.experiment_id = experiment.id

    def process_metadata(self):
        """
        Process each row in the metadata DataFrame to create experiments, flies, and trials.
        """
        for idx, row in self.metadata_df.iterrows():
            exp_data = {
                'date_time': row['date_time'],
                'fps': row['fps'],
                'video_file_path': row['video_file_path']
            }
            exp_id = self.insert_experiment(exp_data)
            # Add further processing steps here for flies, trials, etc.
            self.metadata_df.loc[idx, 'experiment_id'] = exp_id

# Example usage
db_url = 'sqlite:///your_database.db'
csv_path = 'path_to_metadata.csv'
exp_manager = ExperimentManager(db_url, csv_path)
exp_manager.read_metadata()
exp_manager.process_metadata()
exp_manager.update_metadata_file()
