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


    def insert_experiment(self):
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
        self.metadata_df.experiment_id = experiment.id

    def insert_fly(self, row):
        """
        Insert a new fly into the database if it doesn't already exist.

        Args:
            row (pd.Series): A DataFrame row containing the fly's attributes.
        """
        new_fly = Fly(
            is_female=row['is_female'],
            genotype_id=row['genotype_id'],
            age_day_after_eclosion=row['age_day_after_eclosion'],
            fly_attribute_1=row.get('fly_attribute_1'),
            fly_attribute_2=row.get('fly_attribute_2'),
            fly_attribute_3=row.get('fly_attribute_3'),
            fly_attribute_4=row.get('fly_attribute_4'),
            fly_attribute_5=row.get('fly_attribute_5')
        )
        with self.db_handler as db:
            db.add_record(new_fly)
            db.session.flush()  # Ensure ID is assigned
        self.metadata_df.loc[row.name, 'fly_id'] = new_fly.id

    def check_and_if_needed_insert_fly(self,row):
        
        attribute_ids = [row.get(f'fly_attribute_{i}') for i in range(1, 6) if pd.notna(row.get(f'fly_attribute_{i}'))]
        with self.db_handler as db:
            fly_id = self.db_handler.find_fly_by_attributes(
                attribute_ids,
                row['genotype_id'],
                row['is_female'],
                row['age_day_after_eclosion']
            )
        if fly_id is None:
            self.insert_fly(row)
        else:
            self.metadata_df.loc[idx, 'fly_id'] = fly_id
    
    def insert_trial(self,row):

        new_trial = Trial(arena_number= row['arena_number'],
                        experiment_id = row['experiment_id'],
                        fly_id        = row['fly_id'],
                        arena_id      = row['arena_id'],
                        stimuli_01    = row['stimuli_01'],
                        stimuli_02    = row['stimuli_02'],
                        stimuli_03    = row['stimuli_03'],
                        stimuli_04    = row['stimuli_04'],
                        stimuli_05    = row['stimuli_05'],
                        stimuli_06    = row['stimuli_06'],
                        stimuli_07    = row['stimuli_07'],
                        stimuli_08    = row['stimuli_08'],
                        stimuli_09    = row['stimuli_09'],
                        stimuli_10    = row['stimuli_10'])
        
        with self.db_handler as db:
            db.add_record(new_trial)
            db.session.flush()  # Ensure ID is assigned
        self.metadata_df.loc[row.name, 'trial_id'] = new_trial.id

    def process_metadata(self):
        """
        Process each row in the metadata DataFrame to create experiments, flies, and trials.
        Assumes that `insert_experiment` has already been called and set `self.experiment_id`.
        """
        self.insert_experiment()  
        for idx, row in self.metadata_df.iterrows():
            self.check_and_if_needed_insert_fly(row)
            self.update_metadata_file()


