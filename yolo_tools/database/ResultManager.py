import os
import json
import math
import argparse
import datetime
import numpy as np
import pandas as pd
from yolo_tools.database.FlyChoiceDatabase import *
from yolo_tools.analysis_file_manager.AnalysisFileManager import AnalysisFileManager
from tqdm import tqdm
import warnings

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
        self.file_manager = AnalysisFileManager()
        self.file_manager.path_dict = dict()
        self.file_manager.path_dict['choice_analysis'] = os.path.join(result_base_path,'choice_analysis')
        self.file_manager.path_dict['trajectories'] = os.path.join(result_base_path,'trajectories')
        self.metadata_df = None
        self.experiment_id = None

    @staticmethod
    def parse_integer(value):
        """
        Parses the fly attribute value, converting NaN to None and ensuring the value is an integer.

        Args:
            value: The value to parse.

        Returns:
            The parsed value or None if the value is NaN.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    
    @staticmethod
    def parse_float_point_num(value):
        """
        Parses the fly attribute value, converting NaN to None and ensuring the value is an integer.

        Args:
            value: The value to parse.

        Returns:
            The parsed value or None if the value is NaN.
        """
        if value is None or (isinstance(value, int) and math.isnan(value)):
            return None
        return float(value)

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
        self.metadata_df.to_csv(self.meta_data_csv_path, index=False)
    
    def check_files_loadable(self):
        """
        Check if all files referenced in the metadata are loadable. If any files are not loadable,
        print the list of unloadable files and raise an error.
        """
        unloadable_files = []
        unloadable_indices = []

        for idx, row in self.metadata_df.iterrows():
            try:
                decision_base_path = self.file_manager.create_decision_result_base_path(row['arena_number'])
                json_file_path = self.file_manager.create_result_filepath(decision_base_path, 'choice_json')
                four_field_file_path = self.file_manager.create_result_filepath(decision_base_path, 'decision_four_field_matrix')
                duration_file_path = self.file_manager.create_result_filepath(decision_base_path, 'decision_duration_matrix')

                with open(json_file_path, 'r') as json_file:
                    json.load(json_file)
                np.load(four_field_file_path)
                np.load(duration_file_path)

                locomotor_base_path = self.file_manager.create_locomotor_result_base_path(row['arena_number'])
                json_file_path = self.file_manager.create_result_filepath(locomotor_base_path, 'locomotor_json')
                trajectory_file_path = self.file_manager.create_result_filepath(locomotor_base_path, 'tra_mm')
                time_decision_path = self.file_manager.create_result_filepath(decision_base_path, 'time_decision_record')

                with open(json_file_path, 'r') as json_file:
                    json.load(json_file)
                np.load(trajectory_file_path)
                np.load(time_decision_path)

            except Exception as e:
                unloadable_files.append((row['arena_number'], str(e)))
                unloadable_indices.append(idx)

        if unloadable_files:
            for file_info in unloadable_files:
                print(f"Unloadable file for arena {file_info[0]}: {file_info[1]}")
            warnings.warn("Some files could not be loaded. Please check the unloadable files above.")
        return unloadable_indices


    def insert_experiment(self):
        """
        Insert an experiment into the database and update metadata DataFrame with the experiment ID.
        """
        # Parse the datetime string into a datetime object
        date_time_str = self.metadata_df['date_time'][0]
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        experiment = Experiment(
            date_time               = date_time_obj,
            fps                     = self.parse_float_point_num(self.metadata_df.fps[0]),
            video_file_path         = str(self.metadata_df.video_file_path[0]),
            experiment_type         = self.parse_integer(self.metadata_df.experiment_type[0]),
            experimenter_id         = self.parse_integer(self.metadata_df.experimenter_id[0]),
            number_of_arenas        = self.parse_integer(self.metadata_df.number_of_arenas[0]),
            number_of_arena_rows    = self.parse_integer(self.metadata_df.number_of_arena_rows[0]),
            number_of_arena_columns = self.parse_integer(self.metadata_df.number_of_arena_columns[0]))
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
            is_female              = row['is_female'],
            genotype_id            = str(row['genotype_id']),
            age_day_after_eclosion = self.parse_float_point_num(row['age_day_after_eclosion']),
            fly_attribute_1        = self.parse_integer(row.get('fly_attribute_1')),
            fly_attribute_2        = self.parse_integer(row.get('fly_attribute_2')),
            fly_attribute_3        = self.parse_integer(row.get('fly_attribute_3')),
            fly_attribute_4        = self.parse_integer(row.get('fly_attribute_4')),
            fly_attribute_5        = self.parse_integer(row.get('fly_attribute_5'))
        )
        with self.db_handler as db:
            db.add_record(new_fly)
            db.session.flush()  # Ensure ID is assigned
            self.metadata_df.loc[row.name, 'fly_id'] = new_fly.id
            row['fly_id'] = new_fly.id # need for updating inside the for loop of process meta_data

    def check_and_if_needed_insert_fly(self,idx,row):
        
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
            row['fly_id'] = fly_id # need for updating inside the for loop of process meta_data
    
    def insert_trial(self,idx,row):

        new_trial = Trial(arena_number= self.parse_integer(row['arena_number']),
                        experiment_id = self.parse_integer(row['experiment_id']),
                        fly_id        = self.parse_integer(row['fly_id']),
                        arena_id      = self.parse_integer(row['arena_id']),
                        stimuli_01    = self.parse_integer(row['stimuli_01']),
                        stimuli_02    = self.parse_integer(row['stimuli_02']),
                        stimuli_03    = self.parse_integer(row['stimuli_03']),
                        stimuli_04    = self.parse_integer(row['stimuli_04']),
                        stimuli_05    = self.parse_integer(row['stimuli_05']),
                        stimuli_06    = self.parse_integer(row['stimuli_06']),
                        stimuli_07    = self.parse_integer(row['stimuli_07']),
                        stimuli_08    = self.parse_integer(row['stimuli_08']),
                        stimuli_09    = self.parse_integer(row['stimuli_09']),
                        stimuli_10    = self.parse_integer(row['stimuli_10']))
        
        with self.db_handler as db:
            db.add_record(new_trial)
            db.session.flush()  # Ensure ID is assigned
            row['trial_id'] = new_trial.id # need for updating inside the for loop of process meta_data
            self.metadata_df.loc[idx, 'trial_id'] = new_trial.id

    def insert_decision_result(self,row):
        decision_base_path = self.file_manager.create_decision_result_base_path(row['arena_number'])
        json_file_path = self.file_manager.create_result_filepath(decision_base_path,'choice_json')
        four_field_file_path = self.file_manager.create_result_filepath(decision_base_path,'decision_four_field_matrix')
        duration_file_path = self.file_manager.create_result_filepath(decision_base_path,'decision_duration_matrix')
     
        # Load decision data from JSON file
        with open(json_file_path, 'r') as json_file:
            decision_results = json.load(json_file)

        # Load NumPy arrays for the four-field and duration matrices
        four_field_matrix = np.load(four_field_file_path)
        decision_duration_matrix = np.load(duration_file_path)

        new_decision_entry = TwoChoiceDecision(trial_id                           = self.parse_integer(row['trial_id']),
                                               fraction_left                      = self.parse_float_point_num(decision_results['fraction_left']),
                                               fraction_right                     = self.parse_float_point_num(decision_results['fraction_right']), 
                                               fraction_middle                    = self.parse_float_point_num(decision_results['fraction_middle']), 
                                               fraction_positive                  = self.parse_float_point_num(decision_results['fraction_positive']), 
                                               fraction_negative                  = self.parse_float_point_num(decision_results['fraction_negative']), 
                                               preference_index                   = self.parse_float_point_num(decision_results['preference_index']), 
                                               decision_duration_index            = self.parse_float_point_num(decision_results['decision_duration_index']), 
                                               decision_to_positive_num           = self.parse_float_point_num(four_field_matrix[0,0]),
                                               decision_from_positive_num         = self.parse_float_point_num(four_field_matrix[1,0]),
                                               decision_to_negative_num           = self.parse_float_point_num(four_field_matrix[0,1]),
                                               decision_from_negative_num         = self.parse_float_point_num(four_field_matrix[1,1]),
                                               duration_after_positive            = self.parse_float_point_num(decision_duration_matrix[0,0]),
                                               duration_after_negative            = self.parse_float_point_num(decision_duration_matrix[0,1]),
                                               time_of_first_decision_elapsed_sec = self.parse_float_point_num(decision_results['time_of_first_decision_elapsed_sec']))
        

        with self.db_handler as db:
            db.add_record(new_decision_entry)
            db.session.flush()  # Ensure ID is assigned

    def insert_locomotor_result(self,row):
        locomotor_base_path = self.file_manager.create_locomotor_result_base_path(row['arena_number'])
        json_file_path = self.file_manager.create_result_filepath(locomotor_base_path,'locomotor_json')
     
        # Load decision data from JSON file
        with open(json_file_path, 'r') as json_file:
            locomotor_results = json.load(json_file)
        
        new_locomotor_entry = Locomotor(trial_id           = self.parse_integer(row['trial_id']),
                                        distance_walked_mm = self.parse_float_point_num(locomotor_results['distance_walked']),
                                        max_speed_mmPs     = self.parse_float_point_num(locomotor_results['max_speed']),
                                        avg_speed_mmPs     = self.parse_float_point_num(locomotor_results['avg_speed']))
        with self.db_handler as db:
            db.add_record(new_locomotor_entry)
            db.session.flush()  # Ensure ID is assigned



    # def insert_trajectory_data(self,row):
    #     locomotor_base_path = self.file_manager.create_locomotor_result_base_path(row['arena_number'])
    #     trajectory_file_path = self.file_manager.create_result_filepath(locomotor_base_path,'tra_mm')
    #     trajectory_mm = np.load(trajectory_file_path)

    #     with self.db_handler as db:
    #         for frame_i in range(trajectory_mm.shape[0]):
    #             new_entry = Trajectories(trial_id               = self.parse_integer(row['trial_id']),
    #                                     pos_x_mm_arena_centered = self.parse_float_point_num(trajectory_mm[frame_i,0]),
    #                                     pos_y_mm_arena_centered = self.parse_float_point_num(trajectory_mm[frame_i,1]))
    #             db.add_record(new_entry)
    #         db.session.flush()  # Ensure ID is assigned
    def insert_trajectory_data(self, row):
        locomotor_base_path = self.file_manager.create_locomotor_result_base_path(row['arena_number'])
        trajectory_file_path = self.file_manager.create_result_filepath(locomotor_base_path, 'tra_mm')
        trajectory_mm = np.load(trajectory_file_path)
        
        entries = []
        for frame_i in range(trajectory_mm.shape[0]):
            new_entry = Trajectories(
                trial_id=self.parse_integer(row['trial_id']),
                pos_x_mm_arena_centered=self.parse_float_point_num(trajectory_mm[frame_i, 0]),
                pos_y_mm_arena_centered=self.parse_float_point_num(trajectory_mm[frame_i, 1])
            )
            entries.append(new_entry)
        
        with self.db_handler as db:
            db.session.bulk_save_objects(entries)
            db.session.flush()  # Ensure ID is assigned
            db.session.commit()

    def insert_decision_timing(self,row):
        # get the decision time matrix
        decision_base_path = self.file_manager.create_decision_result_base_path(row['arena_number'])
        time_decision_path = self.file_manager.create_result_filepath(decision_base_path,'time_decision_record')
        time_decision = np.load(time_decision_path)

        #get the decision type keys
        with self.db_handler as db:
            records = db.get_all_two_choice_decision_types()
            # Convert to a dictionary with identifier as key and id as value
            decision_types_dict = {record.identifier: record.id for record in records}
        
        
        #make entry objects
        entries = []
        for dec_i in range(time_decision.shape[0]):
            new_entry = TwoChoiceDecisionTiming(
                trial_id=self.parse_integer(row['trial_id']),
                time_sec=self.parse_float_point_num(time_decision[dec_i, 0]),
                decision_type_id=self.parse_integer(decision_types_dict[time_decision[dec_i, 1]])
            )
            entries.append(new_entry)

        #bulk entry
        with self.db_handler as db:
            db.session.bulk_save_objects(entries)
            db.session.flush()  # Ensure ID is assigned
            db.session.commit()

    def process_metadata(self,unloadable_indices):
        """
        Process each row in the metadata DataFrame to create experiments, flies, and trials.
        Assumes that `insert_experiment` has already been called and set `self.experiment_id`.
        """
        self.insert_experiment() 
         
        for idx, row in tqdm(self.metadata_df.iterrows(),desc='writing flies into db'):

            if idx not in unloadable_indices:

                self.check_and_if_needed_insert_fly(idx,row)
                self.insert_trial(idx,row)
                self.insert_decision_result(row)
                self.insert_locomotor_result(row)
                self.insert_trajectory_data(row)
                self.insert_decision_timing(row)
    
        self.update_metadata_file()

def main():
    parser = argparse.ArgumentParser(description='Manage results and insert them into a database.')
    parser.add_argument('--db_url', type=str, required=True, help='Database connection URL.')
    parser.add_argument('--meta_data_csv_path', type=str, required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--result_base_path', type=str, required=True, help='Base path for where result files are stored.')

    args = parser.parse_args()

    # Initialize the ResultManager with parsed arguments
    result_manager = ResultManager(
        db_url=args.db_url,
        meta_data_csv_path=args.meta_data_csv_path,
        result_base_path=args.result_base_path
    )

    # Read metadata to ensure everything is loaded correctly
    result_manager.read_metadata()

    # Check if all files are accesible
    unloadable_indices = result_manager.check_files_loadable()

    # Process metadata which involves inserting data into the database
    result_manager.process_metadata(unloadable_indices)

    # Optionally, print a message or handle further tasks
    print("Data processing complete.")

if __name__ == '__main__':
    main()
