from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json
import os

# Example usage of the setup_experiments method
if __name__ == "__main__":

    
    ###################
    # INPUT VARIABLES #
    ###################

    filenames = ['2024_10_28__09-30-00.mp4','2024_10_29__09-30-00.mp4','2024_11_04__09-15-01.mp4',
                 '2024_11_05__09-20-00.mp4','2024_11_06__09-20-00.mp4','2024_11_08__09-00-00.mp4',
                 '2024_11_08__12-30-00.mp4','2024_11_11__09-05-00.mp4','2024_11_11__12-45-00.mp4',
                 '2024_11_12__11-06-00.mp4','2024_11_12__12-13-00.mp4','2024_11_13__09-15-00.mp4',
                 '2024_11_13__13-22-00.mp4','2024_11_14__09-25-00.mp4','2024_11_15__09-40-00.mp4',
                 '2024_11_15__13-45-00.mp4','2024_11_16__09-45-55.mp4','2024_11_16__13-10-00.mp4',
                 '2024_11_18__09-12-00.mp4','2024_11_18__13-00-11.mp4']

    for mp4_filepath in filenames:

        ##############
        # BASIC CODE # 
        ##############

        folder_name  = mp4_filepath.split('.')[0] # folder where this should be saved
        aoraki_gpu_partition = 'all_aoraki_gpus' # runs on all free gpu resources

        server_path = '/projects/sciences/zoology/geurten_lab/food_experiments/lyall'
        base_output_path = f'{server_path}/analysis_folders/{folder_name}'
        video_file_path = f'{server_path}/new_vids/{mp4_filepath}'
        
        # Make the folder
        os.makedirs(base_output_path, exist_ok=True)

        # Load JSON data from file
        with open('config/path_config_local.json', 'r') as json_file:
            path_config = json.load(json_file)

        print(path_config)

            # Load JSON data from file

        with open('config/path_config_local.json', 'r') as json_file:

            path_config = json.load(json_file)

        print(path_config)

        experiment_setup = ExperimentSetupManager(base_output_path, 
                                                path_config['db_file_path'], 
                                                video_file_path, 
                                                path_config['python_interp'],
                                                path_config['yolo_weights'],
                                                path_config['conda_script_position'],
                                                path_config['conda_env_name'], gpu_partition="aoraki_gpu_L40")

        experiment_setup.setup_experiments()
        experiment_setup.display_experiment_overview()
        experiment_setup.display_experiment_overview_arena_wise()

        if video_file_path:
            experiment_setup.get_video_info()
            experiment_setup.write_meta_data_table()
            go = input("Do you want to run this job? (y/n)")
            if go =='y':
                experiment_setup.run_full_work_flow()
            else:
                print(f'you chose: {go}, therefore we did not start the jobs')
