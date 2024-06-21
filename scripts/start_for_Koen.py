from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json

# Example usage of the setup_experiments method
if __name__ == "__main__":
    base_output_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/analysis_folders/2024_05_29__11-59-19_first_collumn_female_1p_agarose_L_1p_agarose_3p_salt_R/'
    video_file_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/new_vids/2024_05_29__11-59-19_first_collumn_female_1p_agarose_L_1p_agarose_3p_salt_R.mp4'
    

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
                                              path_config['conda_env_name'],
					                          gpu_partition="aoraki_gpu_L40")
    experiment_setup.setup_experiments()
    experiment_setup.display_experiment_overview()
    experiment_setup.display_experiment_overview_arena_wise()
    if video_file_path:
        experiment_setup.get_video_info()
        experiment_setup.write_meta_data_table()
        experiment_setup.run_full_work_flow()
