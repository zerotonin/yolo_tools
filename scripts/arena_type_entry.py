from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json

# Example usage of the setup_experiments method
if __name__ == "__main__":
    base_output_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/analysis_folders/2024_06_13__09-27-22_first_collumn_female_1p_agarose_15_1p_fructose_L_1p_agarose_22_9p_fructose_R'
    video_file_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/new_vids/2024_06_13__09-27-22_first_collumn_female_1p_agarose_15_1p_fructose_L_1p_agarose_22_9p_fructose_R.mp4'
    

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
    experiment_setup.arena_manager.enter_arenas_for_experiment()
