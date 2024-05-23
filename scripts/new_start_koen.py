from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json

# Example usage of the setup_experiments method
if __name__ == "__main__":

    ###################
    # INPUT VARIABLES #
    ###################

    folder_name  = '2024_05_22__12-13-12' # folder where this should be saved
    mp4_filepath = '2024_05_22__12-13-12_first_collumn_female_1p_agarose_Left_1p_agarose_10_6p_fructose_Right.mp4' # file name of the video file  you want to analyse
    aoraki_gpu_partition = 'aoraki_gpu' # 'aoraki_gpu' 'aoraki_gpu_L40' 'aoraki_gpu_H100'

    ##############
    # BASIC CODE # 
    ##############

    server_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen'
    base_output_path = f'{server_path}/analysis_folders/{folder_name}'
    video_file_path = f'{server_path}/new_vids/{mp4_filepath}'
    

    # Load JSON data from file
    with open('config/path_config_local.json', 'r') as json_file:
        path_config = json.load(json_file)

    print(path_config)

    experiment_setup = ExperimentSetupManager(base_output_path, path_config['db_file_path'], 
                                              video_file_path, path_config['python_interp'],
                                              path_config['yolo_weights'],
					      gpu_partition=aoraki_gpu_partition)
    experiment_setup.setup_experiments()
    experiment_setup.display_experiment_overview()
    experiment_setup.display_experiment_overview_arena_wise()
    if video_file_path:
        experiment_setup.get_video_info()
        experiment_setup.write_meta_data_table()
        go = input("Do you want to run this job? (y/n)")
        if go =='y':
            experiment_setup.run_slurm_jobs()
        else:
            print(f'you chose: {go}, therefore we did not start the jobs')
