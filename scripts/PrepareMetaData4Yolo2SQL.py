from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json

# Example usage of the setup_experiments method
if __name__ == "__main__":
    # base_output_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/analysis_folders/template_01'
    # video_file_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/new_vids'
    base_output_path = '/home/geuba03p/food_example_video/output'
    video_file_path = '/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'

    # Load JSON data from file
    with open('config/path_config_local.json', 'r') as json_file:
        path_config = json.load(json_file)

    experiment_setup = ExperimentSetupManager(base_output_path, path_config['db_file_path'], 
                                              video_file_path, path_config['python_interp'],
                                              path_config['yolo_weights'])
    experiment_setup.setup_experiments()
    experiment_setup.display_experiment_overview()
    experiment_setup.display_experiment_overview_arena_wise()
    if video_file_path:
        experiment_setup.get_video_info()
    
        if not experiment_setup.check_loadable_preset('slurm_video_scripts'):
            experiment_setup.make_video_splitting_slurm_script()
