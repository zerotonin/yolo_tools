from yolo_tools.workflow.ExperimentSetupManager import ExperimentSetupManager
import json

# Example usage of the setup_experiments method
if __name__ == "__main__":
    # base_output_path = '/projects/sciences/zoology/geurten_lab/food_experiments/koen/analysis_folders/column1_male_agarose1psalt18.8p_Left_agarose1pright'
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
        experiment_setup.write_meta_data_table()
        experiment_setup.run_slurm_jobs()

"""
inSQL           1pAgaroseR_1pAgarose1pSaltL_MaleL                                       2024_04_04__15-09-21_1pagaroseR_1pagarose1pSaltL_MaleL.mp4
running         column1Female_Agarose1p_Left_Agarose1p_Fructose18_15p_Right             2024_04_25__12-57-11_column1Female_Agarose1Pleft_Agarose1pFructose18-15pRigth.mp4
failed on ana06 FirstColumnMale_Agarose1pLeft_Agarose1p_Fructose26.6P_Salt18.8P_Right   2024_04_15__11-31-42_Firstcolumnmale_Agarose1pLeft_agarose1pFructose26-6Psalt18-8PRight.mp4
inSQL           Column1Male_Agarose1p_Left_Agarose1p_Fructose26.2p_Right                2024_04_10__14-41-42_FirstcolumnMale_agarose1pLeft_agarose1pfructose26-2pRight.mp4
inSQL           Column1_male_Agar1p_Sugar26.3p_Left_Agar1p_Right                        2024_04_08__17-40-26column1male_agar1psugar26-3pLeftagar1pright.mp4
inSQL           2024_04_05__17-18-49                                                    2024_04_05__17-18-49_column1male_agarose1psalt18-8PLeftagarose1Pright.mp4    




"""
