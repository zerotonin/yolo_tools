import subprocess
import os

class SlurmJobManager:
    def __init__(self, analysis_file_manager,arena_num,meta_data_table,gpu_partition = 'aoraki_GPU'):
        self.file_manager = analysis_file_manager
        self.file_base_dir =  self.file_manager.path_dict['output_file_path']
        self.user_name = os.getlogin()
        self.python_path =  self.file_manager.file_dict['python_interpreter']
        self.arena_num = arena_num
        self.meta_data_table = meta_data_table
        self.gpu_partion = gpu_partition

    def submit_job(self, script_path, dependency_id=None):
        """
        Submits a job to the SLURM scheduler with an optional dependency.
        """
        cmd = ['sbatch']
        if dependency_id:
            cmd.append(f'--dependency=afterok:{dependency_id}')
        cmd.append(script_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f'Job {job_id} submitted.')
            return job_id
        else:
            raise Exception(f"Failed to submit job: {result.stderr}")



    def create_slurm_script(self, script_parameters):
        """
        Generates a SLURM script file for a given analysis task using parameters from a dictionary.

        Args:
            script_parameters (dict): Parameters for the SLURM script including:
                - partition (str): The partition where the job should run.
                - filename (str): The filename for the SLURM script.
                - python_script (str): Path to the Python script to execute.
                - jobname (str): Name of the job.
                - memory (str): Memory allocation for the job.
                - script_variables (str): String with variables to pass to the Python script.
                - gpus_per_task (int): Number of GPUs per task.
                - nodes (int): Number of nodes to use.
                - ntasks_per_node (int): Number of tasks per node.
        """
        
        # Build the command line for the Python script with additional variables
        python_command = f"{self.python_path} -m yolo_tools.{script_parameters['module']}.{script_parameters['python_script']} {script_parameters['script_variables']}"
        
        # Construct the SLURM script content
        content =  f'#!/bin/bash\n'
        content += f'#SBATCH --job-name={script_parameters['jobname']}\n'
        content += f'#SBATCH --account={self.user_name}\n'
        content += f'#SBATCH --partition={script_parameters['partition']}\n'
        content += f'#SBATCH --cpus-per-task={script_parameters['cpus_per_task']}\n'
        if script_parameters['partition'] != 'aoraki':
            content += f'#SBATCH --gpus-per-task={script_parameters['gpus_per_task']}\n'

        content += f'#SBATCH --nodes={script_parameters['nodes']}\n'
        content += f'#SBATCH --mem={script_parameters['memory']}G\n'
        content += f'#SBATCH --ntasks-per-node={script_parameters['ntasks_per_node']}\n'
        content += f'#SBATCH --output={self.file_base_dir}/slurm_logs/%x.out\n'
        content += f'#SBATCH --error={self.file_base_dir}/slurm_logs/%x.err\n'
        content += f'\n'
        content += f'{python_command}'

        # Write the SLURM script to a file
        with open(script_parameters['filename'], 'w') as f:
            f.write(content)

    def create_tracking_slurm_script(self,split_video_fileposition,arena_num,gpus_per_task =1, memory_GB_int = 32, nodes = 1, cpus_per_task = 1, ntasks = 1):
        

        script_variables = f'--video_path {split_video_fileposition}  --apriori_classes 0 1 --apriori_class_names arena fly --yolo_weights {self.file_manager.file_dict['yolo_weights']} --output_file {self.file_manager.path_dict['trajectories']}/trajectory_arena_{str(arena_num).zfill(2)}.npy'
        script_parameters = dict()
        script_parameters['partition'] =  self.gpu_partion
        script_parameters['gpus_per_task'] = gpus_per_task
        script_parameters['filename'] = f'{self.file_base_dir}/slurm_scripts/track_arena_{str(arena_num).zfill(2)}.sh'
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'videoAnalyser'
        script_parameters['jobname'] =  f'track arena {arena_num}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'detection'

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']


    def create_trajectory_analysis_slurm_script(self, arena_num, positive_stimulus_on_left,
                                                filter_trajectory=True,midline_tolerance = 0.1,memory_GB_int = 64, 
                                                nodes = 1, cpus_per_task = 1, ntasks = 1):

        input_file_position =  f'{self.file_manager.path_dict['trajectories']}/trajectory_arena_{str(arena_num).zfill(2)}.npy'
        output_locomotion_file =  f'{self.file_manager.path_dict['trajectories']}/locomotor_results_arena_{str(arena_num).zfill(2)}_'
        output_decision_file = f'{self.file_manager.path_dict['choice_analysis']}/choice_results_arena_{str(arena_num).zfill(2)}_'
        
        script_variables = f'--input_file {input_file_position} --midline_tolerance {midline_tolerance} --positive_stimulus_on_left {positive_stimulus_on_left} --filter_trajectory {filter_trajectory} --output_locomotion_file {output_locomotion_file} --output_decision_file {output_decision_file}'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = f'{self.file_base_dir}/slurm_scripts/analyse_arena_{str(arena_num).zfill(2)}.sh'
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'trajectoryAnalyser'
        script_parameters['jobname'] =  f'analyse arena {arena_num}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'trajectory_analysis'

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']
    def create_video_splitting_slurm_script(self,memory_GB_int = 64, nodes = 1, cpus_per_task = 1, ntasks = 1):
        
        script_variables = f'--video_path { self.file_manager.file_dict['video_file_position']} --output_folder {self.file_base_dir}/preprocessed_single_videos --output_type videos'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = f'{self.file_base_dir}/slurm_scripts/split_video.sh'
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'FrameSplitter'
        script_parameters['jobname'] =  f'split_{os.path.basename(self.file_manager.file_dict['video_file_position'])}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'video_preprocessing'

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']

    def anticipate_split_video_position(self,arena_i):
        video_base_name = os.path.basename(self.file_manager.file_dict['video_file_position'])
        filename,extension = video_base_name.split('.')
        splitname = f'{filename}__{str(arena_i).zfill(2)}.{extension}'
        return os.path.join(self.file_manager.path_dict['preprocessed_single_videos'],splitname)


    def manage_workflow(self, num_splits):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: Create and submit the split job
        split_script_position = self.create_video_splitting_slurm_script()
        #split_job_id = self.submit_job(split_script_position)
        split_job_id = 666

        # Step 2: Submit tracking and analysis jobs
        analysis_jobs = []
        for split_i in range(num_splits):
            # Assume each split file is named uniquely
            self.create_tracking_slurm_script(self.anticipate_split_video_position(split_i),split_i)
            #track_job_id = self.submit_job(f'track_job_{i}.sh', dependency_id=split_job_id)
            track_job_id = split_i+100
            self.create_trajectory_analysis_slurm_script(split_i,self.meta_data_table.stimuli_01[0] == self.meta_data_table.stimuli_01[split_i])
            #analysis_job_id = self.submit_job(f'analysis_job_{i}.sh', dependency_id=track_job_id)
            analysis_job_id = split_i+200
            analysis_jobs.append(analysis_job_id)

        # # Step 3: Create and submit the final job that depends on all analysis jobs
        # all_dependencies = ":".join(analysis_jobs)
        # self.create_slurm_script('final_job.sh', final_script, self.file_base_dir, f"{self.file_base_dir}/final_output.sql")
        # self.submit_job('final_job.sh', dependency_id=all_dependencies)


