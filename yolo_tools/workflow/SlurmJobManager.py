import subprocess
import os

class SlurmJobManager:
    def __init__(self, analysis_file_manager, arena_num, meta_data_table, video_duration_sec, gpu_partition='aoraki_gpu'):
        self.file_manager = analysis_file_manager
        self.file_base_dir = self.file_manager.path_dict['output_file_path']
        self.user_name = os.getlogin()
        self.python_path = self.file_manager.file_dict['python_interpreter']
        self.arena_num = arena_num
        self.meta_data_table = meta_data_table
        self.video_duration_sec = video_duration_sec
        self.gpu_partition = gpu_partition
        self.runtime_factor = 1  # This factor is to calculate the time each step (splitting, tracking, analyzing) needs given the video duration.
    
    def format_duration_for_sbatch(self,duration_sec):
        """ Formats the duration in seconds to the SBATCH time format (D-HH:MM:SS)."""
        seconds = int(duration_sec*self.runtime_factor)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def submit_job(self, script_path, dependency_id=None):
        """ Submits a job to the SLURM scheduler with an optional dependency."""
        cmd = ['sbatch']
        if dependency_id:
            cmd.append(f'--dependency=afterok:{dependency_id}')
        cmd.append(script_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f'Job {job_id} submitted.')
            return job_id
        raise Exception(f"Failed to submit job: {result.stderr}")



    def create_slurm_script(self, script_parameters):
        """
        Generates a SLURM script file for a given analysis task using parameters from a dictionary.

        Args:
            script_parameters (dict): Parameters for the SLURM script including:
                - partition (str): The partition where the job should run. 
                                    If set to "all_aoraki_gpus" it will execute on:
                                    aoraki_gpu_L40, aoraki_gpu_A100_40GB, aoraki_gpu_A100_80GB, aoraki_gpu_H100
                - filename (str): The filename for the SLURM script.
                - python_script (str): Path to the Python script to execute.
                - jobname (str): Name of the job.
                - memory (str): Memory allocation for the job.
                - script_variables (str): String with variables to pass to the Python script.
                - gpus_per_task (int): Number of GPUs per task.
                - nodes (int): Number of nodes to use.
                - ntasks_per_node (int): Number of tasks per node.
                - runtime_sec(num): How long the script can run at max in seconds
        """
        
        # Build the command line for the Python script with additional variables
        if type(script_parameters['script_variables']) == list:
            python_command = ''
            for scipt_vars in script_parameters['script_variables']:
                    python_command += f"{self.python_path} -m yolo_tools.{script_parameters['module']}.{script_parameters['python_script']} {scipt_vars}&\n"
            #python_command = python_command[:-2] # to get rid of the last &
        else:
            python_command = f"{self.python_path} -m yolo_tools.{script_parameters['module']}.{script_parameters['python_script']} {script_parameters['script_variables']}"
        
        # Construct the SLURM script content
        content =  f'#!/bin/bash\n'
        content += f'#SBATCH --job-name={script_parameters['jobname']}\n'
        content += f'#SBATCH --account={self.user_name}\n'
        if script_parameters['partition'] == 'all_aoraki_gpus':
            content += f'#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu_A100_40GB,aoraki_gpu_A100_80GB,aoraki_gpu_H100\n'
        else:
            content += f'#SBATCH --partition={script_parameters['partition']}\n'
        content += f'#SBATCH --cpus-per-task={script_parameters['cpus_per_task']}\n'
        if script_parameters['partition'] != 'aoraki':
            content += f'#SBATCH --gpus-per-task={script_parameters['gpus_per_task']}\n'

        content += f'#SBATCH --nodes={script_parameters['nodes']}\n'
        content += f'#SBATCH --mem={script_parameters['memory']}G\n'
        content += f'#SBATCH --ntasks-per-node={script_parameters['ntasks_per_node']}\n'
        content += f"#SBATCH --time={self.format_duration_for_sbatch(script_parameters['runtime_sec'])}\n"
        content += f'#SBATCH --output={self.file_base_dir}/slurm_logs/%x.out\n'
        content += f'#SBATCH --error={self.file_base_dir}/slurm_logs/%x.err\n'
        content += f'\n'
        content += f'sleep 5 # wait on auto mount\n'
        content += f'source {self.file_manager.file_dict['conda_script_position']}\n' # This should come from the filemanager
        content += f'conda activate {self.file_manager.file_dict['conda_env_name']}\n'
        content += f'{python_command}'
        content += f'\nwait\n'


        # Write the SLURM script to a file
        with open(script_parameters['filename'], 'w') as f:
            f.write(content)

    def create_tracking_slurm_script(self,gpu_jobs,gpus_per_task =1, memory_GB_int = 16, nodes = 1, cpus_per_task = 1, ntasks = 1):


        script_variable_list = list()        
        for split_i in gpu_jobs:
            script_variable_list.append(f'--video_path { self.file_manager.anticipate_split_video_position(split_i)}  --apriori_classes 0 1 --apriori_class_names arena fly --yolo_weights {self.file_manager.file_dict['yolo_weights']} --output_file {self.file_manager.create_yolo_trajectory_filepath(split_i)}')
        
        arena_list = list(gpu_jobs)
        arena_id_str = f'{str(min(arena_list)).zfill(2)}-{str(max(arena_list)).zfill(2)}'

        script_parameters = dict()
        script_parameters['partition'] =  self.gpu_partition
        script_parameters['gpus_per_task'] = gpus_per_task
        script_parameters['filename'] = os.path.join(self.file_manager.path_dict['slurm_scripts'], f'track_arena_{arena_id_str}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'videoAnalyser'
        script_parameters['jobname'] =  f'track_arena_{arena_id_str}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variable_list
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'detection'
        script_parameters['runtime_sec'] = self.video_duration_sec*3

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']


    def create_trajectory_analysis_slurm_script(self, arena_num, positive_stimulus_on_left,
                                                filter_trajectory=True,midline_tolerance = 0.1,memory_GB_int = 32, 
                                                nodes = 1, cpus_per_task = 1, ntasks = 1):

        input_file_position    = self.file_manager.create_yolo_trajectory_filepath(arena_num)
        output_locomotion_file = self.file_manager.create_locomotor_result_base_path(arena_num)
        output_decision_file   = self.file_manager.create_decision_result_base_path(arena_num)
        
        script_variables = f'--input_file {input_file_position} --midline_tolerance {midline_tolerance} --positive_stimulus_on_left {positive_stimulus_on_left} --filter_trajectory {filter_trajectory} --output_locomotion_file {output_locomotion_file} --output_decision_file {output_decision_file}'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = os.path.join(self.file_manager.path_dict['slurm_scripts'], f'analyse_arena_{str(arena_num).zfill(2)}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'trajectoryAnalyser'
        script_parameters['jobname'] =  f'analyse_arena_{str(arena_num).zfill(2)}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'trajectory_analysis'
        script_parameters['runtime_sec'] = self.video_duration_sec*0.1

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']
    
    def create_sql_entry_slurm_script(self,memory_GB_int = 32, nodes = 1, cpus_per_task = 1, ntasks = 1):

        
        script_variables = f'--db_url sqlite:///{self.file_manager.file_dict['db_file_position']} --meta_data_csv_path {self.file_manager.file_dict['meta_data_csv_file']} --result_base_path {self.file_manager.path_dict['results']}'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = os.path.join(self.file_manager.path_dict['slurm_scripts'],'enter_SQL.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'ResultManager'
        script_parameters['jobname'] =  f'SQL_entry_2choice'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'database'
        script_parameters['runtime_sec'] = self.video_duration_sec*0.66

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']

    def create_video_splitting_slurm_script(self,memory_GB_int = 32, nodes = 1, cpus_per_task = 1, ntasks = 1):
        
        script_variables = f'--video_path {self.file_manager.file_dict['video_file_position']} --output_folder {self.file_base_dir}/preprocessed_single_videos --output_type videos'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = os.path.join(self.file_manager.path_dict['slurm_scripts'],'split_video.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'FrameSplitter'
        script_parameters['jobname'] =  f'split_{os.path.basename(self.file_manager.file_dict['video_file_position'])}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'video_preprocessing'
        script_parameters['runtime_sec'] = self.video_duration_sec*0.33

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']

    def chunk_list(self, job_list, chunk_size):
        """Split the data into chunks of chunk_size."""
        return [job_list[i:i + chunk_size] for i in range(0, len(job_list), chunk_size)]


    def manage_workflow(self, num_splits,wait_on_job_before_start = None,gpu_chunk_size = 4):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: Create and submit the split job
        split_script_filepath = self.create_video_splitting_slurm_script()
        split_job_id = self.submit_job(split_script_filepath,wait_on_job_before_start)

        # Step 2: Submit tracking and analysis jobs
        gpu_job_chunks = self.chunk_list(range(num_splits),gpu_chunk_size)
        analysis_jobs = []

        for gpu_jobs in gpu_job_chunks:
            
            track_script_filepath = self.create_tracking_slurm_script(gpu_jobs)
            track_job_id = self.submit_job(track_script_filepath, dependency_id=split_job_id)
            
            for split_i in gpu_jobs:
                ana_script_filepath =self.create_trajectory_analysis_slurm_script(split_i,self.meta_data_table.stimuli_01[split_i] == self.meta_data_table.expected_attractive_stim_id[split_i])
                analysis_job_id = self.submit_job(ana_script_filepath, dependency_id=track_job_id)
                analysis_jobs.append(analysis_job_id)

        # Step 3: Create and submit the final job that depends on all analysis jobs
        sql_script_filepath =self.create_sql_entry_slurm_script()
        all_dependencies = ":".join(str(job_id) for job_id in analysis_jobs)
        self.submit_job(sql_script_filepath, dependency_id=all_dependencies)


    def rerun_traj_analysis(self, num_splits,old_dependency=None):
        """
        Rerun the analysis.
        """


        # Step 2: Submit tracking and analysis jobs
        analysis_jobs = []
        for split_i in range(num_splits):
           
            ana_script_filepath =self.create_trajectory_analysis_slurm_script(split_i,self.meta_data_table.stimuli_01[split_i] == self.meta_data_table.expected_attractive_stim_id[split_i])
            analysis_job_id = self.submit_job(ana_script_filepath,dependency_id=old_dependency)
            analysis_jobs.append(analysis_job_id)

        # Step 3: Create and submit the final job that depends on all analysis jobs
        sql_script_filepath =self.create_sql_entry_slurm_script()
        all_dependencies = ":".join(str(job_id) for job_id in analysis_jobs)
        last_job_id = self.submit_job(sql_script_filepath, dependency_id=all_dependencies)
        return last_job_id