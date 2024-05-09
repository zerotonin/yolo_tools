import subprocess
import os

class SlurmJobManager:
    def __init__(self, analysis_file_manager,module_name,arena_num):
        self.file_manager = analysis_file_manager
        self.file_base_dir =  self.file_manager.path_dict['output_file_path']
        self.module_name = module_name
        self.user_name = os.getlogin()
        self.python_path =  self.file_manager.file_dict['python_interpreter']
        self.arena_num = arena_num

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
        python_command = f"{self.python_path} -m yolo_tools.{self.module_name}.{script_parameters['python_script']} {script_parameters['script_variables']}"
        
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


    def create_video_splitting_slurm_script(self,input_file_position,meomory_GB_int = 64, nodes = 1, cpus_per_task = 1, ntasks = 1):
        
        script_variables = f'--video_path { self.file_manager.file_dict['video_file_position']} --output_folder {self.file_base_dir}/preprocessed_single_videos --output_type videos'
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = f'{self.file_base_dir}/slurm_scripts/split_video.sh'
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = f'{self.script_base_dir}/movie_preprocessing/FrameSplitter.py'
        script_parameters['jobname'] =  f'split_{os.path.basename(input_file_position)}'
        script_parameters['memory'] = meomory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks

        self.create_slurm_script(script_parameters)



    def manage_workflow(self, initial_input, num_splits, split_script, tracking_script, analysis_script, final_script):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: Create and submit the split job
        split_output = f"{self.file_base_dir}/split_output"
        self.create_slurm_script('split_job.sh', split_script, initial_input, split_output)
        split_job_id = self.submit_job('split_job.sh')

        # Step 2: Submit tracking and analysis jobs
        analysis_jobs = []
        for i in range(num_splits):
            # Assume each split file is named uniquely
            input_file = f"{split_output}_part_{i}.mov"
            track_output = f"{self.file_base_dir}/track_output_{i}.dat"
            self.create_slurm_script(f'track_job_{i}.sh', tracking_script, input_file, track_output)
            track_job_id = self.submit_job(f'track_job_{i}.sh', dependency_id=split_job_id)
            
            analysis_output = f"{self.file_base_dir}/analysis_output_{i}.dat"
            self.create_slurm_script(f'analysis_job_{i}.sh', analysis_script, track_output, analysis_output)
            analysis_job_id = self.submit_job(f'analysis_job_{i}.sh', dependency_id=track_job_id)
            analysis_jobs.append(analysis_job_id)

        # Step 3: Create and submit the final job that depends on all analysis jobs
        all_dependencies = ":".join(analysis_jobs)
        self.create_slurm_script('final_job.sh', final_script, self.file_base_dir, f"{self.file_base_dir}/final_output.sql")
        self.submit_job('final_job.sh', dependency_id=all_dependencies)



# Usage of the class
if __name__ == "__main__":
    self = SlurmJobManager('/home/geuba03p/food_example_video/output','/home/geuba03p/PyProjects/yolo_tools','/home/geuba03p/miniconda3/envs/yolov8/bin/python')
    script_parameters = dict()

    input_file_position = f'/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
    self.create_video_splitting_slurm_script(input_file_position)

    """

    partition = 'aoraki'
    input_file_position = f'/home/geuba03p/food_example_video/original/2024_03_28__16-19-28.mp4'
    slurm_filename      = f'{self.file_base_dir}/slurm_scripts/split_video.sh'
    video_output_folder = f'{self.file_base_dir}/preprocessed_single_videos'
    meomory_GB_int = 64
    python_script_position = '/home/geuba03p/PyProjects/yolo_tools/movie_preprocessing/FrameSplitter.py'
    user_name = 'geuba03p'
    script_variables = f'--video_path {input_file_position} --output_folder {video_output_folder} --output_type videos'
    gpu_num = 1
    nodes = 1
    cpus_per_task = 1
    ntasks = 1

    if partition == 'aoraki':
        gpu_num = 0

    script_parameters['partition'] =  partition
    script_parameters['filename'] = slurm_filename
    script_parameters['cpus_per_task'] = cpus_per_task
    script_parameters['python_script'] = python_script_position
    script_parameters['jobname'] =  f'split_{os.path.basename(input_file_position)}'
    script_parameters['account_name'] = user_name
    script_parameters['memory'] = meomory_GB_int
    script_parameters['script_variables'] = script_variables
    script_parameters['gpus_per_task'] = gpu_num
    script_parameters['nodes'] = nodes
    script_parameters['ntasks_per_node'] = ntasks
    script_parameters['python_path'] = python_path

    self.create_slurm_script(script_parameters)
"""