#!/bin/bash
#SBATCH --job-name=train_yolov8_2
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=20GB

# python -m yolo_tools.training_tools.train_bash_compatible  -d <dataset_dir> -i <images_dir> -l <labels_dir> -c <config_file> [-n <model_name>] [-e <epochs>] [--classes <class1> <class2> ...]
dataset_dir='/projects/sciences/zoology/geurten_lab/AI_trainData/koen_3_chambers_linear/dataset'
images_dir='/projects/sciences/zoology/geurten_lab/AI_trainData/koen_3_chambers_linear/frames'
labels_dir='/projects/sciences/zoology/geurten_lab/AI_trainData/koen_3_chambers_linear/annotations'
config_file='/projects/sciences/zoology/geurten_lab/AI_trainData/koen_3_chambers_linear/fly_arena_3Clin.yaml'
model_name='fly_arena_3Clin'
epochs=600
#classes


~/miniconda3/envs/yolov8/bin/python -m yolo_tools.training_tools.train_bash_compatible -d $dataset_dir -i $images_dir -l $labels_dir -c $config_file -n $model_name -e $epochs
