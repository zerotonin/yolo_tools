#!/bin/bash
#SBATCH --job-name=train_weta_yolo
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=20GB

# python -m yolo_tools.training_tools.train_bash_compatible  -d <dataset_dir> -i <images_dir> -l <labels_dir> -c <config_file> [-n <model_name>] [-e <epochs>] [--classes <class1> <class2> ...]
dataset_dir='/projects/sciences/zoology/geurten_lab/AI_inferenceGraphs/weta_yolo_model'
images_dir='/projects/sciences/zoology/geurten_lab/AI_trainData/weta_temperature-yoloformat-data/images'
labels_dir='/projects/sciences/zoology/geurten_lab/AI_trainData/weta_temperature-yoloformat-data/labels'
config_file='/projects/sciences/zoology/geurten_lab/AI_trainData/weta_temperature-yoloformat-data/dataset.yaml'
weight_type='11medium'
model_name='weta_yolo_11medium'
epochs=600
#classes


~/miniconda3/envs/yolov8/bin/python -m yolo_tools.training_tools.train_bash_compatible -d $dataset_dir -i $images_dir -l $labels_dir -c $config_file -n $model_name -w $weight_type -e $epochs
