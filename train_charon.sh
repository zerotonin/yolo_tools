#!/bin/bash
#SBATCH --job-name=train_charon
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=20GB

sleep 5  # wait on auto mount (if needed)
source /home/geuba03p/PyProjects/yolo_tools/config/conda.sh
conda activate yolov8

# TRAIN
/home/geuba03p/miniconda3/envs/yolov8/bin/python /home/geuba03p/PyProjects/yolo_tools/single_ultralytics_train.py \
  --model yolo11 \
  --pretrained yolo11l.pt \
  --data /projects/sciences/zoology/geurten_lab/AI_trainData/charon_data_2025/data.yaml \
  --epochs 1000 \
  --batch_size 10 \
  --imgsz 640 \
  --project /home/geuba03p/object_det_comparison \
  --name charon_run \
  --inference_images /projects/sciences/zoology/geurten_lab/AI_trainData/charon_data_2025/images/val
