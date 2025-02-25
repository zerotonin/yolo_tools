#!/bin/bash
#SBATCH --job-name=train_rtdetr
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=20GB

sleep 5
source /home/geuba03p/PyProjects/yolo_tools/config/conda.sh
conda activate yolov8

/home/geuba03p/miniconda3/envs/yolov8/bin/python /home/geuba03p/PyProjects/yolo_tools/single_ultralytics_train.py \
  --model rtdetr \
  --pretrained rtdetr-l.pt \
  --data /projects/sciences/zoology/geurten_lab/AI_trainData/uLytics_Lena/data.yaml \
  --epochs 400 \
  --batch_size 3 \
  --imgsz 640 \
  --project /home/geuba03p/object_det_comparison \
  --name rtdetr_run \
  --inference_images /projects/sciences/zoology/geurten_lab/AI_trainData/uLytics_Lena/images/val
