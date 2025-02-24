#!/bin/bash
#SBATCH --job-name=train_fcos
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=20GB

sleep 5
source /home/geuba03p/PyProjects/yolo_tools/config/conda.sh
conda activate yolov8

/home/geuba03p/miniconda3/envs/yolov8/bin/python /home/geuba03p/PyProjects/yolo_tools/single_model_train.py \
  --model fcos \
  --data_dir /projects/sciences/zoology/geurten_lab/AI_trainData/COCO_annotations_RetinaNet_MaskRCNN/ \
  --output_parent /home/geuba03p/object_det_comparison \
  --num_classes 3 \
  --batch_size 3 \
  --num_workers 2 \
  --num_epochs 400 \
  --split_ratio 0.8
