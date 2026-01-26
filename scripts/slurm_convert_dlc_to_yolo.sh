#!/bin/bash
#SBATCH --job-name=dlc_to_yolo
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=4GB
#SBATCH --output=dlc_to_yolo_%j.out
#SBATCH --error=dlc_to_yolo_%j.err

# Wait for fileserver to be ready
echo "Waiting for fileserver to mount..."
sleep 5

# Print job information
echo "========================================"
echo "DLC to YOLO Conversion Job"
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# Print Python and package versions
echo "Python version:"
python --version
echo ""
echo "Checking required packages:"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "from PIL import Image; print('Pillow: installed')"
python -c "import tqdm; print(f'tqdm: {tqdm.__version__}')"
echo ""

# Count files to process
echo "Counting files in labeled-data directory..."
total_files=$(find labeled-data -type f -name "*.png" | wc -l)
echo "Total PNG files found: $total_files"
echo ""

# Run the conversion script with absolute path
echo "Starting DLC to YOLO conversion..."
echo "----------------------------------------"
python /home/geuba03p/PyProjects/yolo_tools/scripts/convert_dlc_to_yolo.py

# Check exit status
exit_status=$?
echo ""
echo "----------------------------------------"
echo "Job completed at: $(date)"
echo "Exit status: $exit_status"

if [ $exit_status -eq 0 ]; then
    echo ""
    echo "Conversion successful!"
    echo "Checking output directory..."
    if [ -d "yolo_dataset/images" ]; then
        converted_images=$(ls yolo_dataset/images/*.png 2>/dev/null | wc -l)
        converted_labels=$(ls yolo_dataset/labels/*.txt 2>/dev/null | wc -l)
        echo "  Images converted: $converted_images"
        echo "  Labels created: $converted_labels"
    fi
else
    echo ""
    echo "Conversion failed! Check error log: dlc_to_yolo_${SLURM_JOB_ID}.err"
fi

echo "========================================"