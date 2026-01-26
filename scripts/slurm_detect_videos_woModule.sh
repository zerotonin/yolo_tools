#!/bin/bash
#SBATCH --job-name=yolo_detect_weta
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=40GB
#SBATCH --output=yolo_detect_%j.out
#SBATCH --error=yolo_detect_%j.err

# ============================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================

# Which species to process
SPECIES="h_thoracica"  # Options: h_maori, h_thoracica, h_crassidens

# Video input/output settings
VIDEO_DIR="/projects/sciences/zoology/geurten_lab/weta_videos_cropped/${SPECIES}"
VIDEO_OUTPUT_DIR="${VIDEO_DIR}/yolo_videos_$(date +%Y%m%d_%H%M%S)"
TRAJECTORY_OUTPUT_DIR="${VIDEO_DIR}/yolo_trajectories_$(date +%Y%m%d_%H%M%S)"

# Video writing control
SAVE_VIDEOS=false  # Set to false to skip video generation (MUCH faster, less disk space)

# YOLO model weights
YOLO_WEIGHTS="/home/geuba03p/PyProjects/yolo_tools/runs/detect/weta_yolo_11medium/weights/best.pt"

# Processing settings
MAX_PARALLEL=8  # Number of parallel processes

# ============================================================
# END CONFIGURATION
# ============================================================

# Wait for fileserver to be ready
echo "Waiting for fileserver to mount..."
sleep 5

# Print job information
echo "========================================"
echo "YOLO Detection - Weta Videos"
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# Create output directories
mkdir -p "$TRAJECTORY_OUTPUT_DIR"
if [ "$SAVE_VIDEOS" = true ]; then
    mkdir -p "$VIDEO_OUTPUT_DIR"
fi

echo "Configuration:"
echo "  Species: $SPECIES"
echo "  Video directory: $VIDEO_DIR"
echo "  YOLO weights: $YOLO_WEIGHTS"
echo "  Trajectory output: $TRAJECTORY_OUTPUT_DIR"
if [ "$SAVE_VIDEOS" = true ]; then
    echo "  Video output: $VIDEO_OUTPUT_DIR"
else
    echo "  Video output: DISABLED (trajectories only)"
fi
echo "  Max parallel jobs: $MAX_PARALLEL"
echo ""

# Check if weights exist
if [ ! -f "$YOLO_WEIGHTS" ]; then
    echo "ERROR: YOLO weights not found at $YOLO_WEIGHTS"
    exit 1
fi

# Count video files
VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" -type f | wc -l)
echo "Found $VIDEO_COUNT video files to process"
echo ""

# Function to process a single video
process_video() {
    local video_path=$1
    local video_name=$(basename "$video_path" .mp4)
    local gpu_id=$2
    
    echo "[GPU $gpu_id] Processing: $video_name"
    echo "[GPU $gpu_id] Started at: $(date '+%H:%M:%S')"
    
    # Set which GPU to use for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Build command based on whether we're saving videos
    local cmd="~/miniconda3/envs/yolov8/bin/python -m yolo_tools.detection.videoDetectorWithOutput \
        --video_path \"$video_path\" \
        --apriori_classes 0 \
        --apriori_class_names weta \
        --yolo_weights \"$YOLO_WEIGHTS\" \
        --output_file \"$TRAJECTORY_OUTPUT_DIR/${video_name}_trajectories.npy\" \
        --no_progress"
    
    if [ "$SAVE_VIDEOS" = true ]; then
        cmd="$cmd --save_video --output_video \"$VIDEO_OUTPUT_DIR/${video_name}_yolo_labelled.mp4\""
    fi
    
    # Run YOLO detection
    eval $cmd 2>&1 | sed "s/^/[GPU $gpu_id] /"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] ✓ Completed: $video_name at $(date '+%H:%M:%S')"
    else
        echo "[GPU $gpu_id] ✗ FAILED: $video_name (exit code: $exit_code)"
    fi
    
    return $exit_code
}

export -f process_video
export TRAJECTORY_OUTPUT_DIR VIDEO_OUTPUT_DIR YOLO_WEIGHTS SAVE_VIDEOS

# Get list of all video files
mapfile -t VIDEO_FILES < <(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" -type f | sort)

# Process videos in parallel batches
echo "========================================"
echo "Starting parallel processing..."
echo "========================================"
echo ""

processed=0
failed=0
total=${#VIDEO_FILES[@]}

# Process videos in batches
for ((i=0; i<total; i+=MAX_PARALLEL)); do
    batch_end=$((i + MAX_PARALLEL))
    if [ $batch_end -gt $total ]; then
        batch_end=$total
    fi
    
    echo "Processing batch $((i/MAX_PARALLEL + 1)): videos $((i+1)) to $batch_end of $total"
    echo "----------------------------------------"
    
    # Start parallel processes for this batch
    pids=()
    for ((j=i; j<batch_end; j++)); do
        gpu_id=$((j % MAX_PARALLEL))  # Rotate through GPU IDs if needed
        process_video "${VIDEO_FILES[$j]}" "$gpu_id" &
        pids+=($!)
    done
    
    # Wait for all processes in this batch to complete
    for pid in "${pids[@]}"; do
        if wait $pid; then
            ((processed++))
        else
            ((failed++))
        fi
    done
    
    echo ""
    echo "Batch complete. Progress: $processed/$total successful, $failed failed"
    echo ""
done

# Print final summary
echo ""
echo "========================================"
echo "PROCESSING COMPLETE"
echo "========================================"
echo "Total videos: $total"
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo ""
echo "Results saved to:"
echo "  Trajectories: $TRAJECTORY_OUTPUT_DIR"
if [ "$SAVE_VIDEOS" = true ]; then
    echo "  Videos: $VIDEO_OUTPUT_DIR"
fi
echo ""

# List output files
echo "Output files:"
echo ""
echo "Trajectories (.npy):"
ls -lh "$TRAJECTORY_OUTPUT_DIR"/*.npy 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' | head -10
traj_count=$(ls -1 "$TRAJECTORY_OUTPUT_DIR"/*.npy 2>/dev/null | wc -l)
echo "  Total: $traj_count files"

if [ "$SAVE_VIDEOS" = true ]; then
    echo ""
    echo "Labeled videos (.mp4):"
    ls -lh "$VIDEO_OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' | head -10
    video_count=$(ls -1 "$VIDEO_OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "  Total: $video_count files"
fi

echo ""
echo "Job completed at: $(date)"
echo "========================================"