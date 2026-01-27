#!/bin/bash
#SBATCH --job-name=extract_backgrounds
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --output=extract_bg_%j.out
#SBATCH --error=extract_bg_%j.err

# ============================================================
# CONFIGURATION
# ============================================================

# Base directory for all species
BASE_DIR="/projects/sciences/zoology/geurten_lab/weta_videos_cropped"

# Species to process
SPECIES=("h_maori" "h_thoracica" "h_crassidens")

# Sample rate (process every Nth frame, higher = faster)
SAMPLE_RATE=10

# Number of parallel processes
MAX_PARALLEL=3  # Process all 3 species simultaneously

# Script location
EXTRACT_SCRIPT="/home/geuba03p/PyProjects/yolo_tools/scripts/extract_backgrounds.py"

# ============================================================
# END CONFIGURATION
# ============================================================

# Auto-cleanup old log files (keep last 10 runs)
echo "Cleaning up old log files..."
ls -t extract_bg_*.out 2>/dev/null | tail -n +11 | xargs -r rm
ls -t extract_bg_*.err 2>/dev/null | tail -n +11 | xargs -r rm

# Wait for fileserver
echo "Waiting for fileserver to mount..."
sleep 5

# Print job information
echo "========================================"
echo "Video Background Extraction"
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# Print Python info
echo "Python version:"
python --version
echo ""

# Check if script exists
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "ERROR: Extract script not found at $EXTRACT_SCRIPT"
    exit 1
fi

echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Species: ${SPECIES[@]}"
echo "  Sample rate: every $SAMPLE_RATE frames"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Script: $EXTRACT_SCRIPT"
echo ""

# Function to process one species
process_species() {
    local species=$1
    local video_dir="$BASE_DIR/$species"
    
    echo "[$species] Starting at $(date '+%H:%M:%S')"
    echo "[$species] Video directory: $video_dir"
    
    if [ ! -d "$video_dir" ]; then
        echo "[$species] ERROR: Directory not found: $video_dir"
        return 1
    fi
    
    # Count videos
    video_count=$(find "$video_dir" -maxdepth 1 -name "*.mp4" -type f | wc -l)
    echo "[$species] Found $video_count videos to process"
    
    # Run extraction
    python "$EXTRACT_SCRIPT" \
        --video_dir "$video_dir" \
        --sample_rate $SAMPLE_RATE \
        2>&1 | sed "s/^/[$species] /"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$species] ✓ Completed at $(date '+%H:%M:%S')"
    else
        echo "[$species] ✗ FAILED (exit code: $exit_code)"
    fi
    
    return $exit_code
}

export -f process_species
export BASE_DIR SAMPLE_RATE EXTRACT_SCRIPT

# Process all species in parallel
echo "========================================"
echo "Starting parallel processing..."
echo "========================================"
echo ""

pids=()
for species in "${SPECIES[@]}"; do
    process_species "$species" &
    pids+=($!)
done

# Wait for all species to complete
echo "Waiting for all species to complete..."
echo ""

successful=0
failed=0

for i in "${!SPECIES[@]}"; do
    if wait ${pids[$i]}; then
        ((successful++))
    else
        ((failed++))
    fi
done

# Print summary
echo ""
echo "========================================"
echo "PROCESSING COMPLETE"
echo "========================================"
echo "Total species: ${#SPECIES[@]}"
echo "Successfully processed: $successful"
echo "Failed: $failed"
echo ""

# List generated backgrounds for each species
for species in "${SPECIES[@]}"; do
    bg_count=$(find "$BASE_DIR/$species" -maxdepth 1 -name "*_background.png" -type f 2>/dev/null | wc -l)
    echo "$species: $bg_count background images"
    
    # Show first few examples
    if [ $bg_count -gt 0 ]; then
        echo "  Examples:"
        find "$BASE_DIR/$species" -maxdepth 1 -name "*_background.png" -type f 2>/dev/null | head -3 | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "    $(basename "$file") ($size)"
        done
    fi
    echo ""
done

echo "Job completed at: $(date)"
echo "========================================"