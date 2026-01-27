#!/usr/bin/env python3
"""
Extract mean background image from videos
Processes all videos in a directory and saves {filename}_background.png
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_mean_background(video_path, output_path=None, sample_rate=10, show_progress=True):
    """
    Extract mean background image from a video
    
    Args:
        video_path (str|Path): Path to video file
        output_path (str|Path|None): Path to save background image. 
                                      If None, uses {video_name}_background.png
        sample_rate (int): Sample every Nth frame (default: 10 for speed)
        show_progress (bool): Show progress bar
        
    Returns:
        np.ndarray: Mean background image
    """
    video_path = Path(video_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate how many frames we'll actually process
    frames_to_process = total_frames // sample_rate
    
    if show_progress:
        print(f"Processing: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        print(f"  Sampling every {sample_rate} frames ({frames_to_process} frames)")
    
    # Accumulator for mean calculation
    mean_image = np.zeros((height, width, 3), dtype=np.float64)
    frame_count = 0
    
    # Read and accumulate frames
    pbar = tqdm(total=frames_to_process, desc="  Extracting", disable=not show_progress)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame
        if i % sample_rate == 0:
            mean_image += frame.astype(np.float64)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    pbar.close()
    
    # Calculate mean
    if frame_count == 0:
        raise ValueError(f"No frames could be read from {video_path}")
    
    mean_image = (mean_image / frame_count).astype(np.uint8)
    
    # Save image
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_background.png"
    else:
        output_path = Path(output_path)
    
    cv2.imwrite(str(output_path), mean_image)
    
    if show_progress:
        print(f"  ✓ Saved background: {output_path.name}")
    
    return mean_image


def process_directory(video_dir, output_dir=None, pattern="*.mp4", sample_rate=10):
    """
    Process all videos in a directory
    
    Args:
        video_dir (str|Path): Directory containing videos
        output_dir (str|Path|None): Directory to save backgrounds. 
                                     If None, saves next to videos
        pattern (str): File pattern to match (default: *.mp4)
        sample_rate (int): Sample every Nth frame
    """
    video_dir = Path(video_dir)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = sorted(list(video_dir.glob(pattern)))
    
    if not video_files:
        print(f"No videos found matching {pattern} in {video_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Video Background Extractor")
    print(f"{'='*60}")
    print(f"Video directory: {video_dir}")
    print(f"Found {len(video_files)} videos")
    print(f"Sample rate: every {sample_rate} frames")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_path in video_files:
        try:
            if output_dir:
                output_path = output_dir / f"{video_path.stem}_background.png"
            else:
                output_path = None
            
            extract_mean_background(video_path, output_path, sample_rate)
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {video_path.name} - {e}")
            failed += 1
        
        print()  # Blank line between videos
    
    # Summary
    print(f"{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract mean background images from videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all MP4s in current directory:
  python extract_backgrounds.py --video_dir .
  
  # Process specific directory with custom sample rate:
  python extract_backgrounds.py --video_dir /path/to/videos --sample_rate 20
  
  # Save backgrounds to separate directory:
  python extract_backgrounds.py --video_dir videos/ --output_dir backgrounds/
  
  # Process single video:
  python extract_backgrounds.py --video_path video.mp4
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_dir', type=str, 
                      help='Directory containing videos to process')
    group.add_argument('--video_path', type=str, 
                      help='Single video file to process')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for background images (default: same as videos)')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='File pattern for videos (default: *.mp4)')
    parser.add_argument('--sample_rate', type=int, default=10,
                       help='Sample every Nth frame (default: 10, higher=faster)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for single video mode')
    
    args = parser.parse_args()
    
    if args.video_dir:
        process_directory(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            sample_rate=args.sample_rate
        )
    else:  # Single video
        extract_mean_background(
            video_path=args.video_path,
            output_path=args.output_path,
            sample_rate=args.sample_rate,
            show_progress=True
        )


if __name__ == '__main__':
    main()