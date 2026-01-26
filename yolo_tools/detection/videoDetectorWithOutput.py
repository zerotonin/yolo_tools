"""
Standalone YOLO detector with video output capability.
Inherits from the original YOLO_detector without modifying it.

Save this as: yolo_tools/detection/videoDetectorWithOutput.py
"""

import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
from yolo_tools.detection.videoAnalyser import YOLO_detector


class YOLO_VideoDetector(YOLO_detector):
    """
    Extended YOLO detector with video visualization capabilities.
    Inherits all functionality from YOLO_detector and adds video output.
    """
    
    def __init__(self, video_path, apriori_classes, apriori_class_names, yolo_weights, max_frames=None):
        """Initialize with same parameters as parent class"""
        super().__init__(video_path, apriori_classes, apriori_class_names, yolo_weights, max_frames)
        
        # Color map for different classes (BGR format for OpenCV)
        self.class_colors = {
            0: (0, 255, 255),    # Yellow
            1: (255, 0, 255),    # Magenta
            2: (0, 255, 0),      # Green
            3: (255, 0, 0),      # Blue
            4: (0, 165, 255),    # Orange
            5: (203, 192, 255),  # Pink
            6: (128, 0, 128),    # Purple
            7: (0, 255, 127),    # Spring Green
        }
    
    def draw_detections_on_frame(self, frame, frame_result):
        """
        Draw bounding boxes and labels on a frame
        
        Args:
            frame: The video frame (numpy array)
            frame_result: YOLO detection result for this frame
            
        Returns:
            frame: Frame with drawn bounding boxes
        """
        if frame_result.boxes is None or len(frame_result.boxes) == 0:
            return frame
        
        h, w = frame.shape[:2]
        
        try:
            track_ids, classes, confidence, boxes = self._tracking_result_to_cpu(frame_result)
            
            for track_id, cls, conf, box in zip(track_ids, classes, confidence, boxes):
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                # Get color for this class
                color = self.class_colors.get(cls, (255, 255, 255))
                
                # Draw bounding box (thicker for better visibility)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Prepare label text
                class_name = self.apriori_class_names[self.apriori_classes.index(cls)] if cls in self.apriori_classes else f"Class_{cls}"
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                
                # Get text size for background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text (with padding)
                padding = 5
                cv2.rectangle(frame, 
                            (x1, y1 - text_h - baseline - padding * 2), 
                            (x1 + text_w + padding * 2, y1), 
                            color, -1)
                
                # Draw text
                cv2.putText(frame, label, 
                          (x1 + padding, y1 - baseline - padding), 
                          font, font_scale, (0, 0, 0), thickness)
                
        except Exception as e:
            print(f"Warning: Could not draw detections on frame: {e}")
        
        return frame
    
    def analyze_and_save_video(self, trajectory_output, video_output=None, show_progress=True):
        """
        Analyze video, save trajectories, and optionally create labeled video
        
        Args:
            trajectory_output (str|Path): Path to save trajectory .npy file
            video_output (str|Path|None): Path to save labeled video. If None, only saves trajectories.
            show_progress (bool): Show progress bar (default: True)
            
        Returns:
            trajectories (np.ndarray): Array of trajectories
        """
        trajectory_output = Path(trajectory_output)
        
        # Open video to get properties
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"\nVideo properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Run YOLO tracking
        print(f"\nRunning YOLO detection on: {Path(self.video_path).name}")
        results = self.yolo_fly.model.track(
            self.video_path, 
            conf=0.8, 
            stream=True, 
            persist=True, 
            verbose=False
        )
        
        # Process results - collect all data first
        coordinates = []
        frames_for_video = [] if video_output is not None else None
        frame_count = 0
        
        # Determine max frames for progress bar
        max_iter = self.max_frames if self.max_frames is not None else total_frames
        
        pbar = tqdm(total=max_iter, desc="Detecting", disable=not show_progress)
        
        for frame_result in results:
            # Get trajectories
            try:
                coordinates.append(self.get_best_tracking_results(frame_result))
            except Exception as e:
                # Create empty detection with correct size
                num_values = len(self.apriori_classes) * 4
                coordinates.append([np.nan] * num_values)
            
            # Store frame for later video writing (if needed)
            if frames_for_video is not None:
                frame = frame_result.orig_img.copy()
                frame = self.draw_detections_on_frame(frame, frame_result)
                frames_for_video.append(frame)
            
            frame_count += 1
            pbar.update(1)
            
            # Stop if max_frames reached
            if self.max_frames is not None and frame_count >= self.max_frames:
                break
        
        pbar.close()
        
        # PRIORITY 1: Save trajectories FIRST (fast, critical data)
        trajectories = np.array(coordinates)
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(trajectory_output, trajectories)
        print(f"\n✓ Saved trajectories: {trajectory_output} (shape: {trajectories.shape})")
        
        # PRIORITY 2: Write video AFTER trajectories are safe (slow, optional)
        if frames_for_video is not None and len(frames_for_video) > 0:
            video_output = Path(video_output)
            video_output.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\nWriting labeled video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
            
            if video_writer.isOpened():
                for frame in tqdm(frames_for_video, desc="Encoding", disable=not show_progress):
                    video_writer.write(frame)
                video_writer.release()
                print(f"✓ Saved labeled video: {video_output}")
            else:
                print(f"✗ Failed to create video writer for {video_output}")
        
        return trajectories


def main():
    """Command line interface for the video detector"""
    parser = argparse.ArgumentParser(
        description='YOLO Video Detector with trajectory and video output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just save trajectories:
  python videoDetectorWithOutput.py --video_path video.mp4 --yolo_weights best.pt --output_file trajectories.npy
  
  # Save trajectories AND labeled video:
  python videoDetectorWithOutput.py --video_path video.mp4 --yolo_weights best.pt --output_file trajectories.npy --save_video
  
  # Specify custom video output name:
  python videoDetectorWithOutput.py --video_path video.mp4 --yolo_weights best.pt --output_file traj.npy --save_video --output_video labeled.mp4
        """
    )
    
    parser.add_argument('--video_path', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--yolo_weights', type=str, required=True, 
                       help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--output_file', type=str, required=True, 
                       help='Path to save trajectory data (.npy file)')
    
    parser.add_argument('--apriori_classes', nargs='+', type=int, default=[0], 
                       help='List of class IDs to detect (default: [0])')
    parser.add_argument('--apriori_class_names', nargs='+', type=str, default=['weta'], 
                       help='Names for each class (default: [weta])')
    
    parser.add_argument('--save_video', action='store_true', 
                       help='Save labeled video with bounding boxes')
    parser.add_argument('--output_video', type=str, default=None, 
                       help='Path for output video (default: {output_file}_labeled.mp4)')
    
    parser.add_argument('--max_frames', type=int, default=None, 
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--no_progress', action='store_true', 
                       help='Disable progress bar')

    args = parser.parse_args()
    
    # Validate class names match class IDs
    if len(args.apriori_classes) != len(args.apriori_class_names):
        parser.error("Number of class IDs must match number of class names")
    
    # Create output video path if save_video is True but no path specified
    video_output = None
    if args.save_video:
        if args.output_video is None:
            output_path = Path(args.output_file)
            video_output = str(output_path.parent / f"{output_path.stem}_labeled.mp4")
        else:
            video_output = args.output_video
    
    # Create detector
    print("="*60)
    print("YOLO Video Detector")
    print("="*60)
    print(f"Video: {args.video_path}")
    print(f"Weights: {args.yolo_weights}")
    print(f"Classes: {dict(zip(args.apriori_classes, args.apriori_class_names))}")
    print(f"Output trajectory: {args.output_file}")
    if video_output:
        print(f"Output video: {video_output}")
    print("="*60)
    
    detector = YOLO_VideoDetector(
        video_path=args.video_path,
        apriori_classes=args.apriori_classes,
        apriori_class_names=args.apriori_class_names,
        yolo_weights=args.yolo_weights,
        max_frames=args.max_frames
    )
    
    # Run detection
    trajectories = detector.analyze_and_save_video(
        trajectory_output=args.output_file,
        video_output=video_output,
        show_progress=not args.no_progress
    )
    
    print("\n" + "="*60)
    print("✓ Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()