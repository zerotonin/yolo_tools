import numpy as np
from tqdm import tqdm
from itertools import chain
import argparse
from yolo_tools.trajectory_analysis.trajectoryAnalyser import trajectoryAnalyser
from yolo_tools.training_tools.YoloWrapper import YoloWrapper

class YOLO_detector:
    def __init__(self, video_path, apriori_classes, apriori_class_names, yolo_weights, max_frames =None):
        self.video_path = video_path
        self.max_frames = max_frames
        self.apriori_classes = apriori_classes
        self.apriori_class_names = apriori_class_names
        self.yolo_weights = yolo_weights
        self.yolo_fly = YoloWrapper(yolo_weights)

    def _tracking_result_to_cpu(self, results, frame_no):
        track_ids = results[frame_no].boxes.id.int().cpu().tolist()
        classes = results[frame_no].boxes.cls.int().cpu().tolist()
        confidence = results[frame_no].boxes.conf.cpu().tolist()
        boxes = results[frame_no].boxes.xyxyn.cpu().tolist()
        return track_ids, classes, confidence, boxes

    def get_best_tracking_results(self, results, frame_no):
        track_ids, classes, confidence, boxes = self._tracking_result_to_cpu(results, frame_no)
        detections = zip(track_ids, classes, confidence, boxes)
        best_detections = {}
        for track_id, cls, conf, box in detections:
            if cls not in best_detections or conf > best_detections[cls]['confidence']:
                best_detections[cls] = {'track_id': track_id, 'confidence': conf, 'box': box}
        
        all_class_coords = {cls: [np.nan, np.nan, np.nan, np.nan] for cls in self.apriori_classes}
        for cls, det in best_detections.items():
            if cls in all_class_coords:
                all_class_coords[cls] = det['box']
        
        return list(chain(*all_class_coords.values()))

    def get_detection_trajectories(self, results):
        coordinates = []
        for frame_no in range(len(results)):
            try:
                coordinates.append(self.get_best_tracking_results(results, frame_no))
            except:
                coordinates.append([])

        return np.array(coordinates)

    def analyze_video(self,file_name):
        results = self.yolo_fly.model.track(self.video_path, conf=0.8)
        trajectories = self.get_detection_trajectories(results)
        np.save(file_name,trajectories)

def main():
    parser = argparse.ArgumentParser(description='YOLO Detector for analyzing videos.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video.')
    parser.add_argument('--max_frames', type=int, default=3, help='Maximum number of frames to process.')
    parser.add_argument('--apriori_classes', nargs='+', type=int, default=[0, 1], help='List of apriori classes.')
    parser.add_argument('--apriori_class_names', nargs='+', type=str, default=['arena', 'fly'], help='Names of the apriori classes.')
    parser.add_argument('--yolo_weights', type=str, default='runs/detect/fly_arena7/weights/best.pt', help='Path to YOLO weights.')
    parser.add_argument('--output_file', type=str, help='Path where the .npy trajectory file is saved')

    args = parser.parse_args()

    detector = YOLO_detector(args.video_path, args.apriori_classes, args.apriori_class_names, args.yolo_weights, max_frames=args.max_frames)
    detector.analyze_video(args.output_file)

if __name__ == '__main__':
    main()

# /home/geuba03p/miniconda3/envs/yolov8/bin/python -m yolo_tools.detection.videoAnalyser --video_path '/home/geuba03p/2024_03_28__16-19-28_45.mp4' --apriori_classes 0 1 --apriori_class_names arena fly --yolo_weights 'resources/yolov8_weights_for_single_2chamberSeparatedArena_singleFly.pt'