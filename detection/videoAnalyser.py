import training_tools.YoloWrapper as YoloWrapper
import os, cv2
import numpy as np
from tqdm import tqdm
from itertools import chain
import trajectory_analysis.trajectoryAnalyser as trajectoryAnalyser


def _tracking_result_to_cpu(results,frame_no):
    track_ids = results[frame_no].boxes.id.int().cpu().tolist()
    classes = results[frame_no].boxes.cls.int().cpu().tolist()
    confidence = results[frame_no].boxes.conf.cpu().tolist()
    boxes = results[frame_no].boxes.xyxyn.cpu().tolist()

    return track_ids,classes,confidence,boxes

def get_best_tracking_results(results, frame_no, apriori_classes):
    track_ids, classes, confidence, boxes = _tracking_result_to_cpu(results, frame_no)

    detections = zip(track_ids, classes, confidence, boxes)
    best_detections = {}

    for track_id, cls, conf, box in detections:
        if cls not in best_detections or conf > best_detections[cls]['confidence']:
            best_detections[cls] = {'track_id': track_id, 'confidence': conf, 'box': box}
    
    # Initialize a dictionary to hold the coordinates for all apriori classes,
    # filling with np.nan for the box if the class was not detected
    all_class_coords = {cls: [np.nan, np.nan, np.nan, np.nan] for cls in apriori_classes}
    
    # Update the coordinates for detected classes with their best detection box
    for cls, det in best_detections.items():
        if cls in all_class_coords:  # This check ensures cls is among the apriori classes
            all_class_coords[cls] = det['box']
   
    # Flatten the coordinates for all classes into a single list
    return list(chain(*all_class_coords.values()))

def get_detection_trajectories(results):
    coordinates = list()
    for frame_no in range(len(results)):
        coordinates.append(get_best_tracking_results(results,frame_no,apriori_classes))
    return np.array(coordinates)



# User Variables
video_path = '/home/geuba03p/2024_03_28__16-19-28_45.mp4'
max_frames = 3
apriori_classes = [0,1]
apriori_class_names = ['arena','fly']

# tracking
yolo_fly = YoloWrapper.YoloWrapper('runs/detect/fly_arena7/weights/best.pt')
results =  yolo_fly.model.track(video_path, conf=0.8)

# get the trajectories from the GPU
trajectories = get_detection_trajectories(results)
traAna = trajectoryAnalyser.trajectoryAnalyser()
traAna.analyse_trajectory(trajectories,0.1,True,True)

# Split Movie > Object Recognition Yolo > Trajectory Analysis > SQL DataBase + NPY
'''
Experiment Types 

Video Recording

Trial
- stimulus_list list(str)
- is_female (bool)
- genotype



'''


