import cv2
import re
from pathlib import Path

class VideoInfoExtractor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.date = None
        self.time = None
        self.fps = None

    def extract_datetime(self):
        """
        Extracts date and time from the video file name using regular expressions.
        Assumes date and time are formatted as YYYY_MM_DD__HH-MM-SS in the filename.
        """
        filename = self.video_path.name
        # Regular expression to find date and time in the specified format
        match = re.search(r'(\d{4}_\d{2}_\d{2})__(\d{2}-\d{2}-\d{2})', filename)
        if match:
            self.date = match.group(1).replace('_', '-')
            self.time = match.group(2).replace('-', ':')
        else:
            raise ValueError("Date and time format does not match the expected pattern.")

    def detect_fps(self):
        """
        Detects the frames per second (FPS) of the video using OpenCV.
        """
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError("Cannot open the video file.")
        
        # Get the FPS from the video capture
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def detect_duration(self):
        """
        Detects the duration of the video in seconds using OpenCV.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError("Cannot open the video file.")
        
        # Get the total frame count and FPS
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the duration in seconds
        self.duration = total_frames / fps
        cap.release()

    def get_video_info(self):
        """
        Processes the video file to extract date, time, FPS, and duration.
        Returns a dictionary with the extracted information.
        """
        self.extract_datetime()
        self.detect_fps()
        self.detect_duration()
        return {
            'date': self.date,
            'time': self.time,
            'fps': self.fps,
            'duration': self.duration
        }