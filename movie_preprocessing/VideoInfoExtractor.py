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

    def get_video_info(self):
        """
        Processes the video file to extract date, time, and FPS.
        Returns a dictionary with the extracted information.
        """
        self.extract_datetime()
        self.detect_fps()
        return {
            'date': self.date,
            'time': self.time,
            'fps': self.fps
        }