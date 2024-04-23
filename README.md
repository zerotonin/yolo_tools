# YOLO Tools

## Introduction
YOLO Tools is a comprehensive toolkit designed to facilitate the deployment and operation of the YOLO (You Only Look Once) AI model for real-time object detection and tracking. This software pipeline processes video input, detects and tracks objects, and manages data efficiently for various applications such as surveillance, traffic monitoring, and activity recognition.

## Features
- **Real-Time Object Detection**: Utilize the powerful YOLO model for fast and accurate object detection.
- **Video Preprocessing**: Standardize video inputs to enhance model performance.
- **Trajectory Analysis**: Analyze and predict the paths of moving objects within video frames.
- **Database Management**: Store and manage detection data for easy retrieval and analysis.

## Installation
To set up YOLO Tools on your system, follow the steps below:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yolo_tools.git
   cd yolo_tools
   pip install -r requirements.txt
   ```
## Usage

t.b.a.

# Image Recognition Tracking Software Pipeline

## Abstract
This document describes the architecture of an image recognition tracking software pipeline. The pipeline is designed to process video input, detect and track objects, and manage data output using modern machine learning techniques and efficient data handling practices.

## Partition of Labour

## 1. Train the YOLO AI
The first step involves training the YOLO (You Only Look Once) artificial intelligence model for real-time object detection. This phase includes:
- Collecting and preparing a diverse dataset of images annotated with bounding boxes around the target objects. labelImg
- Configuring the YOLO architecture suitable for the specific requirements, balancing detection speed and accuracy.
- Training the model using a split of training and validation data to ensure generalization.
- Evaluating model performance with standard metrics like mean Average Precision (mAP) and Intersection over Union (IoU).

### 2. Movie Preprocessing
Before object detection:
- We correct lense distortions
- We split the multi arena movie into single arena movies

### 3. Object Detection with Trained AI
With the trained YOLO model:
- Each video frame is input into the model.
- The model predicts bounding boxes and class probabilities for each detected object.
- Post-processing refines detections based on confidence thresholds.

### 4. Trajectory Analysis
After detecting objects:
- Trajectories of moving objects are analyzed to track their motion across frames.
- This step involves calculating the movement vectors for each object and predicting future positions.
- Useful in applications like surveillance to monitor paths and behaviors.

### 5. Database Management
Data handling:
- Detection data, including, object classifications, locations, and decisions, are stored in a structured database.
- The database is maintained through sqlite3 and sqlalchemy. 


## Detection Pipeline Flow

Inputs                        Movie Preprocessing           Detection               trajec. analysis     Database Management

file with meta data        -> multiple meta data files   --------------------------------------------->    -------------
                                                                                                           | SQLite DB |
movie with multiple arenas -> multiple movies of 1 arena -> mult trajectories -> mult. decision data ->    -------------

Status:                     meta: open | data: done          data: done           data: work             db: done | handler: open
