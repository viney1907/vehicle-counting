# Vehicle Counting with YOLO

This project uses the YOLO (You Only Look Once) object detection model to count vehicles in a video. The code processes each frame of the video to detect and track vehicles, providing a count of the detected objects within a defined region of interest (ROI).

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Customization](#customization)
7. [Results](#results)

## Features

- Detects and counts vehicles in a video using the YOLO model.
- Customizable region of interest (ROI) for counting objects.
- Visualizes object detection and tracking in real-time.
- Supports multiple object classes (e.g., vehicles, pedestrians).
- Outputs a processed video with counted objects and tracked trajectories.

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/vehicle-counting-yolo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd vehicle-counting-yolo
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your video file in the `data/videos` directory.

2. Modify the `video_path` variable in the code to point to your video file:
    ```python
    cap = cv2.VideoCapture("data/videos/video.mp4")
    ```

3. Run the script:
    ```bash
    python vehicle_counting.py
    ```

4. The processed video will be saved as `object_counting_output.avi` in the project directory.

## How It Works

1. **Loading the Model:** The pre-trained YOLO model is loaded to detect objects in the video frames.
2. **Video Processing:** The video is processed frame by frame to detect and track objects.
3. **Object Counting:** Objects that pass through the defined region of interest (ROI) are counted.
4. **Output:** The processed video with the counted objects and their tracked paths is saved and displayed.

## Customization

- **Region of Interest (ROI):** Modify the `line_points` variable to change the area where objects are counted.
    ```python
    line_points = [(170, 400), (1100, 400)]  # Define your ROI points here
    ```
- **Object Classes:** You can specify which object classes to count using the `classes_to_count` variable.
    ```python
    classes_to_count = [0, 2]  # Modify the class IDs as needed
    ```

## Results

The processed video will display:
- Bounding boxes around detected objects.
- Trajectories of tracked objects.
- Count of objects crossing the defined ROI.