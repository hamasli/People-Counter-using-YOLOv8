# People Counter using YOLOv8

The **People Counter using YOLOv8** project is a real-time system designed for accurate people counting in dynamic environments such as malls, public spaces, or event venues. Leveraging YOLOv8 for object detection, OpenCV for video processing, and SORT (Simple Online and Realtime Tracking) for object tracking, this project provides robust performance and enhances situational awareness.

## Overview

This project utilizes YOLOv8, a powerful object detection model, to detect and localize individuals (persons) in real-time video streams. Using bounding box coordinates provided by YOLOv8, the system tracks these detections across frames using SORT, ensuring accurate and consistent counting of entries and exits.

## Key Features

- **Object Detection**: YOLOv8 identifies and localizes individuals with high accuracy in real-time video footage.
- **Object Tracking**: SORT tracks detected individuals across frames, maintaining unique identifiers for each person to count entries and exits.
- **Visual Feedback**: Real-time graphical overlays provide visual cues within the video feed, indicating counts and movement direction.
- **Performance Optimization**: Integration with efficient algorithms and libraries ensures real-time processing suitable for surveillance and monitoring applications.

## Usage

To use the **People Counter using YOLOv8**:

1. **Clone the Repository**: Clone the repository to your local machine.
       https://github.com/hamasli/People-Counter-using-YOLOv8/
   ```bash
  
   cd People-Counter-using-YOLOv8
