# Compliance Detection System

## Project Overview
This project is a real-time Compliance Detection System built using Python, OpenCV, and ultralytics YOLOv8. It detects persons in an environment and tracks them across frames. It also detects goods/luggage (such as backpacks, suitcases, and handbags). By associating the detected goods with a tracked person using bounding-box overlap, the system classifies each person as "Compliant" (carrying goods) or "Non-Compliant" (no goods detected). All relevant detection and status changes are logged to a CSV file.

## Project Structure
- `main.py`: The entry point script that connects all modules and handles video/webcam input.
- `detector.py`: A wrapper around YOLOv8 that performs object detection on frames and filters targeted classes.
- `tracker.py`: Implements a fast Centroid Tracking algorithm to track persons across frames and assign unique IDs.
- `classifier.py`: Connects persons with detected goods using geometric overlap and sets compliance status.
- `logger.py`: Saves detection events and status changes to `log.csv`.
- `utils.py`: Contains drawing utilities for bounding boxes and an FPS counter.

## Installation Steps
1. Make sure you have Python installed.
2. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Using a Video File
To run on a pre-recorded video:
```bash
python main.py --video path/to/video.mp4
```

### Using a Webcam
To run live detection using your webcam:
```bash
python main.py --webcam
```

## Explanation of Output
- **Visuals:** 
  - **Persons:** A GREEN bounding box indicates "Compliant" (carrying luggage). A RED bounding box indicates "Non-Compliant" (without luggage). The unique Person ID is shown above the box.
  - **Goods/Luggage:** A Light Blue/Cyan bounding box shows the detected bags/boxes with confidence probabilities.
  - **FPS:** Displayed at the top-left corner.
- **Log Data (`log.csv`):**
  - Records events for each unique Person ID. Format: `Timestamp, Person ID, Event Message`.
