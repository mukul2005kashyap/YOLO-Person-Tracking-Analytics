# Fire Detection Module

This module adds robust fire detection capabilities to the Compliance Detection System.

## Setup
By default, the standard `yolov8n.pt` COCO model does not support fire objects. To enable fire detection:
1. Train or download a custom YOLOv8 model tailored for detecting fire (with classes like 'fire', 'flame').
2. Place the weights file in the root directory named `fire_model.pt`.
3. The system will automatically load and use it during real-time processing. 

If this model is not deployed, the module will print a warning but allow the rest of the application to execute smoothly without crashing.
