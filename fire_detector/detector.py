import os
import cv2  # type: ignore
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path="fire_model.pt", conf_thresh=0.45):
        """
        Initializes the YOLO model for fire detection.
        If the specified model_path doesn't exist, it won't crash but will warn.
        """
        self.conf_thresh = conf_thresh
        self.model_loaded = False
        self.model_path = model_path
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"Successfully loaded fire detection model: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load fire model {model_path}: {e}")
        else:
            print(f"Warning: Fire model '{model_path}' not found. Fire detection will be inactive.")
            print(f"Please place a trained fire YOLO model at: {model_path}")
            
        # Target class names for fire. Custom models might use ID 0 or name 'fire'.
        self.fire_class_names = ["fire", "flame", "smoke"]

    def detect(self, img):
        """
        Runs YOLO inference to detect fire.
        Returns a list of dictionaries containing fire detections.
        """
        fire_detections = []
        if not self.model_loaded:
            return fire_detections
            
        # Run inference
        results = self.model(img, stream=False, verbose=False, iou=0.45, imgsz=640)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue
                    
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id].lower()
                
                # Check if it's fire (or if the model only has 1 class, assume it is fire)
                if any(f in class_name for f in self.fire_class_names) or len(self.model.names) == 1:
                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                     fire_detections.append({
                         "box": [x1, y1, x2, y2],
                         "class_name": "FIRE",
                         "conf": conf
                     })
                     
        return fire_detections
