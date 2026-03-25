import cv2
import numpy as np
import time
from ultralytics import YOLO

class FireDetector:
    def __init__(self):
        # Try loading YOLO model (custom or default)
        try:
            self.model = YOLO("yolov8n.pt")  # replace with fire-trained model if available
        except:
            self.model = None

        self.last_alert_time = 0
        self.cooldown = 3  # seconds

    def detect_fire_yolo(self, frame):
        fire_boxes = []

        if self.model is None:
            return fire_boxes

        results = self.model(frame, conf=0.3)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                # YOLO default me fire nahi hota, but just in case
                if "fire" in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    fire_boxes.append((x1, y1, x2, y2))

        return fire_boxes

    def detect_fire_hsv(self, frame):
        fire_boxes = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire color range (red/orange/yellow)
        lower1 = np.array([0, 120, 200])
        upper1 = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower1, upper1)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 500:  # filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                fire_boxes.append((x, y, x+w, y+h))

        return fire_boxes

    def detect_fire(self, frame):
        # Combine YOLO + HSV
        yolo_boxes = self.detect_fire_yolo(frame)
        hsv_boxes = self.detect_fire_hsv(frame)

        return yolo_boxes + hsv_boxes

    def draw_fire(self, frame, boxes):
        alert_triggered = False

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE DETECTED",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            alert_triggered = True

        # Alert system with cooldown
        current_time = time.time()

        if alert_triggered and (current_time - self.last_alert_time > self.cooldown):
            self.last_alert_time = current_time

        if alert_triggered:
            cv2.putText(frame, "🔥 ALERT: FIRE DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        return frame