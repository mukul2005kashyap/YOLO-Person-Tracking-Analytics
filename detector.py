from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4):
        # Initialize the YOLOv8 model (downloads 'yolov8n.pt' automatically if not found)
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        
        # COCO dataset class IDs:
        # 0: person
        # 24: backpack
        # 26: handbag 
        # 28: suitcase
        # 39: bottle
        # 63: laptop
        # 67: cell phone
        # These represent typical carried goods/bags/boxes.
        self.target_classes = {
            0: "person", 
            24: "backpack", 
            26: "handbag", 
            28: "suitcase",
            39: "bottle",
            63: "laptop",
            67: "cell phone"
        }

    def detect(self, img):
        """
        Runs YOLOv8 inference on the given image.
        Returns a list of person bounding boxes and a list of goods/luggage objects.
        """
        # verbose=False reduces terminal spam during live video detection
        # iou=0.45 adjusts NMS threshold to handle overlapping objects
        results = self.model(img, stream=False, verbose=False, iou=0.45, imgsz=640)
        
        persons = []
        goods = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue
                    
                cls_id = int(box.cls[0])
                if cls_id in self.target_classes:
                    # Bounding box integer coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = self.target_classes[cls_id]
                    
                    if cls_id == 0:
                        # Append box for persons
                        persons.append([x1, y1, x2, y2])
                    else:
                        # Append distinct data block for goods
                        goods.append({
                            "box": [x1, y1, x2, y2], 
                            "class_name": class_name, 
                            "conf": conf
                        })
                        
        return persons, goods
