from ultralytics import YOLO

class CarriedObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.15):
        # Initialize YOLO independently (uses same local weights)
        self.model = YOLO(model_path)
        
        # Lower baseline confidence threshold to catch smaller or obfuscated items typically missed.
        self.conf_thresh = conf_thresh
        
        # We want to detect ANY object that is typically "carryable".
        # We ignore persons (0), large vehicles, animals, and large furniture.
        # This increases accuracy by preventing random background structure overlaps.
        self.ignore_classes = {
            0, # person
            1, 2, 3, 4, 5, 6, 7, 8, # vehicles
            9, 10, 11, 12, 13, 14, # fixed street items (traffic lights, benches)
            15, 16, 17, 18, 19, 20, 21, 22, 23, # animals
            56, 57, 58, 59, 60, 61, 62, # large furniture
            68, 69, 70, 71, 72 # large appliances
        }

    def detect(self, img):
        """
        Runs YOLOv8 inference to find all carryable objects with a lower threshold
        to catch hard-to-see items like cameras, laptops, or bottles.
        """
        # We use a lower threshold dynamically to cast a wider net
        results = self.model(img, stream=False, verbose=False)
        
        carried_items = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf < self.conf_thresh:
                    continue
                    
                if cls_id in self.ignore_classes:
                    continue # Skip non-carryable background/persons
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[cls_id]
                
                carried_items.append({
                    "box": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "conf": conf,
                    "associated_person_id": None
                })
                
        return carried_items

    def associate(self, carried_items, tracked_persons):
        """
        Calculates proximity/overlap to associate each generic carried item
        with a tracked person. Filters out unassociated items to avoid 
        false positive environmental clutter.
        """
        associated_items = []
        
        for item in carried_items:
            gx1, gy1, gx2, gy2 = item['box']
            item_area = (gx2 - gx1) * (gy2 - gy1)
            
            best_person_id = None
            max_overlap_ratio = 0.0
            
            for person_id, data in tracked_persons.items():
                px1, py1, px2, py2 = data["box"]
                
                # Expand person bounds logically to capture items held nearby
                # Increased the margin slightly to catch outstretched arms better
                margin_x = int((px2 - px1) * 0.5)
                margin_y = int((py2 - py1) * 0.15)
                
                epx1 = max(0, px1 - margin_x)
                epy1 = max(0, py1 - margin_y)
                epx2 = px2 + margin_x
                epy2 = py2 + margin_y
                
                # Intersection with expanded bounds
                x_left = max(epx1, gx1)
                y_top = max(epy1, gy1)
                x_right = min(epx2, gx2)
                y_bottom = min(epy2, gy2)
                
                if x_right < x_left or y_bottom < y_top:
                    continue # No overlap at all with this person
                    
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                if item_area > 0:
                    overlap_ratio = intersection_area / item_area
                    # Lower the overlap requirement since we filtered for carryables and use a tight bounding loop
                    if overlap_ratio > 0.05 and overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio
                        best_person_id = person_id
                
                # Fallback check by centroid if no major bounding box overlap
                if best_person_id is None:
                    gcx = (gx1 + gx2) / 2
                    gcy = (gy1 + gy2) / 2
                    if epx1 <= gcx <= epx2 and epy1 <= gcy <= epy2:
                        best_person_id = person_id
                        
            if best_person_id is not None:
                item['associated_person_id'] = best_person_id
                # Append only items that are actually physically associated with a person
                associated_items.append(item)
                
        return associated_items
