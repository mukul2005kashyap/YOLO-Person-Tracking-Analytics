class ComplianceClassifier:
    def __init__(self):
        pass
        
    def _check_overlap(self, person_box, goods_box):
        """
        Check if a given 'goods' box is significantly overlapping or 
        proximally attached to a 'person' box. Improved Logic using IoU.
        """
        px1, py1, px2, py2 = person_box
        gx1, gy1, gx2, gy2 = goods_box['box']
        
        # Extend the person's bounding box dynamically to account for goods 
        # being held next to their body (outstretched arms, rolling suitcase, etc.)
        margin_x = int((px2 - px1) * 0.4) # Add 40% of their own width as margin horizontally
        margin_y = int((py2 - py1) * 0.1) # Add 10% of their height vertically
        
        epx1 = max(0, px1 - margin_x)
        epy1 = max(0, py1 - margin_y)
        epx2 = px2 + margin_x
        epy2 = py2 + margin_y
        
        # Calculate intersection rectangle with extended person bounds
        x_left = max(epx1, gx1)
        y_top = max(epy1, gy1)
        x_right = min(epx2, gx2)
        y_bottom = min(epy2, gy2)
        
        # Check if there is NO intersection
        if x_right < x_left or y_bottom < y_top:
            return False
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        goods_area = (gx2 - gx1) * (gy2 - gy1)
        
        # Primary check: If the intersection represents >10% of the goods bounding box area
        if goods_area > 0 and (intersection_area / goods_area) > 0.10:
            return True
            
        # Secondary check: If the goods centroid is inside the expanded person region
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        if epx1 <= gcx <= epx2 and epy1 <= gcy <= epy2:
            return True
            
        return False

    def classify(self, tracked_persons, goods):
        """
        Takes the tracked person data from CentroidTracker and the detected goods.
        Returns a structured list identifying which persons are compliant.
        """
        classified_persons = []
        
        # tracked_persons format: {id: {"centroid": (x,y), "box": [x1,y1,x2,y2]}}
        for person_id, data in tracked_persons.items():
            box = data["box"]
            
            is_compliant = False
            # Check interaction against all goods in the frame to establish compliance
            for g in goods:
                if self._check_overlap(box, g):
                    is_compliant = True
                    break
                    
            status = "Compliant" if is_compliant else "Non-Compliant"
            classified_persons.append({
                "id": person_id,
                "box": box,
                "status": status
            })
            
        return classified_persons
