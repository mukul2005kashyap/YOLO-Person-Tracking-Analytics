import cv2  # type: ignore

def draw_fire_boxes(img, fire_detections):
    """
    Draws bounding boxes and labels for detected fire regions.
    Uses ORANGE (BGR: 0, 165, 255) color for the bounding boxes.
    """
    for f in fire_detections:
        # f is a tuple: (x1, y1, x2, y2)
        x1, y1, x2, y2 = f
        color = (0, 165, 255)  # Orange in BGR
        
        # Draw thick bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw Label
        label = "FIRE DETECTED"
        # Background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        # Text
        cv2.putText(img, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
    return img
