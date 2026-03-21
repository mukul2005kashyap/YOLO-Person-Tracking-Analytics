import time
import os
import cv2  # type: ignore
from datetime import datetime

class FPSCounter:
    def __init__(self):
        self.pTime = 0

    def get_fps(self):
        cTime = time.time()
        # Avoid division by zero
        diff = cTime - self.pTime
        fps = 1 / diff if diff > 0 else 0
        self.pTime = cTime
        return fps

    def draw(self, img):
        fps = self.get_fps()
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2)
        return img

def setup_video_writer(cap, output_dir="outputs"):
    """
    Sets up the OpenCV VideoWriter to save the stream.
    Creates the outputs directory if needed and generates a timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"output_{timestamp}.mp4")
    
    # Get original video width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0 # Default fallback if webcam doesn't report it

    # Setup the output video file format and codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    return out, filename

def draw_boxes(img, persons, goods, carried_items=None, is_recording=False):
    """
    Draws bounding boxes for both persons (with IDs and Status) and detected goods.
    Also handles optionally mapping dynamically carried general items.
    """
    # Draw dynamically tracked generic items in Orange
    if carried_items:
        for item in carried_items:
            x1, y1, x2, y2 = item['box']
            color = (0, 165, 255) # Orange for generic objects
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Item: {item['class_name']} {item['conf']:.2f}",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Draw goods in BLUE
    for g in goods:
        x1, y1, x2, y2 = g['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
        cv2.putText(img, f"{g['class_name'].capitalize()} {g['conf']:.2f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw persons
    for p in persons:
        x1, y1, x2, y2 = p['box']
        # Green for compliant, Red for non-compliant
        color = (0, 255, 0) if p['status'] == 'Compliant' else (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID: {p['id']} - {p['status']}"
        cv2.putText(img, label, (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        # Show what they are carrying if successfully associated
        extra_items = []
        if carried_items:
            extra_items = [item['class_name'] for item in carried_items if item.get('associated_person_id') == p['id']]
        
        if extra_items:
            items_str = ", ".join(extra_items)
            carrying_label = f"Carrying Item: {items_str}"
            cv2.putText(img, carrying_label, (x1, max(0, y1 + 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
    # Display Recording indicator
    if is_recording:
        cv2.putText(img, "Recording...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return img
