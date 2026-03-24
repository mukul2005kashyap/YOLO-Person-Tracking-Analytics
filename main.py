import cv2  # type: ignore
import argparse
import sys
import os
from datetime import datetime

from detector import YOLODetector # type: ignore
from tracker import CentroidTracker # type: ignore
from classifier import ComplianceClassifier # type: ignore
from logger import CSVLogger # type: ignore
from utils import FPSCounter, draw_boxes, setup_video_writer # type: ignore
from carried_object_detector import CarriedObjectDetector # type: ignore
from alert_system import AlertSystem
def parse_args():
    parser = argparse.ArgumentParser(description="Compliance Detection System")
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--source', type=str, help='Input source: 0 for webcam, path to video or image file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_type = None
    input_path = None

    if getattr(args, 'source', None):
        if args.source == '0':
            input_type = 'webcam'
        elif args.source.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_type = 'image'
            input_path = args.source
        else:
            input_type = 'video'
            input_path = args.source
    elif args.webcam:
        input_type = 'webcam'
    elif args.video:
        input_type = 'video'
        input_path = args.video
    else:
        print("Error: Please specify an input source (--source, --webcam, or --video)")
        sys.exit(1)

    # Initialize Components
    detector = YOLODetector(model_path="yolov8n.pt", conf_thresh=0.5) 
    carried_detector = CarriedObjectDetector(model_path="yolov8n.pt", conf_thresh=0.3)
    tracker = CentroidTracker(maxDisappeared=30, maxDistance=90)
    classifier = ComplianceClassifier()
    logger = CSVLogger(filepath="log.csv")
    fps_counter = FPSCounter()
    alert_system = AlertSystem(cooldown=3.0)

    if input_type == 'image':
        print("Processing Image Input...")
        if not os.path.exists(input_path):
            print(f"Error: Image file '{input_path}' not found.")
            sys.exit(1)
            
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not read image '{input_path}'.")
            sys.exit(1)
            
        # 1. Detection Phase (Identify persons and luggage)
        person_boxes, goods = detector.detect(frame)
        
        # 1.5 Optional Detection Layer (Identify other carried generic items)
        generic_items = carried_detector.detect(frame)
        
        # 2. Tracking Phase (Assign unique ID, link centers across frames)
        tracked_objects = tracker.update(person_boxes)
        
        # 2.5 Associate Generic Items
        associated_generic_items = carried_detector.associate(generic_items, tracked_objects)
        
        # Log new person appearances independently
        for person_id in tracked_objects.keys():
            logger.log_detection(person_id)
            
        # 3. Classification Phase (Geographically associate goods to a tracked person)
        classified_persons = classifier.classify(tracked_objects, goods)
        
        # 4. Logging Phase
        for p in classified_persons:
            pid = p['id']
            status = p['status']
            if status == "Compliant":
                logger.log_event(pid, "Contains luggage")
            else:
                logger.log_event(pid, "No goods detected")
                
        # 5. Presentation / Visual Output
        frame = draw_boxes(frame, classified_persons, goods, carried_items=associated_generic_items, is_recording=False)
        frame = alert_system.process(frame, classified_persons, carried_items=associated_generic_items)
        
        # Save processed image
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join("outputs", f"image_output_{timestamp}.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Saved recording successfully to: {output_filename}")
        
        cv2.imshow("Compliance Detection System", frame)
        
        # Graceful exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
            
        cv2.destroyAllWindows()
        return

    # Video / Webcam Mode setup
    if input_type == 'webcam':
        cap = cv2.VideoCapture(0)
    elif input_type == 'video':
        if not os.path.exists(input_path):
            print(f"Error: Video file '{input_path}' not found.")
            sys.exit(1)
        cap = cv2.VideoCapture(input_path)

    # Validate Capture Stream
    if not cap.isOpened():
        print("Error: Video capture failed to open.")
        sys.exit(1)

    # Video output setup (Saves into outputs/ directory)
    video_writer, output_filename = setup_video_writer(cap, output_dir="outputs")
    is_recording = True
    print(f"Recording output to: {output_filename}")

    print("Starting compliance detection... Press 'q' to exit the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended (or unable to read from camera).")
            break
            
        # 1. Detection Phase (Identify persons and luggage)
        person_boxes, goods = detector.detect(frame)
        
        # 1.5 Optional Detection Layer (Identify other carried generic items)
        generic_items = carried_detector.detect(frame)
        
        # 2. Tracking Phase (Assign unique ID, link centers across frames)
        tracked_objects = tracker.update(person_boxes)
        
        # 2.5 Associate Generic Items
        associated_generic_items = carried_detector.associate(generic_items, tracked_objects)
        
        # Log new person appearances independently
        for person_id in tracked_objects.keys():
            logger.log_detection(person_id)
            
        # 3. Classification Phase (Geographically associate goods to a tracked person)
        classified_persons = classifier.classify(tracked_objects, goods)
        
        # 4. Logging Phase
        for p in classified_persons:
            pid = p['id']
            status = p['status']
            if status == "Compliant":
                logger.log_event(pid, "Contains luggage")
            else:
                logger.log_event(pid, "No goods detected")
                
        # 5. Presentation / Visual Output
        frame = draw_boxes(frame, classified_persons, goods, carried_items=associated_generic_items, is_recording=is_recording)
        frame = alert_system.process(frame, classified_persons, carried_items=associated_generic_items)
        frame = fps_counter.draw(frame)
        
        # Write the processed frame to the saved video file
        if is_recording:
            video_writer.write(frame)
        
        cv2.imshow("Compliance Detection System", frame)
        
        # Graceful exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting application...")
            break

    # Clean up and conclude
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Saved recording successfully to: {output_filename}")

if __name__ == "__main__":
    main()
