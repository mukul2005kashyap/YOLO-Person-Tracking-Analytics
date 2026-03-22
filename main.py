import cv2 # type: ignore
import sys
import os
import time
import threading
import winsound
from datetime import datetime
from face_detector import FaceDetector  # type: ignore
from hand_detector import HandDetector # type: ignore
from utils import FPSCounter # type: ignore

def main():
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Real-Time Detectors (Faces & Hands)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Detectors (Faces & Hands)", 1200, 800)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure it's connected and accessible.")
        sys.exit(1)
        
    
    face_detector = FaceDetector()
    hand_detector = HandDetector()
    fps_counter = FPSCounter()

    # Create directory for captured images (Cooldown & Save Logic)
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)
    
    last_capture_time = 0
    message_display_time = 0

    print("Starting webcam... Press 'q' to exit the application.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to grab frame from the webcam.")
            break

        
        img, _ = face_detector.find_faces(img)
        
    
        img = hand_detector.find_hands(img)

        total_fingers = 0

        if hand_detector.results.multi_hand_landmarks:
            for hand_no in range(len(hand_detector.results.multi_hand_landmarks)):
        
                lm_list = hand_detector.find_position(img, hand_no, draw=False)
        
                if len(lm_list) != 0:
                    fingers = hand_detector.fingers_up()
                    total_fingers += fingers.count(1)   # type: ignore

        # Image Capture Logic (10 Fingers)
        current_time = time.time()

        if total_fingers == 10:

            if current_time - last_capture_time > 3.0:                   # type: ignore
                # 1. Capture the frame and create a timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"img_{timestamp}.jpg")
                
                # 2. Save the image
                cv2.imwrite(filename, img)
                print(f"Image Captured: {filename}")
                
                # 3. Optional beep sound (non-blocking)
                threading.Thread(target=winsound.Beep, args=(1000, 200), daemon=True).start()
                
                # 4. Update timers
                last_capture_time = current_time
                message_display_time = current_time

        # 5. Display confirmation message for 2 seconds after capture
        if current_time - message_display_time < 2.0:               # type: ignore
            cv2.putText(img, "Image Captured!", (400, 100), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.putText(img, f'Fingers: {total_fingers}', (20, 100), 
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Calculate and display FPS
        img = fps_counter.calculate_and_draw(img)
        # Show the processed image
        cv2.imshow("Real-Time Detectors (Faces & Hands)", img)

        # Allow pressing 'q' to gracefully exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting application...")
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
