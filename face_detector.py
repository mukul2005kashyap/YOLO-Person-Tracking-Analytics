import cv2 # type: ignore
import mediapipe as mp 

class FaceDetector:
    """Class to detect faces using MediaPipe Face Detection."""
    
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con = min_detection_con
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_detection_con)

    def find_faces(self, img, draw=True):
        """Detects faces in an image and draws a bounding boxes with a confidence score."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb) # type: ignore
        bboxs = []

        if self.results.detections:  # type: ignore
            for id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                       int(bbox_c.width * iw), int(bbox_c.height * ih)
                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 
                                2, (255, 0, 255), 2)
        return img, bboxs

    def fancy_draw(self, img, bbox, l=30, t=5, rt=1):
        """Draws a sleek, fancy cornered bounding box around the detected face."""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw the base rectangle
        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top Left (x, y)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # Top Right (x1, y)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom Left (x, y1)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom Right (x1, y1)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img
