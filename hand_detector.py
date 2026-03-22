import cv2 # type: ignore
import mediapipe as mp # type: ignore

class HandDetector:
    """Class to detect hands and hand landmarks using MediaPipe."""
    
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = float(detection_con) # Ensure float
        self.track_con = float(track_con) # Ensure float

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Tip ids for fingers: Thumb, Index, Middle, Ring, Pinky
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = []

    def find_hands(self, img, draw=True):
        """Finds hands in a BGR image and optionally draws landmarks and connections."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb) # type: ignore
        
        if self.results.multi_hand_landmarks:    # type: ignore
            for hand_lms in self.results.multi_hand_landmarks:   
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Finds pixel positions of landmarks for a specific hand."""
        self.lm_list = []
        if self.results.multi_hand_landmarks:   # type: ignore
            # Check if requested hand_no exists
            if hand_no < len(self.results.multi_hand_landmarks):    # type: ignore
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        """Detects how many fingers are up based on the landmark positions."""
        fingers = []
        if len(self.lm_list) != 0:
            # Thumb
            # Condition based on x-coordinate (assumes right hand, simplified logic)
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Remaining 4 Fingers
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers
