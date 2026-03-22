import time
import cv2 # type: ignore

class FPSCounter:
    """Helper class to calculate and display Frames Per Second (FPS)."""
    
    def __init__(self):
        self.p_time = 0

    def calculate_and_draw(self, img):
        """Calculates the FPS and draws it on the provided image."""
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if (c_time - self.p_time) > 0 else 0
        self.p_time = c_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        return img
