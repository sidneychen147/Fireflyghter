"""
firedetector.py
Detector class to identify fire objects
"""
import cv2
import numpy as np
import detector as dt


def findfire_haar(img):
    dt.find_haarcascade(img, "files/haarcascade_fire.xml", (0, 0, 255))


class FireDetector(dt.Detector):
    def __init__(self):
        super().__init__()
        self.maxframecount = 2

    def detect(self):
        contours = []

        # turn current frame into HSV
        hsv = cv2.cvtColor(self.framelist[-1], cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for fire color in HSV
        lower_bound = np.array([0, 127, 200])
        upper_bound = np.array([18, 255, 255])

        # Create a binary mask for the fire color range
        fire_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Perform morphological operations to remove noise
        '''kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.erode(fire_mask, kernel, iterations=1)
        fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)'''

        # Check for fire motion
        if len(self.framelist) >= self.maxframecount:
            if (len(cv2.cvtColor(self.framelist[-2], cv2.COLOR_BGR2HSV)[:, :, 2]) == len(hsv[:, :, 2])):
                frame_diff = cv2.absdiff(cv2.cvtColor(self.framelist[-2], cv2.COLOR_BGR2HSV)[:, :, 2], hsv[:, :, 2])
                _, threshold_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

                # Check for both fire color and fire motion, generate overlapped contours
                intersection = cv2.bitwise_and(fire_mask, threshold_diff)
                contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Draw red rectangles around detected regions
            for contour in contours:
                if cv2.contourArea(contour) > 5:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(self.framelist[-1], (x, y), (x + w, y + h), (255, 0, 0), 2)
