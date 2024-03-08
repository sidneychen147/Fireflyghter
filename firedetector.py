"""
firedetector.py
Detector class to identify fire objects
"""
import cv2
import numpy as np
import detector as dt


# find fire using pretrained haar cascade
def findfire_haar(img):
    dt.find_haar_cascade(img, "files/haarcascade_fire.xml", (255, 0, 0))


class FireDetector(dt.Detector):
    def __init__(self):
        super().__init__()
        self.maxframecount = 2
        self.displaytext = 'Potential fire detected'

    def set_displaytext(self, new_text):
        self.displaytext = new_text

    def detect(self, image=None):
        super().detect(image)
        contours = []

        # turn current frame into HSV
        hsv = cv2.cvtColor(self.framelist[-1], cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for fire color in HSV
        lower_bound = np.array([0, 127, 200])  # 18, 50, 50])
        upper_bound = np.array([18, 255, 255])  # 35, 255, 255])

        # Create a binary mask for the fire color range
        fire_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Perform morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.erode(fire_mask, kernel, iterations=1)
        fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

        # Check for fire motion
        if len(self.framelist) >= self.maxframecount:
            previous_frame = cv2.cvtColor(self.framelist[-2], cv2.COLOR_BGR2HSV)[:, :, 2]
            current_frame = hsv[:, :, 2]
            if len(previous_frame) == len(current_frame):
                frame_diff = cv2.absdiff(previous_frame, current_frame)
                _, threshold_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

                # Check for both fire color and fire motion, generate overlapped contours
                intersection = cv2.bitwise_and(fire_mask, threshold_diff)
                contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # refine contours by clustering and filtering out potential false positives (remove the smallest contours)
        contours = dt.cluster_contours(contours, 2)
        contours = dt.cluster_contours(filter(lambda cntr: cv2.contourArea(cntr) > 10, contours))
        if len(contours) > 0:
            max_contour = contours[0]
            for contour in contours:
                if cv2.contourArea(contour) > cv2.contourArea(max_contour):
                    max_contour = contour

            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(self.framelist[-1], (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(self.framelist[-1], self.displaytext, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return True
        '''# If any regions are detected, identify them
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.framelist[-1], (x, y), (x + w, y + h), (255, 0, 0), 2)'''
        return False
