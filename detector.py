"""
detector.py
Base class for derived detector classes
"""
import numpy as np
import cv2


def find_haar_cascade(img, haar_cascade_path, output_color):
    hc = cv2.CascadeClassifier(haar_cascade_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = hc.detectMultiScale(img_gray, 1.2, 8)

    for (x, y, w, h) in contours:
        cv2.rectangle(img, (x, y), (x+w, y+h), output_color, 2)


def calculate_contour_distance(contour_1, contour_2):
    # find center of contour 1
    x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
    c_x_1 = x_1 + w_1 / 2
    c_y_1 = y_1 + h_1 / 2

    # find center of contour 2
    x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)
    c_x_2 = x_2 + w_2 / 2
    c_y_2 = y_2 + h_2 / 2

    d_x = abs(c_x_1 - c_x_2) - (w_1 + w_2) / 2
    d_y = abs(c_y_1 - c_y_2) - (h_1 + h_2) / 2

    return max(d_x, d_y)


def cluster_contours(contours, threshold_distance=40.0):
    current_contours = list(contours)
    # loop while there are still multiple contours that have distances less than the threshold
    while len(current_contours) > 1:
        min_distance = None
        min_coordinates = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinates = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinates = (x, y)

        if min_distance < threshold_distance:
            index_1, index_2 = min_coordinates
            current_contours[index_1] = np.concatenate((current_contours[index_1], current_contours[index_2]), axis=0)
            del current_contours[index_2]
        else:
            break
    return current_contours


class Detector:
    def __init__(self):
        self.maxframecount = 1
        self.framelist = []

    def nextframe(self, nextframe):
        if len(self.framelist) >= self.maxframecount:
            self.framelist.pop(0)
        self.framelist.append(nextframe)

    def detect(self, image=None):
        if isinstance(image, np.ndarray):
            self.nextframe(image)
