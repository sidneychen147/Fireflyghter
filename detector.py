"""
detector.py
Base class for derived detector classes
"""
import numpy as np
import cv2


def find_haarcascade(img, haarcascadepath, outputcolor):
    hc = cv2.CascadeClassifier(haarcascadepath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hc.detectMultiScale(img_gray,1.2, 8)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)


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
