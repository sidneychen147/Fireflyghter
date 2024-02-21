import cv2


class Detector:
    def __init__(self):
        self.maxframecount = 1
        self.framelist = []
        return

    def nextframe(self, nextframe):
        if len(self.framelist) >= self.maxframecount:
            self.framelist.pop(0)
        self.framelist.append(nextframe)
        return

    def detect(self):

        return