import cv2
import detector as dt


def findFace(img):
    dt.find_haar_cascade(img, "files/haarcascade_frontalface_default.xml", (0, 255, 0))


def findBody(img):
    dt.find_haar_cascade(img, "files/haarcascade_upperbody.xml", (0, 255, 255))
