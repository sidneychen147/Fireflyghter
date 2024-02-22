import cv2
import detector as dt


def findFace(img):
    faceCascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.2,8)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h),(255,0,0),2)


def findBody(img):
    bodyCascade = cv2.CascadeClassifier("files/haarcascade_upperbody.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = bodyCascade.detectMultiScale(imgGray, 1.05, 3, minSize=(100, 100))

    for (x, y, w, h) in bodies:
        cv2.rectangle(img,(x, y), (x+w, y+h), (0, 255, 0), 2)
