from djitellopy import tello
import cv2
import numpy as np
from time import sleep
import keyboardcontrol as kc
from persondetector import findBody, findFace
from firedetector import FireDetector


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kc.getKey("LEFT"):
        lr = -speed
    elif kc.getKey("RIGHT"):
        lr = speed

    if kc.getKey("UP"):
        fb = speed
    elif kc.getKey("DOWN"):
        fb = -speed

    if kc.getKey("w"):
        ud = speed
    elif kc.getKey("s"):
        ud = -speed

    if kc.getKey("a"):
        yv = -speed
    elif kc.getKey("d"):
        yv = speed

    if kc.getKey("q"):
        yv = drone.land()
    #if kc.getKey("e"): yv = drone.takeoff()

    return[lr, fb, ud, yv]


drone = tello.Tello()
drone.connect()

print(drone.get_battery())

drone.streamon()

kc.init()

start = 0
firedetector = FireDetector()

while True:
    image = drone.get_frame_read().frame

    if start == 0:
        drone.takeoff()
        start = 1

    findBody(image)
    findFace(image)
    firedetector.nextframe(image)
    firedetector.detect()

    image = cv2.resize(image, (360, 240))
    cv2.imshow("Image", image)

    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.streamoff()
        break

cv2.destroyAllWindows()
