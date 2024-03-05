from djitellopy import tello
import cv2
import numpy as np
from time import sleep
from persondetector import findBody, findFace
from firedetector import FireDetector

drone = tello.Tello()
drone.connect()

print(drone.get_battery())

drone.streamon()

firedetector = FireDetector()

while True:
    image = drone.get_frame_read().frame


    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    findBody(image)
    findFace(image)

    firedetector.nextframe(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    firedetector.detect()

    image = cv2.resize(image, (800, 600))
    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.streamoff()
        break

cv2.destroyAllWindows()