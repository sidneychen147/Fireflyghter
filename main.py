"""
main.py
main program; controls drone and handles fire/person detection
"""
from djitellopy import tello
import cv2
import numpy as np
from time import sleep
from persondetector import findBody, findFace
from firedetector import FireDetector
import requests
import threading


drone = tello.Tello()
drone.connect()
drone.streamon()

cap = cv2.VideoCapture('http://11.26.15.61:8000/video_feed')

firedetector = FireDetector()

temp = 0

def process_drone_cam(drone, firedetector):
    start = 0
    while True:
        image = drone.get_frame_read().frame
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        firedetector.nextframe(image)
        fire_detected, coordinates = firedetector.detect()
        if temp > 300 and fire_detected:
            firedetector.set_displaytext('Real fire detected')
        if temp < 300:
            firedetector.set_displaytext('Potential fire detected')
        left_right_velocity = 0
        for_back_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0
        if fire_detected:
            x, y, x2, y2 = coordinates
            xcenter = 320
            ycenter = 240
            objcenterx = (x + x2) // 2
            objcentery = (y + y2) // 2
            errorx = objcenterx - xcenter
            errory = objcentery - ycenter
            if abs(errorx) > 125:
                if errorx < 0:
                    yaw_velocity = -75
                    print("move left")
                else:
                    yaw_velocity = 75
                    print("move right")
            if abs(errory) > 125:
                if errory < 0:
                    up_down_velocity = 40
                    print("move up")
                else:
                    up_down_velocity = -40
                    print("move down")
        elif temp < 300:
            left_right_velocity = 0
            for_back_velocity = 20
            up_down_velocity = 0
            yaw_velocity = 0
        else:
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
        image = cv2.resize(image, (640, 480))
        cv2.imshow("Drone camera", image)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            drone.takeoff()
            drone.move_up(50)
            start = 1
        if drone.send_rc_control:
            print(f"Battery Level: {drone.get_battery()}")
            drone.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.streamoff()
            drone.land()
            break

# process thermal camera

def process_therm_cam(cap, firedetector):
    global temp
    frame_counter = 0
    while True:
        _, imagef = cap.read()
        frame_counter += 1
        if frame_counter % 5 == 0:
            print(temp)
            imagef = cv2.resize(imagef, (640, 480))
            cv2.imshow("Thermal camera", imagef)
            print(f"Battery Level: {drone.get_battery()}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# get max temp
def get_temp():
    global temp
    while True:
        response = requests.get("http://11.26.15.61:8000/get_temp")
        temp = float(response.text)


# Start the threads
drone_thread = threading.Thread(target=process_drone_cam, args=(drone, firedetector))
therm_thread = threading.Thread(target=process_therm_cam, args=(cap, firedetector))
temp_thread = threading.Thread(target=get_temp)

# Start the threads
drone_thread.start()
therm_thread.start()
temp_thread.start()

# Wait for threads to finish
drone_thread.join()
therm_thread.join()
temp_thread.join()

# Clean up
drone.streamoff()
drone.land()
cap.release()
cv2.destroyAllWindows()
