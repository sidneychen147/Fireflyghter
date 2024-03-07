import cv2
from persondetector import findBody, findFace
from firedetector import FireDetector
import threading
from djitellopy import tello
import requests
from firedetector import FireDetector, set_text

# initialize
drone = tello.Tello()
drone.connect()
drone.streamon()

cap = cv2.VideoCapture('http://169.254.208.144:8000/video_feed')

firedetector = FireDetector()
temp = 0

# get max temp
def get_temp():
    global temp
    while True:
        response = requests.get("http://169.254.208.144:8000/get_temp")
        temp = float(response.text)


# process drone camera
def process_drone_cam(drone, firedetector):
    while True:
        image = drone.get_frame_read().frame
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #findBody(image)
        #findFace(image)
        firedetector.nextframe(image)
        firedetector.detect()
        if temp > 300 and firedetector.detect() == True:
            set_text('Real fire detected')
        if temp < 300:
            set_text('Potential fire detected')
        image = cv2.resize(image, (640, 480))
        cv2.imshow("Drone camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.streamoff()
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
cap.release()
cv2.destroyAllWindows()