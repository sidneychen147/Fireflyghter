import cv2
from persondetector import findBody, findFace
from firedetector import FireDetector, findfire_haar

cap = cv2.VideoCapture(0)
firedetector = FireDetector()

while True:
    _, image = cap.read()
    findBody(image)
    findFace(image)

    firedetector.nextframe(image)
    firedetector.detect()

    image = cv2.resize(image, (800, 600))
    cv2.imshow("Output", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
