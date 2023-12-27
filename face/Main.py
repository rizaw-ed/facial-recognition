import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import time


cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

detector = FaceDetector(minDetectionCon=0.75)
output_directory = "face_images"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

while True:
    success, img = cap.read()

    if success:
        img, bboxs = detector.findFaces(img, draw=True)

        cv2.imshow("Image", img)

        if len(bboxs) > 0:
            cv2.imwrite(f"{output_directory}/face_{int(time.time())}.jpg", img)

    if cv2.waitKey(25) & 0xFF == ord('e'):
        break

cv2.destroyAllWindows()
