import cv2
import numpy as np

video_path = "data/short_video.mkv"
capture = cv2.VideoCapture(video_path)

face_cascade_path = "data/haarcascades/haarcascade_profileface.xml"
eye_cascade_path = "data/haarcascades/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if eye_cascade.empty():
    print("Can't load eyes cascade")
    exit()
if face_cascade.empty():
    print("Can't load face cascade")
    exit()

num_test = 0
scale_factor = 0.3
x1, x2, y1, y2 = 700, 1300, 900, 1700
if scale_factor!= 1:
    x1, x2, y1, y2 = int(x1*scale_factor), int(x2*scale_factor), int(y1*scale_factor), int(y2*scale_factor)

while True:
    num_test += 1
    if num_test > 20: break
    ret, frame = capture.read()
    if ret is False:
        print("Failed to open the video")
        break
    if scale_factor != 1:
        dim = (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor))
        frame = cv2.resize(frame, dim)

    roi = frame[x1: x2, y1: y2]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()