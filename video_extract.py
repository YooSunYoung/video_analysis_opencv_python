import cv2
import numpy as np

video_path = "data/short_video.mkv"
#video_path = "data/sample_video.mkv"
capture = cv2.VideoCapture(video_path)

face_cascade_path = "data/haarcascades/haarcascade_profileface.xml"
eye_cascade_path = "data/haarcascades/haarcascade_righteye_2splits.xml"
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
alpha = 1.2
beta = 1
gamma = 0

contour_threshold = 10

#x1, x2, y1, y2 = 100, 1200, 500, 3000
y1, y2, x1, x2 = 0, 5000, 0, 5000
left_eye_x, left_eye_y, right_eye_x, right_eye_y = 0, 0, 5000, 5000
#y1, y2, x1, x2  = 500, 1300, 800, 3000
#y1, y2, x1, x2  = 0, 1300, 0, 3000
if scale_factor!= 1:
    x1, x2, y1, y2 = int(x1*scale_factor), int(x2*scale_factor), int(y1*scale_factor), int(y2*scale_factor)

while True:
    num_test += 1
    if num_test > 300: break
    ret, frame = capture.read()
    if ret is False:
        print("Failed to open the video")
        break
    # Video Size Adjustment
    if scale_factor != 1:
        dim = (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor))
        frame = cv2.resize(frame, dim)
    # Video Brightness, Contrast Adjustment
    frame = cv2.addWeighted(frame, alpha, frame, beta, gamma)

    eyes = eye_cascade.detectMultiScale(frame)

    eyes = sorted(eyes, key=lambda x: x[0])
    if len(eyes) > 1:
        left_eye_x = eyes[0][0]
        left_eye_y = eyes[0][1]
        right_eye_x = eyes[1][0] + eyes[1][2]
        right_eye_y = eyes[1][1] + eyes[1][3]

    roi = frame
    #roi = frame[y1: y2, x1: x2]
    rows, cols, _ = roi.shape
    eyes_zone = frame[left_eye_y:right_eye_y, left_eye_x:right_eye_x]
    gray_roi = cv2.cvtColor(eyes_zone, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, threshold = cv2.threshold(gray_roi, contour_threshold, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    #print("Detected eyes: "+str(len(eyes)))
    for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #print("Detected pupils: "+str(len(contours)))
    for cnt in contours[:2]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = x + left_eye_x
        y = y + left_eye_y
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    if num_test == 1: cv2.imwrite("test.png", roi)
    key = cv2.waitKey(100)
    if key == 27:
        break

cv2.destroyAllWindows()