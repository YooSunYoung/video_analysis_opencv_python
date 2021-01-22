import cv2
import json
import numpy as np


# metadata_path = "data/short_video.json"
metadata_path = "data/sample_video.json"
with open(metadata_path) as f:
    metadata = json.load(f)
video_path = metadata['video_path']
brightness = metadata['brightness']
contour_threshold = metadata['contour_threshold']

capture = cv2.VideoCapture(video_path)

visualize_result = True
# object detection haar cascade classifiers for face and eyes
face_cascade_path = "data/haarcascades/haarcascade_profileface.xml"
eye_cascade_path = "data/haarcascades/haarcascade_righteye_2splits.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

param = cv2.SimpleBlobDetector_Params()
param.blobColor = 255
param.minArea = 25
param.minCircularity = 0.01
param.minConvexity = 0.75
blob_detector = cv2.SimpleBlobDetector_create(param)
if eye_cascade.empty():
    print("Can't load eyes cascade")
    exit()
if face_cascade.empty():
    print("Can't load face cascade")
    exit()


# eye zones
left_eye_x, left_eye_y = 0, 0,
right_eye_x, right_eye_y = 5000, 5000


def adjust_image_size(image, scale_factor):
    dim = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    image = cv2.resize(image, dim)
    return image


scale_factor = 0.3
frame_ranges = [range(1, 100)]

for frame_range in frame_ranges:
    capture.set(1, frame_range[0])
    for frame_number in frame_range:
        ret, frame = capture.read()
        if ret is False:
            print("Failed to open the video")
            exit(FileNotFoundError)
        # Video Size Adjustment
        if scale_factor != 1:
            frame = adjust_image_size(frame, scale_factor)
        # Video Brightness Adjustment
        frame = cv2.addWeighted(frame, 0, frame, brightness, 0)
        # Video Contrast Adjustment should be followed

        # find eyes
        eyes = eye_cascade.detectMultiScale(frame)
        eyes = sorted(eyes, key=lambda x: x[0])
        if len(eyes) > 1:
            left_eye_x = eyes[0][0]
            left_eye_y = eyes[0][1]
            right_eye_x = eyes[1][0] + eyes[1][2]
            right_eye_y = eyes[1][1] + eyes[1][3]
        eyes_zone = frame[left_eye_y:right_eye_y, left_eye_x:right_eye_x]
        roi = frame
        rows, cols, _ = roi.shape
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # find pupils
        gray_roi = cv2.cvtColor(eyes_zone, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        _, threshold = cv2.threshold(gray_roi, contour_threshold, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        circle_keypoints = blob_detector.detect(threshold)
        gray_roi = cv2.drawKeypoints(gray_roi, circle_keypoints, np.array([]), (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pupils = []
        for cnt in contours[:2]:
            (x, y, w, h) = cv2.boundingRect(cnt)
            pupil_zone_threshold = threshold[y-h:y+h, x-w:x+w]
            x = x + left_eye_x
            y = y + left_eye_y
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            pupils.append([y-h, y + h, x-w, x + w])

        if visualize_result:
            cv2.imshow("Roi", roi)
            cv2.imshow("Threshold", threshold)
            cv2.imshow("gray roi", gray_roi)

        key = cv2.waitKey(30)
        if key == 27:
            break

cv2.destroyAllWindows()