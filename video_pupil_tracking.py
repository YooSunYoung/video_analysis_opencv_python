import cv2
import json
import math
import csv
import numpy as np
import tensorflow as tf
from blink_detection import simple_model

metadata_path = "data/short_video.json"
# metadata_path = "data/sample_video.json"
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
param.minArea = 40
param.minCircularity = 0.01
param.minConvexity = 0.4
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


scale_factor = 0.5
frame_ranges = [range(0, 150)]

X = tf.placeholder(tf.float32, [None, 50, 50, 1], name="normalized_gray_image")
checkpoint_path = 'data/model'
output_0 = simple_model(X, training=False)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
saver.restore(sess, checkpoint_path)

tmp_eye_zones = [[0, 0, 3000, 3000], [0, 0, 3000, 3000]]


def adjust_image_size(image, scale_factor=1):
    if scale_factor != 1:
        dim = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        image = cv2.resize(image, dim)
    return image


def adjust_image_brightness(image, brightness=1):
    image = cv2.addWeighted(image, 0, frame, brightness, 0)
    return image


def find_eyes_from_image(image, eye_detector):
    eyes = eye_detector.detectMultiScale(image)
    if len(eyes) == 0:
        return None
    eyes = sorted(eyes, key=lambda x: x[2])
    eyes = sorted(eyes, key=lambda x: x[0])
    eye_zones = []
    for eye in eyes:
        x1 = eye[0]
        x2 = eye[0]+eye[2]
        y1 = eye[1]
        y2 = eye[1]+eye[3]
        eye_zones.append(tuple([x1, y1, x2, y2]))
    return eye_zones


def closed_or_open(eye_zones, image,  sess, output_0):
    gray_eye_zones = []
    for x1, y1, x2, y2 in eye_zones:
        eye_img = image[y1:y2, x1:x2].copy()
        eye_img = eye_img[int(eye_img.shape[1]/3):eye_img.shape[1], 0:eye_img.shape[0]]
        gray_eye_zone = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye_zone = cv2.GaussianBlur(gray_eye_zone, (7, 7), 0)
        gray_eye_zone = cv2.resize(gray_eye_zone, (50, 50), interpolation=cv2.INTER_AREA)
        gray_eye_zone = np.reshape(gray_eye_zone, (50, 50, 1))
        gray_eye_zone = gray_eye_zone / 225
        gray_eye_zones.append(gray_eye_zone)
    eyes_closed = sess.run(output_0, feed_dict={X: gray_eye_zones})
    for il, eye_label in enumerate(eyes_closed):
        eye_label = 0 if eye_label < 0.5 else 1
        eyes_closed[il] = eye_label
    return eyes_closed


def find_pupil(eye, contour_threshold, blob_detector, minimum_area=20):
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold = cv2.threshold(gray_eye, contour_threshold, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(contours) == 0 or cv2.contourArea(contours[0]) < minimum_area:
        return find_pupil(eye, contour_threshold + 2, blob_detector, minimum_area=minimum_area)

    circle_keypoints = blob_detector.detect(threshold)
    circle_keypoints = sorted(circle_keypoints, key=lambda x: x.size, reverse=True)
    x, y, r = None, None, None
    if len(circle_keypoints) > 0:
        circle_keypoints = [circle_keypoints[0]]
        x = int(circle_keypoints[0].pt[0])
        y = int(circle_keypoints[0].pt[1])
        r = int(circle_keypoints[0].size)
        if r >= 35:
            gray_eye = cv2.drawKeypoints(gray_eye, circle_keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if None in [x, y, r] or r < 35:
        cnt = contours[0]
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)
        r = int(r)
        # (x, y, w, h) = cv2.boundingRect(cnt)
        # x = int(x + w/2)
        # y = int(y + h/2)
        # r = int((math.sqrt(w * w + h * h)) / 2)
        if len(gray_eye.shape) < 3:
            gray_eye = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2RGB)
        gray_eye = cv2.circle(gray_eye, (x, y), r, (0, 0, 255))
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    gray_eye = np.concatenate((gray_eye, threshold), axis=1)
    gray_eye = cv2.circle(gray_eye, (x, y), 0, (0, 0, 255), thickness=2)
    return gray_eye, [x, y, r]


def find_pupils(eye_zones, image, contour_threshold, blob_detector, minimum_area=10, visualize=True):
    pupils = []
    for (x1, y1, x2, y2) in eye_zones:
        eye_img = image[y1:y2, x1:x2].copy()
        gray_eye, [x, y, r] = find_pupil(eye_img, contour_threshold, blob_detector, minimum_area)
        x = x + x1  # real x coordinate on original frame
        y = y + y1  # real y coordinate on original frame
        pupils.append([gray_eye, x, y, r])
    if visualize is True:
        eyes = pupils[0][0]
        if len(pupils) > 1:
            eye1 = pupils[0][0]
            eye2 = pupils[1][0]
            width = max(eye1.shape[0], eye2.shape[0])
            eye1 = cv2.copyMakeBorder(eye1, 0, width - eye1.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            eye2 = cv2.copyMakeBorder(eye2, 0, width - eye2.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            eyes = np.concatenate((eye1, eye2), axis=1)
        return pupils, eyes
    return pupils


video_writer = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('output.avi', video_writer, 5, (2200, 1500))

f = open("result.csv", 'w')
result_csv = csv.writer(f)

for frame_range in frame_ranges:
    capture.set(1, frame_range[0])
    for frame_number in frame_range:
        ret, frame = capture.read()
        if ret is False:
            print("Failed to open the video")
            exit(FileNotFoundError)
        # Video Size Adjustment
        frame = adjust_image_size(frame, scale_factor)
        # Video Brightness Adjustment
        frame = adjust_image_brightness(frame, brightness)
        # Video Contrast Adjustment should be followed

        # find eyes
        eye_zones = find_eyes_from_image(frame, eye_cascade)
        # see if eyes are closed
        if len(eye_zones) > 1:
            tmp_eye_zones = eye_zones  # if eyes were not found, assume eyes are still at the same place.
        eyes_closed = closed_or_open(tmp_eye_zones, frame, sess, output_0)
        if 0 in eyes_closed or len(eye_zones) < 2:
            cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        #if 0 not in eyes_closed and len(eye_zones) > 0:
        result_row = [frame_number]
        if not (0 in eyes_closed) and len(eye_zones) > 1:
        # if len(eye_zones) > 1:
            # find pupils
            pupils, eyes = find_pupils(eye_zones, frame, contour_threshold, blob_detector, minimum_area=100)
            # draw eyes and pupils
            tmp = []
            if len(pupils) == 0:
                tmp.append(None)
            for (x1, y1, x2, y2), (img, x, y, r) in zip(eye_zones[:2], pupils[:2]):
                if r < 20:
                    cv2.putText(frame, "Can't find pupil", (150, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    tmp = [None]
                    break
                tmp.append(x)
                tmp.append(y)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # frame = cv2.circle(frame, (x, y), r, (0, 0, 255))
                frame = cv2.circle(frame, (x, y), 0, (0, 0, 255), thickness=5)
            width = max(frame.shape[1], eyes.shape[1])
            eyes = cv2.copyMakeBorder(eyes, 0, 0, 0, width-eyes.shape[1], cv2.BORDER_CONSTANT)
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, width-frame.shape[1], cv2.BORDER_CONSTANT)
            frame = np.concatenate((frame, eyes), axis=0)
            for i in tmp:
                result_row.append(i)
        else:
            result_row.append('None')
        frame = cv2.copyMakeBorder(frame, 0, 1500-frame.shape[0], 0, 2200-frame.shape[1], cv2.BORDER_CONSTANT)
        if visualize_result:
        #    cv2.imshow("Original", frame)
            video_output.write(frame)
        # key = cv2.waitKey(20)
        # if key == 27:
        #   break
        result_csv.writerow(result_row)

f.close()
capture.release()
video_output.release()
cv2.destroyAllWindows()