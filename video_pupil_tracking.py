import csv
import cv2
import json
import numpy as np
import os
import sys
from blink_detection import simple_model


def adjust_image_size(image, scale_factor=1):
    if scale_factor != 1:
        dim = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        image = cv2.resize(image, dim)
    return image


def adjust_image_brightness(image, brightness=1):
    image = cv2.addWeighted(image, 0, image, brightness, 0)
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
        x2 = eye[0] + eye[2]
        y1 = eye[1]
        if y1 > image.shape[0] * 0.5: continue
        y2 = eye[1] + eye[3]
        eye_zones.append(tuple([x1, y1, x2, y2]))
    return eye_zones


def closed_or_open(eye_zones, image, model):
    gray_eye_zones = []
    for x1, y1, x2, y2 in eye_zones:
        eye_img = image[y1:y2, x1:x2].copy()
        eye_img = eye_img[int(eye_img.shape[1] / 3):eye_img.shape[1], 0:eye_img.shape[0]]
        eye_img = cv2.flip(eye_img, 1)
        gray_eye_zone = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye_zone = cv2.GaussianBlur(gray_eye_zone, (7, 7), 0)
        gray_eye_zone = cv2.resize(gray_eye_zone, (50, 50), interpolation=cv2.INTER_AREA)
        gray_eye_zone = np.reshape(gray_eye_zone, (50, 50, 1))
        gray_eye_zone = gray_eye_zone + gray_eye_zone * 0.4
        gray_eye_zone = gray_eye_zone / 225
        gray_eye_zones.append(gray_eye_zone)
    eyes_closed = model.predict([gray_eye_zones])
    for il, eye_label in enumerate(eyes_closed):
        # print(eye_label)
        # cv2.imshow('test', gray_eye_zones[il])
        # cv2.waitKey(200)
        eye_label = 0 if eye_label < 0.8 else 1
        eyes_closed[il] = eye_label
    return eyes_closed


def find_pupil(eye, contour_threshold, blob_detector, minimum_area=20, minimum_radius=10):
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
        if r >= minimum_radius:
            gray_eye = cv2.drawKeypoints(gray_eye, circle_keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if None in [x, y, r] or r < minimum_radius:
        cnt = contours[0]
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)
        r = int(r)
        if len(gray_eye.shape) < 3:
            gray_eye = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2RGB)
        gray_eye = cv2.circle(gray_eye, (x, y), r, (0, 0, 255))
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    gray_eye = np.concatenate((gray_eye, threshold), axis=1)
    gray_eye = cv2.circle(gray_eye, (x, y), 0, (0, 0, 255), thickness=2)
    return gray_eye, [x, y, r]


def find_pupils(eye_zones, image, contour_threshold, blob_detector, minimum_area=10, visualize=True, minimum_radius=10):
    pupils = []
    for (x1, y1, x2, y2) in eye_zones:
        eye_img = image[y1:y2, x1:x2].copy()
        gray_eye, [x, y, r] = find_pupil(eye_img, contour_threshold, blob_detector, minimum_area, minimum_radius)
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


def extract_pupil_from_video(video_path, visualize_result=False,
                             scale_factor=0.5,
                             brightness=2.2,
                             contour_threshold=12,
                             # face_cascade_path="data/haarcascades/haarcascade_profileface.xml",
                             eye_cascade_path="data/haarcascades/haarcascade_righteye_2splits.xml",
                             blob_detector_param=None,
                             minimum_pupil_radius=15,
                             frame_ranges=None,
                             model=None,
                             checkpoint_path='./data/model',
                             output_directory='result/'):
    capture = cv2.VideoCapture(video_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    if eye_cascade.empty():
        print("Can't load eyes cascade")
        exit()
    # face_cascade = cv2.CascadeClassifier(face_cascade_path)
    # if face_cascade.empty():
    #     print("Can't load face cascade")
    #     exit()

    blob_detector = None
    if blob_detector_param is None:
        param = cv2.SimpleBlobDetector_Params()
        param.blobColor = 255
        param.minArea = 40
        param.minCircularity = 0.01
        param.minConvexity = 0.4
        blob_detector = cv2.SimpleBlobDetector_create(param)
    else:
        blob_detector = cv2.SimpleBlobDetector_create(blob_detector_param)

    if model is None:
        model = simple_model()
    model.load_weights(checkpoint_path)

    video_width = int(capture.get(3) * scale_factor)
    video_height = int(capture.get(4) * scale_factor)
    tmp_eye_zones = [[0, 0, video_width, video_height], [0, 0, video_width, video_height]]

    if visualize_result:
        video_writer = cv2.VideoWriter_fourcc(*'MJPG')
        video_output = cv2.VideoWriter(os.path.join(output_directory, 'output.avi'),
                                       video_writer, 5,
                                       (2200, 1500))

    f = open(os.path.join(output_directory, 'result.csv'), 'w')
    result_csv = csv.writer(f)
    result_csv.writerow(['frame_num',
                         'left_eye_x', 'left_eye_y', 'left_eye_r',
                         'right_eye_x', 'right_eye_y', 'right_eye_r'])

    def terminate():
        f.close()
        capture.release()
        cv2.destroyAllWindows()
        if visualize_result:
            video_output.release()

    if frame_ranges is None:
        frame_ranges = [range(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))]
    for frame_range in frame_ranges:
        capture.set(1, frame_range[0])
        for frame_number in frame_range:
            ret, frame = capture.read()
            if ret is False:
                print("Failed to open the video")
                terminate()
                exit(FileNotFoundError)
            # Video Size Adjustment
            frame = adjust_image_size(frame, scale_factor)
            # Video Brightness Adjustment
            frame = adjust_image_brightness(frame, brightness)
            # Video Contrast Adjustment should be followed

            # find eyes
            eye_zones = find_eyes_from_image(frame, eye_cascade)
            # if you could find two eyes
            if eye_zones is not None and len(eye_zones) > 1:
                tmp_eye_zones = eye_zones  # if eyes were not found, assume eyes are still at the same place.
            eyes_closed = closed_or_open(tmp_eye_zones, frame, model)
            # if 0 in eyes_closed or len(eye_zones) < 2:
            for ir, result in enumerate(eyes_closed):
                cv2.putText(frame, str(result), (50 * ir, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            result_row = [frame_number]
            if (eye_zones is None or len(eye_zones) < 2) and 0 in eyes_closed:
                cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                # find pupils
                pupils_eyes = find_pupils(eye_zones, frame, contour_threshold,
                                          blob_detector, minimum_area=100, visualize=visualize_result,
                                          minimum_radius=minimum_pupil_radius)
                if visualize_result:
                    pupils, eyes = pupils_eyes
                else: pupils = pupils_eyes
                # draw eyes and pupils
                tmp = []
                for (x1, y1, x2, y2), (img, x, y, r) in zip(eye_zones[:2], pupils[:2]):
                    if r < minimum_pupil_radius:
                        cv2.putText(frame, "Can't find pupil", (150, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        break
                    tmp.append(x)
                    tmp.append(y)
                    tmp.append(r)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (x, y), 0, (0, 0, 255), thickness=5)
                width = max(frame.shape[1], eyes.shape[1])
                eyes = cv2.copyMakeBorder(eyes, 0, 0, 0, width - eyes.shape[1], cv2.BORDER_CONSTANT)
                frame = cv2.copyMakeBorder(frame, 0, 0, 0, width - frame.shape[1], cv2.BORDER_CONSTANT)
                frame = np.concatenate((frame, eyes), axis=0)
                for i in tmp:
                    result_row.append(i)

            frame = cv2.copyMakeBorder(frame, 0, 1500 - frame.shape[0],
                                       0, 2200 - frame.shape[1], cv2.BORDER_CONSTANT)
            if visualize_result:
                video_output.write(frame)
            result_csv.writerow(result_row)

    terminate()


if __name__ == "__main__":
    arguments = sys.argv
    metadata_path = "data/videos/short_video.json"
    # metadata_path = "data/videos/sample_video.json"
    if len(arguments) > 1:
        metadata_path = arguments[1]
    with open(metadata_path) as f:
        metadata = json.load(f)
    param = cv2.SimpleBlobDetector_Params()
    param.blobColor = 255
    param.minArea = 40
    param.minCircularity = 0.01
    param.minConvexity = 0.4
    model = simple_model()
    extract_pupil_from_video(metadata['video_path'],
                             visualize_result=True,
                             model=model,
                             brightness=metadata.get('brightness', 2),
                             contour_threshold=metadata.get('contour_threshold', 20),
                             scale_factor=metadata.get('size_scale_factor', 0.5),
                             minimum_pupil_radius=metadata.get('pupil_minimum_radius', 10),
                             # frame_ranges=[range(0, 100)],
                             blob_detector_param=param,
                             output_directory='result/0/'
                             )
