import os
import cv2
import json
import csv
import numpy as np
from blink_detection import simple_model
from video_processing_utilities.image_utils import adjust_image_size
from video_processing_utilities.image_utils import adjust_image_brightness
from video_processing_utilities.image_utils import find_eyes_from_image
from video_processing_utilities.image_utils import closed_or_open
from video_processing_utilities.image_utils import find_pupils


# load a video and parameters where we already tuned
metadata_path = "data/videos/sample_video.json"
with open(metadata_path) as f:
    metadata = json.load(f)
video_path = metadata['video_path']
brightness = metadata.get('brightness', 2)
contour_threshold = metadata.get('contour_threshold', 20)
scale_factor = metadata.get('size_scale_factor', 0.5)
minimum_pupil_radius = metadata.get('pupil_minimum_radius', 15)

# load video as cv2 VideoCapture
capture = cv2.VideoCapture(video_path)
# size of the video
video_width = int(capture.get(3) * scale_factor)
video_height = int(capture.get(4) * scale_factor)
# we will update this when we can find eyes and use it for the next frame where we can't find any eyes
tmp_eye_zones = [[0, 0, video_width, video_height], [0, 0, video_width, video_height]]

# cascade classifier to find where eyes are in each frame
haarcasecade_dir = os.path.dirname(cv2.__file__)
eye_cascade_path = os.path.join(haarcasecade_dir, "data", "haarcascade_righteye_2splits.xml")
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
if eye_cascade.empty():
    print("Can't load eyes cascade")
    exit()

# blob detector to find pupil in the threshold image of eyes
param = cv2.SimpleBlobDetector_Params()
param.blobColor = 255
param.minArea = 40
param.minCircularity = 0.01
param.minConvexity = 0.4
blob_detector = cv2.SimpleBlobDetector_create(param)
# binary classification model to determine if an eye is open or closed
model = simple_model()
checkpoint_path = 'data/weight_files/model'
model.load_weights(checkpoint_path)

# save output as a csv file
output_directory = 'result/test/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
f = open(os.path.join(output_directory, 'result.csv'), 'w')
result_csv = csv.writer(f)
result_csv.writerow(['frame_num',
                     'left_eye_x', 'left_eye_y', 'left_eye_r',
                     'right_eye_x', 'right_eye_y', 'right_eye_r'])
# save the result as a video (if needed)
visualize_result = True
visualize_result_realtime = False
if visualize_result:
    video_writer = cv2.VideoWriter_fourcc(*'MJPG')
    video_output = cv2.VideoWriter(os.path.join(output_directory, 'output.avi'),
                                   video_writer, 5,
                                   (2200, 1500))

# close or release all resources
def terminate():
    f.close()
    capture.release()
    cv2.destroyAllWindows()
    if visualize_result:
        video_output.release()

# if you want to use only a small part of the video
frame_ranges = [range(0, 100)]

# for the full video
# frame_ranges = [range(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))]

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
        if visualize_result_realtime:
            cv2.imshow("Video Analysis", frame)
        result_csv.writerow(result_row)

terminate()
