import sys, os
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


def extract_pupil_from_video(video_path, visualize_result=False,
                             scale_factor=0.5,
                             brightness=2.2,
                             contour_threshold=12,
                             # face_cascade_path="data/haarcascades/haarcascade_profileface.xml",
                             eye_cascade_path="/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_righteye_2splits.xml",
                             blob_detector_param=None,
                             minimum_pupil_radius=15,
                             frame_ranges=None,
                             model=None,
                             checkpoint_path='./data/weight_files/model',
                             output_directory='result/'):
    capture = cv2.VideoCapture(video_path)
    print(eye_cascade_path)
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

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
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
    haarcasecade_dir = os.path.dirname(cv2.__file__)
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
                             frame_ranges=[range(0, 100)],
                             eye_cascade_path=os.path.join(haarcasecade_dir, "data", "haarcascade_righteye_2splits.xml"),
                             blob_detector_param=param,
                             output_directory='result/test/'
                             )
