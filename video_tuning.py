import cv2
import json
import sys


def detect_eyes(frame, eye_classifier):
    pass


def brightness_tuning(frame, contrast=0):
    pass


def contrast_tuning(frame, brightness=0):
    pass


def brightness_contrast_tuning(frame, brightness=0, contrast=0):
    pass


def generate_metadata(video_path, **kwargs):
    capture = cv2.VideoCapture(video_path)
    ret, frame = None, None
    ret_tmp, frame_tmp = capture.read()
    if ret_tmp is False:
        print("Failed to load the video from the file path, " + video_path)
        return False
    if 'reference_frame_number' in kwargs.keys():
        capture.set(1, kwargs.get('reference_frame_number'))
        ret, frame = capture.read()
    if ret is False:
        print("Failed to read the reference frame.")
        print("Continue with the first frame.")
        ret, frame = ret_tmp, frame_tmp

    brightness = kwargs.get("brightness", 0)
    contrast = kwargs.get("contrast", 0)
    size_scale_factor = kwargs.get("size_scale_factor", 0.3)
    pupil_minimum_radius = kwargs.get("pupil_minimum_radius", 15)
    contour_threshold = kwargs.get("contour_threshold", 12)
    
    if brightness == 'autotune' and contrast == 'autotune':
        pass
    if brightness == 'autotune':
        pass
    if contrast == 'autotune':
        pass
    if pupil_minimum_radius == 'autotune':
        pass
    if contour_threshold == 'autotune':
        pass
    metadata = {'video_path': video_path,
                'brightness': brightness,
                'contrast': contrast,
                'size_scale_factor': size_scale_factor,
                'pupil_minimum_radius': pupil_minimum_radius,
                'contour_threshold': contour_threshold}
    return metadata


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) > 1:
        video_path = arguments[1]
    metadata_path = "data/videos/test.json"
    metadata = generate_metadata(video_path)
    metadata_file = open(metadata_path, 'w+')
    json.dump(metadata, metadata_file)
