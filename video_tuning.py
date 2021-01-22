import cv2
import json


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
    if brightness == 'autotune' and contrast == 'autotune':
        pass
    if brightness == 'autotune':
        pass
    if contrast == 'autotune':
        pass
    size_scale_factor = kwargs.get("size_scale_factor", 0.3)
    contour_threshold = kwargs.get("contour_threshold", 10)
    if contour_threshold == 'autotune':
        pass
    metadata = {'video_path': video_path,
                'brightness': brightness,
                'contrast': contrast,
                'size_scale_factor': size_scale_factor,
                'contour_threshold': contour_threshold}
    return metadata


if __name__ == "__main__":
    video_path = "data/short_video.mkv"
    metadata_path = video_path.replace("mkv", "json")
    metadata = generate_metadata(video_path)
    metadata_file = open(metadata_path, 'w')
    json.dump(metadata, metadata_file)
