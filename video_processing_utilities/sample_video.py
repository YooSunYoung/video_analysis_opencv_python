import argparse
import os
import cv2


def sample_video(input_video, output_video, frame_num=100):
    capture = cv2.VideoCapture(input_video)
    video_width = int(capture.get(3))
    video_height = int(capture.get(4))

    extension = output_video.split('.')[-1]
    codec = "MJPG"
    if extension=='h264':
        output_video = output_video.split('.')
        output_video[-1] = 'mkv'
        output_video = '.'.join(output_video)
        print('h264 not supported, changed output video path as {}'.format(output_video))

    video_writer = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_video, video_writer, 5, (video_width, video_height))

    def terminate():
        capture.release()
        cv2.destroyAllWindows()
        video.release()

    for i in range(frame_num):
        ret, frame = capture.read()
        if ret is False:
            print("Failed to open the video")
            terminate()
            exit(FileNotFoundError)
        video.write(frame)

    terminate()


def sample_image(input_video, output_image):
    capture = cv2.VideoCapture(input_video)

    def terminate():
        capture.release()
        cv2.destroyAllWindows()

    ret, frame = capture.read()
    if ret is False:
        print("Failed to open the video")
        terminate()
        exit(FileNotFoundError)
    cv2.imwrite(output_image, frame)
    terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sampling videos")
    parser.add_argument('--input', type=str,
                        help='input video file')
    parser.add_argument('--output', type=str,
                        help='output video file')
    parser.add_argument('--frame_num', type=int,
                        help='number of frames you want to sample from the original video')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    frame_num = args.frame_num
    if os.path.exists(output_file):
        print("{} already exists. Please delete the file first or choose another name.".format(output_file))
        exit(AssertionError)
    sample_video(input_file, output_file, frame_num)
