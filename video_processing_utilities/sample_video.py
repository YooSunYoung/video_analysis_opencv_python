import argparse
import os
import cv2


def sample_video(input_video, output_video, frame_num=100):
    capture = cv2.VideoCapture(input_video)
    video_width = int(capture.get(3))
    video_height = int(capture.get(4))

    video_writer = cv2.VideoWriter_fourcc(*'MJPG')
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
