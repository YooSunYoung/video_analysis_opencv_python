# Video Analysis

The files are structured like below.
```
├── README.md
├── analysis_scripts
│   ├── movement_analysis.py
│   ├── movement_analysis_2.py
│   └── pupil_radius_analysis.py
├── data
├── environment
│   ├── Dockerfile
│   ├── copy_directory_structure.sh
│   └── requirements.txt
├── preprocessing_scripts
│   ├── blink_detection.py
│   ├── jupyter_notebooks
│   ├── video_pupil_tracking.py
│   └── video_tuning.py
├── result
└── video_processing_utilities
    ├── image_utils.py
    └── sample_video.py
```

- `analysis_scripts` has python scripts for plotting some statistics with the result of exctracted data from videos, which is in csv format.
- `data` and `result` directories will be made if you download a sample video or weight files and run the preprocessing scripts.
- `preprocessing_scripts` has python scripts for preprocessing videos. `video_pupil_tracking.py` gives you the csv files containing the position of pupil of each frame of a video.
- `video_processing_utilities` has python modules called from other scripts.
- `environment` has some virtual engironment description files (`Dockerfile` and `requirements.txt` for pip). `copy_directory_structure` is a script for copying a directory structure as well as some fractions of videos under the directory.

## Environment Settings
This project needs docker image which has OpenCV and tensorflow.

ref: [How to use OpenCV docker image](https://learnopencv.com/install-opencv-docker-image-ubuntu-macos-windows/)

Docker image for this repository can be found on the docker hub : 

> [https://hub.docker.com/r/yoosunyoung/opencv_tensorflow](https://hub.docker.com/r/yoosunyoung/opencv_tensorflow)

`docker pull yoosunyoung/opencv_tensorflow:latest`

or

`docker pull yoosunyoung/opencv_tensorflow:tensorflow2`

you can run the container with the command below.
```
docker run
-p 5000:5000 -p 8888:8888
-v /tmp/.X11-unix:/tmp/.X11-unix
-v {PATH_TO_THIS_REPOSITORY}:/root/video-extract
--env DISPLAY=$DISPLAY
--name opencv_tensorflow_2
--ipc host
yoosunyoung/opencv_tensorflow:tensorflow2 /bin/bash
```

+ Image for tensorflow 1 is also available.

`docker pull yoosunyoung/opencv_tensorflow:tensorflow1` 

+ If docker can't connect to the display, try
```
xhost +local:docker
```

## Eyes Closed or Open Classification

`blink_detection.py` has scripts for binary classification training model which detect if the eyes are open or closed.
The link to the data used for the training is in the reference, `pupil annotation dataset`.
Also, I uploaded the trained weight in `data/` directory.

## Video Tuning

Videos needs some tuning regarding brightness and contrastness.
`video_tuning.py` will generate a metadata containing tuning parameters as well as thresholds.
Default parameters are set as below.
What each parameter does can be found in `video_pupil_tracking.py`.
```
contour_threshold:55
video_path:"./data/videos/sample_video.mkv"
contrast:0
size_scale_factor:0.5
pupil_minimum_radius:5
brightness:0.8
```

## Pupil Tracking

There are 3 steps to find the center point of the pupil.
#### 1. Detect Eyes
This is done with opencv cascade eyes detector
#### 2. Determine if Eyes are closed or open 
#### 3. Find Pupils on the Eyes.
The center point of the pupil is obtained from the fragment images of eyes.
The real center point is then calculated regarding the point of the eyes from the original image. 

Result is saved as a csv file.
Each column contains 7 columns like below.

| 'frame_num' | 'left_eye_x' | 'left_eye_y' | 'left_eye_r' | 'right_eye_x' | 'right_eye_y' | 'right_eye_r' |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| frame number | x coordinate of left eye | y coordinate of left eye | radius of left eye | x coornidate of right eye | y coordinate or right eye | radius of right eye|

You can also have a result visualized as a video.
It will be easier if you copy the data directory structure into a `result` directory and then save the csv files under the same directory as the video file.

## Haarcascades Files

You might need to download haar cascade xml files from opencv github repository: [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades).

Default path to the xml files is `data/haarcascades/*.xml`.

Otherwise, you need to change the xml file path.

## Sample Video

You can find the sample video and the metadata file, `sample_video.mkv` and `sample_video.json` in the branch named `sample_video`.

You can download the video by the commands below.

Then you can have the files in your `data/video` directory.

After checkout the remote branch, git might have already added the new files, so check this before you push the commits. 
```
git checkout {your remote repository}/sample_video -- data/videos/sample_video.*
git reset
```
Then you can keep these files for master branch and use for your analysis.
Note that the `sample_video` branch might not be up-to-date.

## References
- Eye motion tracking: [https://pysource.com/2019/01/04/eye-motion-tracking-opencv-with-python/](https://pysource.com/2019/01/04/eye-motion-tracking-opencv-with-python/)
- Face and eye detection: [https://www.hackster.io/trivediswap25/face-and-eye-detection-in-python-using-opencv-5a5b10](https://www.hackster.io/trivediswap25/face-and-eye-detection-in-python-using-opencv-5a5b10)
- Brightness and contrast correction: [https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html](https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
- Pupil Annotation Dataset: [http://mrl.cs.vsb.cz/eyedataset](http://mrl.cs.vsb.cz/eyedataset)
- Binary Classification Tutorial: [https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s/blob/master/Binary_classification.ipynb](https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s/blob/master/Binary_classification.ipynb)
