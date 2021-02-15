FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN add-apt-repository -y ppa:graphics-drivers/ppa
RUN apt-get update -y
RUN apt-get install -y \
    python3.7 \
    python3.7-dev \
    cmake \
    apt-utils
RUN apt-get install -y python3-pip
RUN python3.7 -m pip install futures==2.2.0 \
                scikit-build \
                tensorflow-gpu==2.0.0a0 \
                h5py==2.10.0
RUN python3.7 -m pip install opencv-python==3.4.2.17
RUN apt update && apt install -y libsm6 libxext6
RUN apt install -y libxrender-dev

RUN echo 'alias python="python3.7"' >> ~/.bashrc
RUN echo 'alias pip="python3.7 -m pip"' >> ~/.bashrc
