FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install -y \
    python3.7 \
    python3.7-dev \
    cmake \
    apt-utils
RUN alias python='python3.7'
RUN apt-get install -y python3-pip
RUN alias pip='python3.7 -m pip'
RUN python3.7 -m pip install futures==2.2.0 \
                scikit-build \
                tensorflow-gpu==2.0.0a0
RUN python3.7 -m pip install opencv-python

RUN echo 'alias python="python3.7"' >> ~/.bashrc
RUN echo 'alias pip="python3.7 -m pip"' >> ~/.bashrc

RUN python3.7 -m pip install h5py==2.10.0