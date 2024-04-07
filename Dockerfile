FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/open-mmlab/mmdetection.git /TRECVID \
    && cd /TRECVID \
    && pip install -r requirements.txt