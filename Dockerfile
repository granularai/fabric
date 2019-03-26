FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

ENV LC_ALL en_US.UTF-8

ENV LANG en_US.UTF-8

ENV LANGUAGE en_US.UTF-8

# Use bash as default shell, rather than sh

ENV SHELL /bin/bash

WORKDIR /code

RUN apt-get update && apt-get -y -qq install libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1

RUN pip install torchvision

RUN pip install --no-cache-dir -U polyaxon-client==0.4.1

RUN pip install polyaxon-client[gcs]

RUN pip install opencv-python

RUN pip install tqdm

RUN pip install pandas

RUN pip install scikit-learn

RUN pip install rasterio

RUN pip install scipy

RUN pip install scikit-image

RUN pip install comet_ml

RUN  pip install --no-cache-dir -U polyaxon-helper

RUN  pip install --no-cache-dir -U polyaxon-gpustat

COPY build /code
