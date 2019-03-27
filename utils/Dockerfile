FROM tensorflow/tensorflow:1.4.1-py3

ENV LC_ALL en_US.UTF-8

ENV LANG en_US.UTF-8

ENV LANGUAGE en_US.UTF-8

# Use bash as default shell, rather than sh

ENV SHELL /bin/bash

WORKDIR /code

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda update conda \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH $PATH:/opt/conda/bin

RUN pip3 install --no-cache-dir -U polyaxon-client==0.4.1

RUN pip3 install polyaxon-client[gcs]

RUN conda install git && git clone https://a9842bc4083b3ca64eebd5af853f42d097165bd7:@github.com/granularai/moonshot

RUN pip3 install ./moonshot

# RUN conda install -y -c conda-forge --file moonshot/alt_requirements.txt

RUN conda install -y -c conda-forge geopandas

COPY build /code
