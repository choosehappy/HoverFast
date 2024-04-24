# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# System update and install basic tools
RUN apt update && \
    apt upgrade -y && \
    apt install -y software-properties-common wget bzip2 git ninja-build \
    vim nano libjpeg-dev libcairo2-dev libgdk-pixbuf2.0-dev libglib2.0-dev \
    libxml2-dev sqlite3 libopenjp2-7-dev libtiff-dev libsqlite3-dev libhdf5-dev libgl1-mesa-glx \
    build-essential && \
    apt clean

# Install latest openslide version
RUN add-apt-repository ppa:openslide/openslide
RUN apt install -y openslide-tools



# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# Install Python 3.11 using Conda
RUN conda install -c anaconda python=3.11

# Install conda packages
RUN conda install -c anaconda hdf5

# This line removes local apt repo and makes container more compact
RUN rm -rf /var/lib/apt/lists/*

# Install Python requirements
WORKDIR /
COPY ./ /HoverFast
WORKDIR /HoverFast
RUN pip install .

WORKDIR /app
