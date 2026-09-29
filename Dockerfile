# syntax=docker/dockerfile:1

# NVIDIA's CUDA base image. CUDA 13.0 requires a host driver >= 580.
FROM nvidia/cuda:13.0.3-runtime-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# System update and install basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common wget bzip2 git ninja-build \
    vim nano libjpeg-dev libcairo2-dev libgdk-pixbuf2.0-dev libglib2.0-dev \
    libxml2-dev sqlite3 libopenjp2-7-dev libtiff-dev libsqlite3-dev libhdf5-dev libgl1-mesa-glx \
    spatialite-bin libsqlite3-mod-spatialite build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install latest openslide version
RUN add-apt-repository ppa:openslide/openslide && \
    apt-get install -y openslide-tools && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# Install Python 3.11 using Conda
RUN conda install -c anaconda python=3.11.5

# Install conda packages
RUN conda install -c anaconda hdf5
RUN conda install -c conda-forge libstdcxx-ng

# This line removes local apt repo and makes container more compact
RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this heavy layer is cached unless
# requirements.txt changes (the source tree changes on every build).
WORKDIR /HoverFast
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install uv && \
    uv pip install -r requirements.txt --system

# Install the HoverFast package itself (dependencies already present above).
COPY ./ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps . --system

WORKDIR /app
