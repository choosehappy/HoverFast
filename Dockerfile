# syntax=docker/dockerfile:1
#
# HoverFast with PyTorch + TensorRT (CUDA 13.0): the fastest image. Needs an NVIDIA driver >= 580.
# For older (CUDA 12) drivers use Dockerfile.cu126, which runs plain PyTorch.
#
#   docker build -t hoverfast:latest .
#
# Python comes from Ubuntu; PyTorch and TensorRT bring their own CUDA libraries, so the small
# CUDA "base" image is enough (it still refuses to start on a driver that is too old).

# ---------------------------------------------------------------- build stage
FROM nvidia/cuda:13.0.3-base-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
RUN python3 -m venv /opt/venv && pip install --no-cache-dir uv

WORKDIR /HoverFast
COPY requirements.txt requirements-tensorrt.txt ./
# torch 2.14.1 on PyPI is the CUDA 13.0 build; requirements-tensorrt.txt pins the matching TensorRT.
# Then drop what HoverFast never uses: Triton (torch.compile only) and TensorRT's builder
# resources for Windows targets.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch==2.14.1" && \
    uv pip install -r requirements.txt -r requirements-tensorrt.txt && \
    uv pip uninstall triton && \
    rm -f /opt/venv/lib/python3*/site-packages/tensorrt_libs/*_win_*

COPY ./ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps . && \
    pip uninstall -y uv

# ---------------------------------------------------------------- final image
FROM nvidia/cuda:13.0.3-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
# libsqlite3-mod-spatialite: SpatiaLite output (-d); uses the same system SQLite as Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 libsqlite3-mod-spatialite ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /HoverFast /HoverFast
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

# Check the install without a GPU: SpatiaLite loads, OpenCV imports, and TensorRT's libraries are
# where HoverFast loads them from (otherwise inference would silently fall back to eager PyTorch).
RUN python - <<'EOF'
import os

import cv2
import torch
from hoverfast.common.spatialite import spatialite_available
from hoverfast.models.wsi_model import _find_tensorrt_libs

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| opencv", cv2.__version__)
assert spatialite_available(), "mod_spatialite not loadable"
missing = [p for p in _find_tensorrt_libs() if not os.path.isfile(p)]
assert not missing, f"TensorRT libraries missing: {missing}"
print("TensorRT libraries OK")
EOF

WORKDIR /app
