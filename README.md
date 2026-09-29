![HoverFast Logo](docs/source/_static/images/hoverfast_logo.png)

![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/choosehappy/HoverFast)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Docker Pulls](https://img.shields.io/docker/pulls/petroslk/hoverfast)
![GitHub issues](https://img.shields.io/github/issues/choosehappy/HoverFast)
![GitHub stars](https://img.shields.io/github/stars/choosehappy/HoverFast)

Welcome to the official repository of HoverFast, a high-performance tool designed for efficient nuclear segmentation in Whole Slide Images (WSIs).

## Overview

HoverFast utilizes advanced computational methods to facilitate rapid and accurate segmentation of nuclei within large histopathological images, supporting research and diagnostics in medical imaging. For more info on the inner workings of HoverFast, do not hesitate to go over our [paper](https://joss.theoj.org/papers/10.21105/joss.07022#)

## Repository Structure

The `hoverfast/` package is organized by concept:

```
hoverfast/
├── main.py       # CLI entry point (infer_wsi, infer_roi, train, build)
├── models/       # network architecture, model loading and TensorRT engine
├── wsi/          # whole-slide inference pipeline, post-processing and image I/O
├── roi/          # region-of-interest inference pipeline
├── training/     # training loop and data augmentation
└── common/       # shared SpatiaLite and stain-deconvolution helpers
```

Comprehensive unit tests live under `tests/unit/`, with end-to-end CLI tests in `tests/`.

## Documentation

An overview of the documentation is provided in this repository, but for more details, please visit the full [official documentation](https://hoverfast.readthedocs.io/en/latest/)

## Installation

### Prerequisites

- Python 3.11.5
- CUDA installation for GPU support (version > 12.1.0)

### Using Docker

We recommend using HoverFast within a Docker or Singularity (Apptainer) container for ease of setup and compatibility.

You can either pull the pre-built image from Docker Hub or build it locally from the provided `Dockerfile`.

- **Option 1: Pull the Pre-built Docker Image (recommended)**
```
docker pull petroslk/hoverfast:latest
```

- **Option 2: Build the Docker Image from the Dockerfile**

Clone the repository and build the image locally. This compiles an NVIDIA CUDA 12.1 runtime image, installs the Python dependencies and the HoverFast package, and tags the result `hoverfast:latest`:
```
git clone https://github.com/choosehappy/HoverFast.git
cd HoverFast
docker build -t hoverfast:latest .
```

The build does not require a GPU (only running inference or training does), but it requires a Docker installation with BuildKit enabled. The `Dockerfile` uses BuildKit cache mounts (`--mount=type=cache`), which recent Docker versions enable by default. If your Docker daemon does not, prefix the build with `DOCKER_BUILDKIT=1`:
```
DOCKER_BUILDKIT=1 docker build -t hoverfast:latest .
```

Once built, use `hoverfast:latest` in place of `petroslk/hoverfast:latest` in the run commands below.

### Using Singularity

For systems that support Singularity (Apptainer), you can pull the HoverFast container as follows:

- **Pull Singularity Container**
```
singularity pull docker://petroslk/hoverfast:latest
```

### Local Installation with Conda

For local installations, especially for development purposes, follow these steps:

- **Create and activate a Conda environment**
```
conda create -n HoverFast python=3.11
conda activate HoverFast
```

- **Install HoverFast**
```
git clone https://github.com/choosehappy/HoverFast.git
cd HoverFast
pip install .
```

### Verify Installation

- **Check the installed version**
```
HoverFast --version
```

## Usage

All tasks can be run natively, with a Docker container, or with a Singularity (Apptainer) container. Only the command prefix differs:

| Method | Command prefix |
|---|---|
| Local | `HoverFast` |
| Docker | `docker run -it --gpus all -v /path/to/data:/app petroslk/hoverfast:latest HoverFast` |
| Singularity | `singularity exec --nv hoverfast_latest.sif HoverFast` |

For Docker, mount the directory containing your input data to `/app` (the container's working directory) and write outputs to a path inside that mount. For Singularity, the container accesses the host filesystem directly, so paths are used as-is. If you built the image locally, substitute `hoverfast:latest` for `petroslk/hoverfast:latest`.

> **Shared memory (`--shm-size`).** `infer_wsi` and `train` use PyTorch `DataLoader` workers, which hand tensors to the main process through shared memory. Docker's default `/dev/shm` is only 64 MB, which is exhausted immediately and fails with `No space left on device`. Add `--shm-size=16g` (shown in those examples below) or `--ipc=host`. `build` and `infer_roi` do not use `DataLoader` and need no change. Lowering `-n/--n_process` also reduces shared-memory demand.

### Whole Slide Image Inference (`infer_wsi`)

- **Basic usage (local)**
```
HoverFast infer_wsi path/to/slides/*.svs -o hoverfast_output
```

- **Docker**
```
docker run -it --gpus all --shm-size=16g -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi /app/*.svs -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -o /app/hoverfast_output
```

- **Singularity**
```
singularity exec --nv hoverfast_latest.sif HoverFast infer_wsi path/to/slides/*.svs -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -o hoverfast_output
```

- **With binary masks**

Although HoverFast does have a simple threshold based tissue detection, we highly recommend the use of QC tools such as HistoQC for generating tissue masks to avoid computing on artefactual regions and reducing computation time.
You can give the path to the directory where the masks are stored. HoverFast will search for a mask with the same name as the slide with a .png extension.

```
HoverFast infer_wsi path/to/slides/*.svs -b path/to/masks/ -o hoverfast_output
```

- **For IHC Nuclear DAB stain**

If your IHC DAB stain is nuclear, you should use the ihc_dab flag to segment nuclei. If your IHC DAB stain is not nuclear, regular H&E segmentation might be a better option.

```
HoverFast infer_wsi path/to/slides/*.svs -b path/to/masks/ -st ihc_dab -o hoverfast_output
```

- **Using a compiled TensorRT engine**

Build an engine for the current GPU first (see [Building a TensorRT Engine](#building-a-tensorrt-engine-build)), then pass it with `-e`. If `-e` is omitted, `./unet_trt.ts` is used when present:

```
HoverFast infer_wsi path/to/slides/*.svs -e unet_trt.ts -o hoverfast_output
```

For the full list of arguments, see the [infer_wsi documentation](https://hoverfast.readthedocs.io/en/latest/infer_wsi.html).

### Region of Interest Inference (`infer_roi`)

- **Basic usage (local)**
```
HoverFast infer_roi path/to/rois/*png -o hoverfast_output
```

- **Docker**
```
docker run -it --gpus all -v /path/to/rois/:/app petroslk/hoverfast:latest HoverFast infer_roi /app/*png -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -o /app/hoverfast_output
```

- **Singularity**
```
singularity exec --nv hoverfast_latest.sif HoverFast infer_roi path/to/rois/*png -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -o hoverfast_output
```

- **Using a compiled TensorRT engine**

As for `infer_wsi`, pass `-e` to point at an engine built for the current GPU. If omitted, `./unet_trt.ts` is used when present:

```
HoverFast infer_roi path/to/rois/*png -e unet_trt.ts -o hoverfast_output
```

For the full list of arguments, see the [infer_roi documentation](https://hoverfast.readthedocs.io/en/latest/infer_roi.html).

### Building a TensorRT Engine (`build`)

TensorRT engines are machine-specific and must be compiled on the GPU where inference will run. The `build` sub-command produces an engine tuned for the current GPU from a `.safetensors` model:

- **Local**
```
HoverFast build -m hoverfast_crosstissue_best_model.safetensors -o unet_trt.ts
```

- **Docker**
```
docker run -it --gpus all -v /path/to/models/:/app petroslk/hoverfast:latest HoverFast build -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -o /app/unet_trt.ts
```

Inference then picks up the engine in one of two ways:

1. Point at it explicitly with `-e/--engine_path`:
```
HoverFast infer_wsi path/to/slides/*.svs -e unet_trt.ts -o hoverfast_output
```
In the Docker example above the engine was written to `/app/unet_trt.ts`, i.e. `/path/to/models/unet_trt.ts` on the host. Re-mount that directory and pass the container path:
```
docker run -it --gpus all --shm-size=16g -v /path/to/models/:/app petroslk/hoverfast:latest HoverFast infer_wsi /app/*.svs -m /HoverFast/hoverfast_crosstissue_best_model.safetensors -e /app/unet_trt.ts -o /app/hoverfast_output
```
The same `-e` flag is available on `infer_roi`.

2. Or place the engine at `./unet_trt.ts` in the working directory and omit `-e`; that path is used by default.

If no compatible engine is found, inference transparently falls back to eager PyTorch and prints a build hint, so building is optional but recommended for maximum throughput.

### Training

To train HoverFast on your data, you may need to generate a local dataset first using our provided container.

#### Generating a Local Dataset

- **Structure your data directory**

```
└── dir
    config.ini
    └── slides/
    ├── slide_1.svs
    ├── ...
    └── slide_n.svs
```

- **Generate Dataset**

```
docker run --gpus all -it -v /path/to/dir/:/HoverFastData petroslk/data_generation_hovernet:latest hoverfast_data_generation -c '/HoverFastData/config.ini'
```

This should generate two files in the directory called "data_train.pytable" and "data_test.pytable". You can use these to train the model.

#### Training the Model

The training batch size can be adjusted based on available VRAM.

- **Local**
```
HoverFast train data -o training_model -p /path/to/pytable_files/ -b 16 -n 20 -e 100
```

- **Docker**
```
docker run -it --gpus all --shm-size=16g -v /path/to/pytables/:/app petroslk/hoverfast:latest HoverFast train data -o /app/training_metrics -p /app -b 16 -n 20 -e 100
```

- **Singularity**
```
singularity exec --nv hoverfast_latest.sif HoverFast train data -o training_metrics -p /path/to/pytables/ -b 16 -n 20 -e 100
```

## Testing

Since HoverFast utilizes GPU for almost all tasks, most tests have to be run locally using pytest.

First, install pytest:

```
pip install pytest
```

Then, you can just run the following command inside the HoverFast repo:

```
pytest -vv
```
Note that the first time you run these, the infer_wsi test can take longer since the slide will be downloaded locally

For more detailed instructions, including setting up your environment and running specific tests, please refer to the [testing documentation](https://hoverfast.readthedocs.io/en/latest/unit_testing.html)

## How to Cite HoverFast

If you use HoverFast in your research, please cite our paper:

```bibtex
@article{Liakopoulos2024, doi = {10.21105/joss.07022}, url = {https://doi.org/10.21105/joss.07022}, year = {2024}, publisher = {The Open Journal}, volume = {9}, number = {101}, pages = {7022}, author = {Petros Liakopoulos and Julien Massonnet and Jonatan Bonjour and Medya Tekes Mizrakli and Simon Graham and Michel A. Cuendet and Amanda H. Seipel and Olivier Michielin and Doron Merkler and Andrew Janowczyk}, title = {HoverFast: an accurate, high-throughput, clinically deployable nuclear segmentation tool for brightfield digital pathology images}, journal = {Journal of Open Source Software} }
```
By citing HoverFast, you help us to continue our research and development. Thank you for your support!


## Authors

- **Julien Massonnet** - [JulienMassonnet](https://github.com/JulienMassonnet)
- **Petros Liakopoulos**  - [petroslk](https://github.com/petroslk)
- **Andrew Janowczyk**  - [choosehappy](https://github.com/choosehappy)
