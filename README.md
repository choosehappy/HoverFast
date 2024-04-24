# HoverFast :microscope: :fast_forward:

Welcome to the official documentation for HoverFast, a high-performance tool designed for efficient nuclei segmentation in Whole Slide Images (WSIs).

## Overview

HoverFast utilizes advanced computational methods to facilitate rapid and accurate segmentation of nuclei within large histopathological images, supporting research and diagnostics in medical imaging.

## Installation

### Prerequisites

- Python 3.11
- CUDA installation for GPU support (version > 12.1.0)

### Using Docker

We recommend using HoverFast within a Docker or Singularity (Apptainer) container for ease of setup and compatibility.

- **Pull Docker Image**
```
docker pull petroslk/hoverfast:latest
```

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
git clone https://github.com/JulienMassonnet/HoverFastMultipross.git
cd HoverFastMultipross
pip install .
```

## Usage

### Command Line Interface

HoverFast offers a versatile CLI for processing WSIs, ROIs, and for model training.

#### For Whole Slide Images (WSI) Inference

- **Basic Usage**
```
HoverFast infer_wsi --help
```

- **Example Command without binary masks**
```
HoverFast infer_wsi path/to/slides/*.svs -m hoverfast_crosstissue_best_model.pth -n 20 -o hoverfast_output
```

- **Example Command with binary masks**

```
HoverFast infer_wsi path/to/slides/*.svs -b masks/ -m hoverfast_crosstissue_best_model.pth -n 20 -o hoverfast_output
```

#### For Region of Interest (ROI) Inference

- **Example Command**

```
HoverFast infer_roi path/to/rois/*png -m hoverfast_pretrained_pannuke.pth -o hoverfast_output
```

### Using Containers

Containers simplify the deployment and execution of HoverFast across different systems. We highly recommend using them!

#### Docker

- **Run Inference**

```
docker run -it --gpus all -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi *svs -m /HoverFast/hoverfast_crosstissue_best_model.pth -o hoverfast_results
```

#### Singularity

- **Run Inference**

```
singularity exec --nv hoverfast_latest.sif HoverFast infer_wsi /path/to/wsis/*svs -m /HoverFast/hoverfast_crosstissue_best_model.pth -o hoverfast_results
```

## Training

To train HoverFast on your data, you may need to generate a local dataset first using our provided container.

### Generating Local Dataset

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

- **Train model**

You can use these to train the model as follows:

The training batch size can be adjusted based on available VRAM

```
HoverFast train data -l training_metrics -p /path/to/pytable_files/ -b 16 -n 20 -e 100
```

```
docker run -it --gpus all -v /path/to/pytables/:/app petroslk/hoverfast:latest HoverFast train data -l training_metrics -p /app -b 16 -n 20 -e 100
```

```
singularity exec --nv hoverfast_latest.sif HoverFast train data -l training_metrics -p /path/to/pytables/ -b 16 -n 20 -e 100
```


## Authors

- **Julien Massonnet** - [JulienMassonnet](https://github.com/JulienMassonnet)
- **Petros Liakopoulos**  - [petroslk](https://github.com/petroslk)
- **Andrew Janowczyk**  - [choosehappy](https://github.com/choosehappy)
