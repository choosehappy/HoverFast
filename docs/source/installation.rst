Installation Guide
==================

This document provides detailed instructions on how to install HoverFast, a high-performance tool designed for efficient nuclear segmentation in Whole Slide Images (WSIs).

Prerequisites
-------------

Before installing HoverFast, ensure you have the following prerequisites:

- Python 3.9 to 3.12 (3.11 recommended; TensorRT acceleration needs 3.10 or newer)
- An NVIDIA GPU and driver. The PyTorch wheels ship their own CUDA runtime, so no CUDA toolkit is needed.

OpenSlide and HDF5 are installed automatically as Python packages (``openslide-bin`` and ``tables``); no system libraries are required.

Speed and Compatibility
^^^^^^^^^^^^^^^^^^^^^^^

HoverFast runs in plain PyTorch everywhere. TensorRT is optional and gives the maximum speed, but needs a recent driver:

.. list-table::
   :header-rows: 1

   * - Setup
     - NVIDIA driver
     - Speed vs. HoverFast 1.0
   * - PyTorch + TensorRT (CUDA 13)
     - >= 580
     - about 3.9x
   * - PyTorch only, CUDA 13
     - >= 580
     - about 2.5x
   * - PyTorch only, CUDA 12.6
     - >= 525 (CUDA 12 drivers)
     - about 2.5x

Speeds were measured on 8 whole-slide images on an RTX A5000. Run ``nvidia-smi`` to see your driver version.

Using Docker
------------

We recommend using HoverFast within a Docker or Singularity (Apptainer) container for ease of setup and compatibility.

Install Docker
^^^^^^^^^^^^^^^^

If you don't already have Docker installed, follow the instructions on the Docker website (https://docs.docker.com/get-docker/) to install Docker for your operating system.

Install NVIDIA Docker
^^^^^^^^^^^^^^^^^^^^^^^

For GPU support in Docker, you also need to install NVIDIA Container Toolkit. Follow the instructions on the NVIDIA Container Toolkit GitHub page (https://github.com/NVIDIA/nvidia-container-toolkit) to install the necessary components.

Pull Docker Image
^^^^^^^^^^^^^^^^^^^

There are two images. Pick the one that matches your NVIDIA driver (``nvidia-smi`` shows the "Driver Version"):

.. list-table::
   :header-rows: 1

   * - Image
     - Dockerfile
     - Contents
     - NVIDIA driver
   * - ``petroslk/hoverfast:latest``
     - ``Dockerfile``
     - PyTorch + TensorRT, CUDA 13.0 (fastest)
     - >= 580
   * - ``petroslk/hoverfast:cu126``
     - ``Dockerfile.cu126``
     - PyTorch only, CUDA 12.6
     - >= 525

.. code-block:: sh

    docker pull petroslk/hoverfast:latest     # or: petroslk/hoverfast:cu126

The TensorRT image will not start on a driver older than 580; Docker then reports ``unsatisfied condition: cuda>=13.0``. Use the ``cu126`` image instead.

To build an image yourself instead of pulling it:

.. code-block:: sh

    git clone https://github.com/choosehappy/HoverFast.git
    cd HoverFast
    docker build -t hoverfast:latest .                        # PyTorch + TensorRT
    docker build -f Dockerfile.cu126 -t hoverfast:cu126 .     # PyTorch only

The TensorRT build fails if the TensorRT libraries are not usable. To use TensorRT, build an engine inside the container once per machine with ``HoverFast build`` and pass it to inference with ``-e``.

Run HoverFast with Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After pulling the Docker image, you can run HoverFast using the following command:

.. code-block:: sh

    docker run -it --gpus all --shm-size=16g -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi /app/*.svs -o /app/output/

This command runs HoverFast in a Docker container with GPU support, mounting the directory `/path/to/slides/` on your host to `/app` in the container, and outputs the results to the `/app/output/` directory. The `--shm-size=16g` flag is required because `infer_wsi` uses PyTorch `DataLoader` workers that exchange tensors through shared memory; Docker's 64 MB default `/dev/shm` is too small (`--ipc=host` is an alternative).

Containers run as root by default, so output files are owned by root. To write them as your own user, add ``--user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -e HOME=/tmp``. The ``/etc/passwd`` mount is required for TensorRT, because ``torch-tensorrt`` looks up the current user at import and fails with ``KeyError: 'getpwuid(): uid not found'`` otherwise. Singularity maps your user automatically.

Using Singularity
-----------------

For systems that support Singularity (Apptainer), you can pull the HoverFast container as follows:

Install Singularity
^^^^^^^^^^^^^^^^^^^^
If Singularity is not already installed on your system, you can follow the installation guide on the Singularity website (https://sylabs.io/guides/3.0/user-guide/installation.html).

Pull Singularity Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To pull the Singularity container, run the following command:

.. code-block:: sh

    singularity pull docker://petroslk/hoverfast:latest     # or: docker://petroslk/hoverfast:cu126

Run HoverFast with Singularity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After pulling the container, you can run HoverFast using the following command:

.. code-block:: sh

    singularity exec --nv hoverfast_latest.sif HoverFast infer_wsi /path/to/wsis/*.svs -o /path/to/output/

This command runs HoverFast in a Singularity container with GPU support, processing WSIs located in `/path/to/wsis/` and saving the results to `/path/to/output/`.

Local Installation with Conda
------------------------------

For local installations, especially for development purposes, follow these steps:

Install Conda
^^^^^^^^^^^^^^^^^

If you don't already have Conda installed, you can download and install Miniconda or Anaconda from the Conda website (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Create and Activate a Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, create and activate a Conda environment for HoverFast:

.. code-block:: sh

    conda create -n HoverFast python=3.11
    conda activate HoverFast

Install HoverFast
^^^^^^^^^^^^^^^^^

Next, clone the HoverFast repository and install it:

.. code-block:: sh

    git clone https://github.com/choosehappy/HoverFast.git
    cd HoverFast
    pip install .

This runs inference in plain PyTorch. pip installs the default PyTorch build from PyPI, which currently targets CUDA 13 and needs a driver >= 580. With an older (CUDA 12) driver, install PyTorch for CUDA 12.6 first, then HoverFast:

.. code-block:: sh

    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install .

Optional: TensorRT for Maximum Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorRT makes inference about 1.5x faster than plain PyTorch (about 3.9x faster than HoverFast 1.0). It needs an NVIDIA driver >= 580. Install the pinned TensorRT stack (torch 2.14, torch-tensorrt 2.14, TensorRT 11.1, listed in ``requirements-tensorrt.txt``):

.. code-block:: sh

    pip install ".[tensorrt]"

Then build an engine for your GPU with ``HoverFast build`` and pass it to inference with ``-e``. Engines are specific to the GPU, driver and TensorRT version they were built with.

Optional: SpatiaLite Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing a SpatiaLite database instead of JSON (``infer_wsi -d``) needs the ``mod_spatialite`` SQLite extension, which pip cannot install:

.. code-block:: sh

    conda install -c conda-forge libspatialite

On Ubuntu without Conda, use ``sudo apt install libsqlite3-mod-spatialite`` instead.

Verify Installation
^^^^^^^^^^^^^^^^^^^

To verify the installation, you can run a simple command to check if HoverFast is installed correctly:

.. code-block:: sh

    HoverFast --help

Advanced Installation Options
-----------------------------

For users who need more control over the installation process, here are some advanced options:

Installing from Source without Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to install HoverFast without using Conda and are on a Linux Ubuntu machine, you can follow these steps:

1. Clone the repository:

    .. code-block:: sh

        git clone https://github.com/choosehappy/HoverFast.git
        cd HoverFast

2. Create a virtual environment and activate it:

    .. code-block:: sh

        python -m venv venv
        source venv/bin/activate

3. Install HoverFast (add ``[tensorrt]`` for TensorRT acceleration, see above):

    .. code-block:: sh

        pip install .

4. Optional, only for SpatiaLite output (``infer_wsi -d``):

    .. code-block:: sh

        sudo apt install libsqlite3-mod-spatialite


Version
^^^^^^^
You can check the version of HoverFast that you are currently running:

    .. code-block:: sh

        HoverFast --version


Troubleshooting
---------------

If you encounter issues during installation, here are some common solutions:

CUDA Not Detected

Ensure that your NVIDIA driver is installed and up to date, then check that PyTorch can see the GPU:

.. code-block:: sh

    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

TensorRT Engine Not Used

If inference prints ``Falling back to eager PyTorch``, the engine could not be loaded. Check that you installed ``pip install ".[tensorrt]"`` (not unpinned ``tensorrt``/``torch-tensorrt`` packages), and rebuild the engine with ``HoverFast build`` on the machine where inference runs.

Dependency Conflicts

If you encounter dependency conflicts, consider creating a new Conda environment or virtual environment to isolate the installation.

Insufficient Permissions

For Docker and Singularity installations, you may need administrative privileges. Ensure you have the necessary permissions or contact your system administrator.

Additional Resources

For further assistance, refer to the following resources:

- HoverFast GitHub Repository (https://github.com/choosehappy/HoverFast.git)
- Docker Documentation (https://docs.docker.com/)
- Singularity Documentation (https://sylabs.io/docs/)
- Conda Documentation(https://docs.conda.io/)

By following these detailed instructions, you should be able to successfully install and run HoverFast on your system. If you have any questions or need further assistance, please refer to the official documentation or contact the support team.
