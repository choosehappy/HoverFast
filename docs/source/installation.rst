Installation Guide
==================

This document provides detailed instructions on how to install HoverFast, a high-performance tool designed for efficient nuclear segmentation in Whole Slide Images (WSIs).

Prerequisites
-------------

Before installing HoverFast, ensure you have the following prerequisites:

- Python 3.11
- CUDA installation for GPU support (version > 12.1.0)
- HDF5 (available here https://www.hdfgroup.org/downloads/hdf5/)
- Openslide (available here https://openslide.org/download/)

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

To pull the latest Docker image, run the following command:

.. code-block:: sh

    docker pull petroslk/hoverfast:latest

Run HoverFast with Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After pulling the Docker image, you can run HoverFast using the following command:

.. code-block:: sh

    docker run -it --gpus all -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi /app/*.svs -o /app/output/

This command runs HoverFast in a Docker container with GPU support, mounting the directory `/path/to/slides/` on your host to `/app` in the container, and outputs the results to the `/app/output/` directory.

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

    singularity pull docker://petroslk/hoverfast:latest

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

Install CUDA Toolkit
^^^^^^^^^^^^^^^^^^^^^

If you plan to use GPU support, install the CUDA toolkit. Follow the instructions on the NVIDIA CUDA Toolkit website (https://developer.nvidia.com/cuda-downloads) to install the appropriate version for your system.

Install HDF5 and Openslide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the required dependencies HDF5 and Openslide, run the following commands:

.. code-block:: sh

    conda install -c anaconda hdf5
    conda install -c conda-forge openslide

Install HoverFast
^^^^^^^^^^^^^^^^^

Next, clone the HoverFast repository and install it:

.. code-block:: sh

    git clone https://github.com/choosehappy/HoverFast.git
    cd HoverFast
    pip install .

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

3. Install HDF5 and Openslide:

    .. code-block:: sh

        apt-get install libhdf5-serial-dev
        apt-get install openslide-tools

4. Install the required dependencies:

    .. code-block:: sh

        pip install -r requirements.txt


5. Install HoverFast:

    .. code-block:: sh

        pip install .


Version
^^^^^^^
You can check the version of HoverFast that you are currently running:

    .. code-block:: sh

        HoverFast --version


Troubleshooting
---------------

If you encounter issues during installation, here are some common solutions:

CUDA Not Detected

Ensure that your CUDA installation is correctly configured and that your GPU drivers are up to date. You can verify the CUDA installation by running:

.. code-block:: sh

    nvcc --version

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
