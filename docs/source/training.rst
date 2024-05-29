Train Module
============

This module contains functions to train a model from pytable files of segmented nuclei. Before being able to train, you might need to generate a dataset from HoverNet.
We provide a docker container to do this since local HoverNet installation can be tricky.

Dataset Generation
------------------

Structure Your Data Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Structure your data directory as follows:

.. code-block:: none

    └── dir
        config.ini
        └── slides/
        ├── slide_1.svs
        ├── ...
        └── slide_n.svs


Generate Dataset
^^^^^^^^^^^^^^^^^

To generate the dataset, run the following command:

.. code-block:: sh

    docker run --gpus all -it -v /path/to/dir/:/HoverFastData petroslk/data_generation_hovernet:latest hoverfast_data_generation -c '/HoverFastData/config.ini'

This should generate two files in the directory called `data_train.pytable` and `data_test.pytable`. You can use these to train the model.

Config File for Data Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of what the `config.ini` file should look like:

.. code-block:: ini

    [General]
    dataset = data
    classes = 0, 1
    gpu_id = 0
    seed = 3981674709

    [Dataset_train_test]
    test_set_size = 0.2
    num_tiles = 10
    tile_size = 1024
    level = 0
    mask_level = 3
    mask_th = 0.2
    save_tiles = False
    rois = False

Explanation:

- **[General]**:

  - **dataset**: Name of the dataset.
  - **classes**: Classes to segment.
  - **gpu_id**: GPU ID to use for data generation.
  - **seed**: Random seed for reproducibility.

- **[Dataset_train_test]**:

  - **test_set_size**: Proportion of the dataset to use for testing.
  - **num_tiles**: Number of tiles to select from each WSI.
  - **tile_size**: Size of each tile.
  - **level**: Level of the WSI to use.
  - **mask_level**: Level of the mask to use.
  - **mask_th**: Threshold for the mask.
  - **save_tiles**: Whether to save the extracted tiles.
  - **rois**: Whether to use ROIs.


Train Parser
------------

The `train` parser allows you to configure various parameters for training the model. Below are the details of each argument:

- **dataname** (positional argument): Dataset name, corresponds to the pytables name under the following format: `(dataname)_(phase).pytables`.
- **-o, --outdir**: Output directory path for tensorboard and trained model. Default is `./output/`.
- **-p, --dataset_path**: Path to the directory that contains the pytables. Default is `./`.
- **-b, --batch_size**: Number of workers for the dataloader. Default is `5`.
- **-n, --n_worker**: Number of workers for the dataloader. Default is `min(batch_size, os.cpu_count())`.
- **-e, --epoch**: Number of epochs. Default is `100`.
- **-d, --depth**: Depth of the model. Default is `3`.
- **-w, --width**: Width of the model. Defines the number of filters in the first layer (`2**w`) with an exponential growth rate respective to the depth of the model. Default is `4`.

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

This example demonstrates the basic usage of the `train` command with minimal arguments, using default settings.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset, which corresponds to the pytables name.
- Dataset Path: Directory that contains the pytables.
- Output directory: Directory to save TensorBoard logs and trained model.

Custom Batch Size
^^^^^^^^^^^^^^^^^

In this example, we specify a custom batch size for the dataloader.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -b 10 -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset.
- Dataset Path: Directory that contains the pytables.
- Batch Size: Set to 10 workers for the dataloader.
- Output directory: Directory to save TensorBoard logs and trained model.

Custom Number of Workers
^^^^^^^^^^^^^^^^^^^^^^^^

Setting a custom number of workers for the dataloader to optimize data loading.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -n 8 -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset.
- Dataset Path: Directory that contains the pytables.
- Number of Workers: Set to 8 workers for the dataloader.
- Output directory: Directory to save TensorBoard logs and trained model.

Adjusting Model Depth
^^^^^^^^^^^^^^^^^^^^^

Adjusting the depth of the model for more complex training scenarios.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -d 5 -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset.
- Dataset Path: Directory that contains the pytables.
- Model Depth: Set the depth of the model to 5.
- Output directory: Directory to save TensorBoard logs and trained model.

Adjusting Model Width
^^^^^^^^^^^^^^^^^^^^^

Customizing the width of the model, which defines the number of filters in the first layer.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -w 6 -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset.
- Dataset Path: Directory that contains the pytables.
- Model Width: Set the width of the model to 6.
- Output directory: Directory to save TensorBoard logs and trained model.

Custom Epochs
^^^^^^^^^^^^^

Setting a custom number of epochs for the training process.

.. code-block:: sh

    HoverFast train dataset_name -p /path/to/dataset -e 200 -o /path/to/outdir

Explanation:

- Dataset: Name of the dataset.
- Dataset Path: Directory that contains the pytables.
- Epochs: Set the number of epochs to 200.
- Output directory: Directory to save TensorBoard logs and trained model.

Functions
---------

The `train` module contains the following functions:

.. dropdown:: Click to show/hide functions

    .. automodule:: hoverfast.training_utils
        :members:
        :undoc-members:
        :show-inheritance:

.. dropdown:: Click to show/hide functions

    .. automodule:: hoverfast.augment
        :members:
        :undoc-members:
        :show-inheritance:

.. dropdown:: Click to show/hide functions

    .. automodule:: hoverfast.hoverfast
        :members:
        :undoc-members:
        :show-inheritance:
