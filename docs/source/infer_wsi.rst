Infer WSI Module
================

This module contains functions to perform nuclei detection on Whole Slide Images (WSI).

Infer WSI Parser
----------------

The `infer_wsi` parser allows you to configure various parameters for nuclei detection. Below are the details of each argument:

- **slide_folder** (positional argument): Input filename pattern.
- **-o, --outdir**: Output directory. Default is `./output/`.
- **-b, --binmask_dir**: Quality control mask directory. Default is `None`.
- **-m, --model_path**: Path to the pre-trained model. Default is `./hoverfast_crosstissue_best_model.pth`.
- **-l, --magnification**: Magnification to work on. Default is `40`.
- **-p, --poly_simplify**: Float representing the tolerance for simplifying the polygons. Default is `6`.
- **-s, --size_threshold**: Minimum size threshold for nuclei area in square micrometers. Default is `5`.
- **-r, --batch_mem**: Maximum batches saved in memory (RAM). Default is `500`.
- **-g, --batch_gpu**: Target batch size for GPU. Default is calculated based on available GPU VRAM.
- **-t, --tile_size**: Region size to compute on. Default is `1024`.
- **-n, --n_process**: Number of workers for multiprocessing. Default is `os.cpu_count()`.

Usage Examples
------------------

Basic Usage
^^^^^^^^^^^^^^^^

This example demonstrates the basic usage of the `infer_wsi` command with minimal arguments, processing a batch of slides using default settings.
This assumes that the default pretrained model is in the current working directory


.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/ with the .svs extension.
- Output: Results saved in the directory /path/to/output/.
- Uses default values for all other parameters.

Custom Model Path
^^^^^^^^^^^^^^^^^^^^^^

In this example, we specify a custom path for the pre-trained model.


.. code-block:: sh
    
    HoverFast infer_wsi /path/to/slides/*.svs -m /path/to/custom_model.pth -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- Model: Custom model located at /path/to/custom_model.pth.
- Output: Results saved in /path/to/output/.



Using Binary Masks
^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to use quality control masks during inference. These can be generated with tools like HistoQC and HistoBlur


.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -b /path/to/masks/ -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- Masks: Quality control masks located at /path/to/masks/ (masks should have the same name as the slides with a .png extension).
- Output: Results saved in /path/to/output/.

Adjusting Magnification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjusting the magnification for specific analysis requirements. Note that the provided pretrained model has been trained at 0.25mpp, therefore lower magnifications may not work optimally.


.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -l 20 -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- Magnification: 20x instead of the default 40x.
- Output: Results saved in /path/to/output/.

Custom GPU Batch Sizes
^^^^^^^^^^^^^^^^^^^^^^

Setting custom sizes GPU batches to optimize VRAM consumption. By default, HoverFast tries to maximize VRAM usage, but sometimes,
using less vram can be useful.

.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -g 3 -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- GPU Batch Size: GPU processes 50 batches at a time.
- Output: Results saved in /path/to/output/.

Using Multiprocessing
^^^^^^^^^^^^^^^^^^^^^

Utilizing multiple CPU cores for faster processing. HoverFast can highly benefit from using more CPU threads.


.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -n 20 -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- CPU Cores: Use 20 threads for multiprocessing.
- Output: Results saved in /path/to/output/.

Simplifying Polygon Output
^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjusting the polygon simplification to reduce output file size and speed up file writing. This can be useful if you do not require the nuclei
contours and only need nuclei centroids. 

.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -p 8 -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- Polygon Simplification: Set the tolerance to 8 for simplifying the contour of polygons.
- Output: Results saved in /path/to/output/.

Thresholding Nuclei Size
^^^^^^^^^^^^^^^^^^^^^^^^
Setting a minimum size threshold for detected nuclei to filter out small detections. By default, objects below 5 square micrometers will be filtered.
This can be lowered or increased accordingly.

.. code-block:: sh

    HoverFast infer_wsi /path/to/slides/*.svs -s 8 -o /path/to/output/


Explanation:

- Input: Slides located at /path/to/slides/.
- Size Threshold: Minimum nuclei area of 8 square micrometers.
- Output: Results saved in /path/to/output/.



