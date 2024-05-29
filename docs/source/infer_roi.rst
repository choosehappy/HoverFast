Infer ROI Module
================

This module contains functions to perform nuclei detection on Regions of Interest (ROIs).

Infer ROI Parser
----------------

The `infer_roi` parser allows you to configure various parameters for nuclei detection. Below are the details of each argument:

- **slide_folder** (positional argument): Input filename pattern.
- **-o, --outdir**: Output directory. Default is `./output/`.
- **-m, --model_path**: Path to the pre-trained model. Default is `./hoverfast_crosstissue_best_model.pth`.
- **-p, --poly_simplify**: Float representing the tolerance for simplifying the polygons. Default is `6`.
- **-s, --size_threshold**: Minimum size threshold for nuclei area in square micrometers. Default is `85`.
- **-r, --batch_mem**: Maximum batches saved in memory (RAM). Default is `500`.
- **-g, --batch_gpu**: Target batch size for GPU. Default is calculated based on available GPU VRAM.
- **-n, --n_process**: Number of workers for multiprocessing. Default is `os.cpu_count()`.
- **-w, --width**: Width of the cells border shown in the overlay. Default is `2`.
- **-c, --color**: Color of polygon shown on the overlay. Refer to Matplotlib colors for more information. Default is `limegreen`.

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

This example demonstrates the basic usage of the `infer_roi` command with minimal arguments, processing a batch of ROIs using default settings.
This assumes that the default pretrained model is in the current working directory

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/ with the .png extension.
- Output: Results saved in the directory /path/to/output/.
- Uses default values for all other parameters.

Custom Model Path
^^^^^^^^^^^^^^^^^

In this example, we specify a custom path for the pre-trained model.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -m /path/to/custom_model.pth -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- Model: Custom model located at /path/to/custom_model.pth.
- Output: Results saved in /path/to/output/.

Adjusting Polygon Simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjusting the polygon simplification tolerance to reduce output file size and speed up file writing. This can be useful if you do not require detailed nuclei contours.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -p 8 -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- Polygon Simplification: Set the tolerance to 8 for simplifying the contour of polygons.
- Output: Results saved in /path/to/output/.

Thresholding Nuclei Size
^^^^^^^^^^^^^^^^^^^^^^^^

Setting a minimum size threshold for detected nuclei to filter out small detections. This can be adjusted according to your analysis requirements.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -s 10 -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- Size Threshold: Minimum nuclei area of 10 square micrometers.
- Output: Results saved in /path/to/output/.

Custom GPU Batch Sizes
^^^^^^^^^^^^^^^^^^^^^^

Setting custom sizes for GPU batches to optimize VRAM consumption. By default, HoverFast tries to maximize VRAM usage, but sometimes, using less VRAM can be useful.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -g 3 -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- GPU Batch Size: GPU processes 3 batches at a time.
- Output: Results saved in /path/to/output/.

Using Multiprocessing
^^^^^^^^^^^^^^^^^^^^^

Utilizing multiple CPU cores for faster processing. HoverFast can highly benefit from using more CPU threads.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -n 20 -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- CPU Cores: Use 20 threads for multiprocessing.
- Output: Results saved in /path/to/output/.

Adjusting Overlay Width and Color
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Customizing the width and color of the cells border shown in the overlay. This can be useful for better visual representation of detected nuclei.

.. code-block:: sh

    HoverFast infer_roi /path/to/rois/*.png -w 3 -c red -o /path/to/output/

Explanation:

- Input: ROIs located at /path/to/rois/.
- Width: Set the border width to 3 pixels.
- Color: Set the border color to red.
- Output: Results saved in /path/to/output/.

