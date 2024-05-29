Unit testing
============

This section contains the necessary information to run the unit tests for HoverFast. The tests are designed to ensure the functionality and reliability of the software. Since many tests require GPU support, 
it is necessary to run the tests on a local machine with the necessary hardware.

Setting Up the Environment
---------------------------

Before running the tests, you need to set up your environment. Follow the steps from the installation section.

Running the Tests
-----------------

Once your environment is set up, you can run the tests. Ensure that your GPU is available and properly configured.

1. **Running All Tests**:

   You can run all the tests using `pytest`. The `-vv` option increases verbosity and `--tb=long` ensures that the full traceback is shown if any test fails.

   .. code-block:: bash

      pytest -vv --tb=long

2. **Understanding the Test Structure**:

   The tests are located in the `tests` directory and are organized into different files based on the functionality they test. Here are some examples:

   - `test_installation.py`: Tests related to the installation and basic functionality of HoverFast.
   - `test_wsi_inference.py`: Tests for Whole Slide Image (WSI) inference.
   - `test_roi_inference.py`: Tests for Region of Interest (ROI) inference.
   - `test_training.py`: Tests for the training functionality.

3. **Sample Test Command**:

   Here's an example of how to run a specific test file, for instance, the WSI inference tests:

   .. code-block:: bash

      pytest tests/test_wsi_inference.py -vv --tb=long

4. **Output and Logs**:

   During test execution, you will see detailed output in the terminal. This includes information about which tests passed, which failed, and any errors encountered. The `--tb=long` option ensures that full traceback information is provided, which is useful for debugging.

Special Considerations
----------------------

Since most tests require GPU support, it is important to ensure that your environment is properly configured to utilize the GPU. This includes having the appropriate CUDA version installed and ensuring that your system recognizes the GPU.

- **Check GPU Availability**:

  You can verify that your GPU is available and recognized by your system using the following command:

  .. code-block:: bash

     nvidia-smi

  This should display information about your GPU.


Conclusion
----------

Running unit tests is a critical part of ensuring the reliability and correctness of HoverFast. By following the steps outlined above, you can set up your environment and run the tests to validate the functionality of the software. For any issues or further assistance, refer to the detailed output provided by `pytest` or consult the project's documentation and support resources.

