.. HoverFast documentation master file, created by
   sphinx-quickstart on Tue May 28 11:55:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HoverFast's documentation!
=====================================

.. image:: _static/images/hoverfast_logo.png
   :alt: HoverFast logo

Introduction
============

Welcome to the official documentation for HoverFast, a high-performance tool designed for efficient nuclear segmentation in Digital Pathology images.

Overview
--------

HoverFast was designed to provide fast and accurate nuclear segmentation for brightfield digital pathology images. It can take full advantage of multiple CPU cores
and provide an output that you can drag-and-drop onto QuPath to visualize your segmentation results.

Complementary tools
-------------------

As useful as HoverFast is for nuclear segmentation, it can greatly benefit with the use of other tools to help delinate the part of the tissue that should be analyzed. By default, HoverFast
will just use a white threshold to detect tissue regions. This can often be suboptimal. For example, in an ideal situation, you would not want to spend time segmenting nuclei on artefacts such as coverslips or dust. 
For that reason, we highly recommend looking into other tools such as HistoQC (https://github.com/choosehappy/HistoQC) or HistoBlur (https://github.com/choosehappy/HistoBlur) to remove those by generating accurate tissue masks.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   infer_wsi
   infer_roi
   training
   unit_testing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
