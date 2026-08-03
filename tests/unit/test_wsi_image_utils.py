#!/usr/bin/env python3
"""Unit tests for WSI image utilities (hoverfast/wsi_image_utils.py)."""

import gzip
import os
import tempfile

import numpy as np
from hoverfast.wsi_image_utils import magnification_from_mpp, rgba2rgb, save_poly, writer
from PIL import Image


class TestMagnificationFromMpp:
    def test_standard_20x(self):
        mag = magnification_from_mpp(0.49)
        assert abs(mag - 20.0) < 1.0

    def test_standard_40x(self):
        mag = magnification_from_mpp(0.245)
        assert abs(mag - 40.0) < 1.0

    def test_high_mag(self):
        mag = magnification_from_mpp(0.1225)
        assert abs(mag - 80.0) < 1.0

    def test_low_mag(self):
        mag = magnification_from_mpp(0.98)
        assert abs(mag - 10.0) < 1.0

    def test_positive_output(self):
        for mpp in [0.1, 0.5, 1.0]:
            assert magnification_from_mpp(mpp) > 0


class TestRgba2rgb:
    def test_rgba_to_rgb_conversion(self):
        rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        rgb = rgba2rgb(rgba)
        assert rgb.mode == "RGB"
        assert rgb.size == (64, 64)

    def test_fully_opaque_unchanged(self):
        solid = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
        rgb = rgba2rgb(solid)
        pixel = np.array(rgb)[0, 0]
        assert pixel[0] == 0
        assert pixel[1] == 255
        assert pixel[2] == 0

    def test_fully_transparent_becomes_white(self):
        transparent = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        rgb = rgba2rgb(transparent)
        pixel = np.array(rgb)[0, 0]
        assert pixel[0] == 255
        assert pixel[1] == 255
        assert pixel[2] == 255

    def test_output_size_preserved(self):
        img = Image.new("RGBA", (128, 64), (100, 100, 100, 200))
        rgb = rgba2rgb(img)
        assert rgb.size == (128, 64)


class TestSavePoly:
    def test_basic_polygon(self):
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        centroid = np.array([5, 5])
        feature = save_poly(poly, centroid)
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["object_type"] == "cell"

    def test_custom_object_class(self):
        poly = np.array([[0, 0], [10, 0], [10, 10]])
        centroid = np.array([5, 5])
        obj_class = {"name": "Tumor", "colorRGB": -1}
        feature = save_poly(poly, centroid, obj_class)
        assert feature["properties"]["classification"]["name"] == "Tumor"

    def test_centroid_integers(self):
        poly = np.array([[0, 0], [10, 0], [10, 10]])
        centroid = np.array([5.7, 3.2])
        feature = save_poly(poly, centroid)
        cx, cy = feature["geometry"]["centroid"]
        assert isinstance(cx, int)
        assert isinstance(cy, int)

    def test_coordinates_closed_ring(self):
        poly = np.array([[0, 0], [10, 0], [10, 10]])
        centroid = np.array([5, 5])
        feature = save_poly(poly, centroid)
        coords = feature["geometry"]["coordinates"]
        assert coords[0] == coords[-1]


class TestWriter:
    def test_writer_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.json.gz")
            from multiprocessing import Manager

            manager = Manager()
            queue = manager.Queue()

            writer_proc = None
            try:
                import multiprocessing

                writer_proc = multiprocessing.Process(target=writer, args=(queue, output_path))
                writer_proc.start()

                queue.put(['{"test": 1}', '{"test": 2}'])
                queue.put(None)
            finally:
                if writer_proc and writer_proc.is_alive():
                    writer_proc.join(timeout=5)

            assert os.path.exists(output_path)
            with gzip.open(output_path, "rt", encoding="utf-8") as f:
                content = f.read()
            assert '"test"' in content
