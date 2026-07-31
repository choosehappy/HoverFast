#!/usr/bin/env python3
"""Unit tests for ROI utilities (hoverfast/utils_roi.py)."""

import numpy as np
from hoverfast.utils_roi import divide_batch, int_coords, save_poly_dict


class TestIntCoords:
    def test_basic(self):
        result = int_coords([[1.4, 2.6], [3.9, 0.1]])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_already_integers(self):
        coords = [[1, 2], [3, 4]]
        result = int_coords(coords)
        assert result[0, 0] == 1
        assert result[0, 1] == 2


class TestDivideBatch:
    def test_even_split(self):
        items = list(range(10))
        batches = list(divide_batch(items, 3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]

    def test_exact_division(self):
        items = list(range(9))
        batches = list(divide_batch(items, 3))
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 3

    def test_single_element_batches(self):
        items = [1, 2, 3]
        batches = list(divide_batch(items, 1))
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_larger_than_list(self):
        items = [1, 2]
        batches = list(divide_batch(items, 10))
        assert len(batches) == 1
        assert batches[0] == [1, 2]

    def test_empty_list(self):
        batches = list(divide_batch([], 5))
        assert len(batches) == 0


class TestSavePolyDict:
    def test_basic_feature(self):
        poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        feature = save_poly_dict(poly)
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"

    def test_default_object_class(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        feature = save_poly_dict(poly)
        assert feature["properties"]["classification"]["name"] == "Nuclei"

    def test_custom_object_class(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        obj_class = {"name": "Tumor", "colorRGB": -1}
        feature = save_poly_dict(poly, obj_class)
        assert feature["properties"]["classification"]["name"] == "Tumor"

    def test_coordinates_closed(self):
        poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        feature = save_poly_dict(poly)
        coords = feature["geometry"]["coordinates"]
        assert coords[0] == coords[-1]

    def test_properties_structure(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        feature = save_poly_dict(poly)
        assert feature["properties"]["object_type"] == "cell"
        assert feature["properties"]["isLocked"] is False
