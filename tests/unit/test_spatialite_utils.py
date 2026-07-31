#!/usr/bin/env python3
"""Unit tests for SpatiaLite utilities (hoverfast/spatialite_utils.py)."""

import os
import struct
import tempfile

import numpy as np
from hoverfast.spatialite_utils import (
    bulk_insert_nuclei_wkb,
    configure_for_bulk_load,
    get_spatialite_connection,
    init_spatialite_db_deferred_index,
    point_to_wkb,
    poly_to_wkb,
)


class TestPolyToWkb:
    def test_triangle(self):
        poly = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        wkb = poly_to_wkb(poly)
        assert isinstance(wkb, bytes)
        byte_order = struct.unpack("<B", wkb[0:1])[0]
        assert byte_order == 1

    def test_quad(self):
        poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        wkb = poly_to_wkb(poly)
        assert len(wkb) > 0

    def test_ring_closed(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        wkb = poly_to_wkb(poly)
        num_coords = struct.unpack("<I", wkb[5:9])[0]
        assert num_coords == 4

    def test_squeeze_works(self):
        poly = np.array([[[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]]])
        wkb = poly_to_wkb(poly)
        assert isinstance(wkb, bytes)


class TestPointToWkb:
    def test_basic_point(self):
        wkb = point_to_wkb((5.0, 10.0))
        assert isinstance(wkb, bytes)
        x, y = struct.unpack("<dd", wkb[5:])
        assert abs(x - 5.0) < 1e-10
        assert abs(y - 10.0) < 1e-10

    def test_integer_coords(self):
        wkb = point_to_wkb((0, 0))
        x, y = struct.unpack("<dd", wkb[5:])
        assert x == 0.0
        assert y == 0.0


class TestSpatiaLiteConnection:
    def test_get_connection(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            assert conn is not None
            cur = conn.cursor()
            cur.execute("SELECT spatialite_version()")
            row = cur.fetchone()
            assert row is not None
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_init_db_creates_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            init_spatialite_db_deferred_index(conn, srid=0)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM sqlite_master WHERE name='nuclei'")
            assert cur.fetchone()[0] == 1
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_bulk_insert(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            init_spatialite_db_deferred_index(conn, srid=0)

            poly = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
            centroid = (5.0, 5.0)
            obj_class = {"name": "Nuclei", "colorRGB": -65536}

            from hoverfast.spatialite_utils import build_row_wkb

            records = [build_row_wkb(poly, centroid, obj_class)]
            bulk_insert_nuclei_wkb(conn, records, srid=0)

            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM nuclei")
            assert cur.fetchone()[0] == 1
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_configure_bulk_load(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            configure_for_bulk_load(conn)
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode")
            assert cur.fetchone()[0] == "wal"
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
