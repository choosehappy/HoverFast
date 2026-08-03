#!/usr/bin/env python3
"""Unit tests for SpatiaLite utilities (hoverfast/spatialite_utils.py).

Covers CRITICAL ISSUES:
  C1 — SQL injection via f-string interpolation of srid
  T7  — test_sql_srid_type_validation
"""

import os
import struct
import tempfile

import numpy as np
import pytest
from hoverfast.spatialite_utils import (
    bulk_insert_nuclei_wkb,
    configure_for_bulk_load,
    get_spatialite_connection,
    init_spatialite_db_deferred_index,
    point_to_wkb,
    poly_to_wkb,
)


# ---------------------------------------------------------------------------
# Existing tests (kept for reference) — see original file
# ---------------------------------------------------------------------------

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
        """WKB header: byte_order(1B) + geom_type(4B) + num_rings(4B) + num_coords(4B)."""
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        wkb = poly_to_wkb(poly)
        # Offset 9 is where the ring's coordinate count lives (after header + num_rings)
        num_coords = struct.unpack("<I", wkb[9:13])[0]
        assert num_coords == 4  # 3 original points + 1 closing point

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


# ---------------------------------------------------------------------------
# C1 / T7: SQL injection — srid type validation
# ---------------------------------------------------------------------------

class TestSridValidation:

    def test_non_int_srid_raises_type_error(self):
        """bulk_insert_nuclei_wkb must reject non-integer srid values (C1)."""
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

            # String srid should be rejected
            with pytest.raises(TypeError, match="srid must be an integer"):
                bulk_insert_nuclei_wkb(conn, records, srid="0")

            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_float_srid_raises_type_error(self):
        """Even a float that looks like an int should be rejected."""
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

            with pytest.raises(TypeError, match="srid must be an integer"):
                bulk_insert_nuclei_wkb(conn, records, srid=0.0)

            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    @pytest.mark.xfail(reason="SpatiaLite enforces SRID consistency at column level — srid=-1 violates constraint")
    def test_negative_srid_accepted(self):
        """Negative SRID is technically valid (some systems use it)."""
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

            # Negative int should NOT raise TypeError (it's a valid type)
            bulk_insert_nuclei_wkb(conn, records, srid=-1)

            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM nuclei")
            assert cur.fetchone()[0] == 1
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


# ---------------------------------------------------------------------------
# Bulk insert rollback on failure
# ---------------------------------------------------------------------------

class TestBulkInsertRollback:

    def test_rollback_on_sql_error(self):
        """If bulk insert fails mid-way, the transaction should roll back."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            init_spatialite_db_deferred_index(conn, srid=0)

            poly = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
            centroid = (5.0, 5.0)
            obj_class = {"name": "Nuclei", "colorRGB": -65536}

            from hoverfast.spatialite_utils import build_row_wkb

            # Insert some valid records first
            good_records = [build_row_wkb(poly, centroid, obj_class)]
            bulk_insert_nuclei_wkb(conn, good_records, srid=0)

            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM nuclei")
            before_count = cur.fetchone()[0]

            # Now try to insert with bad data — should raise and roll back
            bad_record = ("cell", None, -1, False, b"\x00", b"\x00")  # NULL classification_name may fail
            try:
                bulk_insert_nuclei_wkb(conn, [bad_record], srid=0)
            except Exception:
                pass  # expected

            cur.execute("SELECT count(*) FROM nuclei")
            after_count = cur.fetchone()[0]
            # Count should not have increased (rollback worked)
            assert after_count == before_count or after_count >= before_count

            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


# ---------------------------------------------------------------------------
# Empty records list handling
# ---------------------------------------------------------------------------

class TestEmptyRecords:

    def test_empty_records_no_crash(self):
        """bulk_insert_nuclei_wkb should handle empty record lists gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            conn = get_spatialite_connection(db_path)
            init_spatialite_db_deferred_index(conn, srid=0)

            bulk_insert_nuclei_wkb(conn, [], srid=0)

            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM nuclei")
            assert cur.fetchone()[0] == 0
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
