#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import struct
from typing import Any

import numpy as np


def get_spatialite_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a connection to a (Spatia)Lite DB with mod_spatialite loaded.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    # name varies by platform: 'mod_spatialite', 'mod_spatialite.so', 'mod_spatialite.dylib'
    conn.load_extension("mod_spatialite")
    conn.enable_load_extension(False)
    return conn


def poly_to_wkb(poly: np.ndarray) -> bytes:
    """
    Build a WKB POLYGON blob directly (little-endian, no SRID prefix --
    we pass SRID separately to GeomFromWKB).
    """
    pts = poly.squeeze()
    coords = [(float(x), float(y)) for x, y in pts]
    coords.append(coords[0])  # close ring
    n = len(coords)

    # byte order (1 = little endian), geom type (3 = polygon), num rings (1)
    header = struct.pack("<BII", 1, 3, 1)
    ring_header = struct.pack("<I", n)
    ring_body = b"".join(struct.pack("<dd", x, y) for x, y in coords)

    return header + ring_header + ring_body


def point_to_wkb(centroid: tuple[float, float]) -> bytes:
    x, y = float(centroid[0]), float(centroid[1])
    # byte order, geom type (1 = point)
    return struct.pack("<BI", 1, 1) + struct.pack("<dd", x, y)


def init_spatialite_db_deferred_index(conn: sqlite3.Connection, srid: int = 0) -> None:
    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM sqlite_master WHERE name='spatial_ref_sys'")
    if cur.fetchone()[0] == 0:
        cur.execute("SELECT InitSpatialMetaData(1)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS nuclei (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type           TEXT,
            classification_name   TEXT,
            classification_color  INTEGER,
            is_locked             BOOLEAN
        )
    """)
    cur.execute("SELECT AddGeometryColumn('nuclei', 'geom', ?, 'POLYGON', 'XY')", (srid,))
    cur.execute("SELECT AddGeometryColumn('nuclei', 'centroid', ?, 'POINT', 'XY')", (srid,))
    conn.commit()
    # NOTE: no CreateSpatialIndex() call here -- do that after loading


def build_spatial_indexes(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT CreateSpatialIndex('nuclei', 'geom')")
    cur.execute("SELECT CreateSpatialIndex('nuclei', 'centroid')")
    conn.commit()


def configure_for_bulk_load(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")  # or OFF for max speed, less safe
    cur.execute("PRAGMA synchronous = OFF")  # skip fsync on every commit
    cur.execute("PRAGMA cache_size = -200000")  # ~200MB page cache (negative = KB)
    cur.execute("PRAGMA temp_store = MEMORY")
    cur.execute("PRAGMA mmap_size = 30000000000")  # optional, if you have RAM/64-bit
    conn.commit()


def build_row_wkb(
    poly: np.ndarray,
    centroid: tuple[float, float],
    object_class: dict[str, Any],
) -> tuple[str, str, int, bool, bytes, bytes]:
    return (
        "cell",
        object_class["name"],
        object_class["colorRGB"],
        False,
        poly_to_wkb(poly),  # bytes -> bound as BLOB
        point_to_wkb(centroid),  # bytes -> bound as BLOB
    )


def bulk_insert_nuclei_wkb(
    conn: sqlite3.Connection,
    records: list[tuple[str, str, int, bool, bytes, bytes]],
    srid: int = 0,
    batch_size: int = 50_000,
) -> None:
    insert_sql = f"""
        INSERT INTO nuclei
            (object_type, classification_name, classification_color,
             is_locked, geom, centroid)
        VALUES
            (?, ?, ?, ?, GeomFromWKB(?, {srid}), GeomFromWKB(?, {srid}))
    """

    cur = conn.cursor()
    cur.execute("BEGIN")

    for i in range(0, len(records), batch_size):
        cur.executemany(insert_sql, records[i : i + batch_size])

    conn.commit()


def load_millions(
    db_path: str,
    records: list[tuple[str, str, int, bool, bytes, bytes]],
    srid: int = 0,
) -> None:
    conn = get_spatialite_connection(db_path)
    init_spatialite_db_deferred_index(conn, srid=srid)
    configure_for_bulk_load(conn)

    bulk_insert_nuclei_wkb(conn, records, srid=srid, batch_size=50_000)

    # build R-Tree indexes once, after all data is in
    build_spatial_indexes(conn)

    # optional: reclaim/optimize
    conn.execute("PRAGMA optimize")
    conn.execute("VACUUM")  # only if you can afford the I/O/time; not strictly required

    conn.commit()
    conn.close()
