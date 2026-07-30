"""
Diagnose where the bytes are going in hoverfast_outputwsi.sqlite.

Compares:
  - raw file size on disk
  - SQLite's own page accounting (page_count, freelist_count, page_size)
  - per-table/per-index space usage via dbstat (if compiled in)
  - average bytes per row for geometry columns
"""

import sqlite3
import os

DB_PATH = "hoverfast_outputwsi.sqlite"


def get_spatialite_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.enable_load_extension(False)
    return conn


def main():
    file_size = os.path.getsize(DB_PATH)
    conn = get_spatialite_connection(DB_PATH)
    cur = conn.cursor()

    page_size = cur.execute("PRAGMA page_size").fetchone()[0]
    page_count = cur.execute("PRAGMA page_count").fetchone()[0]
    freelist_count = cur.execute("PRAGMA freelist_count").fetchone()[0]

    used_bytes = (page_count - freelist_count) * page_size
    free_bytes = freelist_count * page_size

    print("=" * 60)
    print(f"File size on disk:        {file_size:,} bytes ({file_size / 1e6:.1f} MB)")
    print(f"page_size:                {page_size:,} bytes")
    print(f"page_count:               {page_count:,}")
    print(f"freelist_count:           {freelist_count:,}")
    print(f"Used (allocated) space:   {used_bytes:,} bytes ({used_bytes / 1e6:.1f} MB)")
    print(f"Free (reclaimable) space: {free_bytes:,} bytes ({free_bytes / 1e6:.1f} MB)")
    print(f"  -> {'Run VACUUM: significant reclaimable space' if free_bytes > 0.05 * file_size else 'Little free space; size is mostly real data'}")
    print()

    # row count + average bytes/row for geometry columns specifically
    cur.execute("SELECT COUNT(*) FROM nuclei")
    n_rows = cur.fetchone()[0]
    print(f"Row count: {n_rows:,}")

    if n_rows > 0:
        # LENGTH() on a BLOB column gives byte size of that column's stored value
        cur.execute("""
            SELECT
                AVG(LENGTH(geom))      AS avg_geom_bytes,
                AVG(LENGTH(centroid))  AS avg_centroid_bytes,
                SUM(LENGTH(geom))      AS total_geom_bytes,
                SUM(LENGTH(centroid))  AS total_centroid_bytes
            FROM nuclei
        """)
        avg_geom, avg_centroid, total_geom, total_centroid = cur.fetchone()
        print(f"Avg geom bytes/row:      {avg_geom:.1f}")
        print(f"Avg centroid bytes/row:  {avg_centroid:.1f}")
        print(f"Total geom column:       {total_geom:,} bytes ({total_geom / 1e6:.1f} MB)")
        print(f"Total centroid column:   {total_centroid:,} bytes ({total_centroid / 1e6:.1f} MB)")
        print(f"Sum of geometry columns: {(total_geom + total_centroid) / 1e6:.1f} MB "
              f"({(total_geom + total_centroid) / used_bytes * 100:.1f}% of used space)")
    print()

    # per-table breakdown via dbstat, if available (most builds have this)
    try:
        cur.execute("""
            SELECT name, SUM(pgsize) AS bytes
            FROM dbstat
            GROUP BY name
            ORDER BY bytes DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        print("=" * 60)
        print("Per-table/per-index space usage (dbstat):")
        for name, nbytes in rows:
            print(f"  {name:35s} {nbytes:>12,} bytes  ({nbytes / 1e6:6.1f} MB)")
    except sqlite3.OperationalError as e:
        print(f"(dbstat virtual table not available: {e})")
        print("Can't get per-table breakdown -- your sqlite3 build may lack DBSTAT support.")

    conn.close()


if __name__ == "__main__":
    main()