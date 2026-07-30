"""
Quick sanity-check script for hoverfast_outputwsi.sqlite

Pulls a random sample of rows from the `nuclei` table, decodes the geometry
columns back to GeoJSON/coords, and prints them in a readable form so you
can eyeball that the WKB was written correctly.
"""

import sqlite3
import json

DB_PATH = "hoverfast_outputwsi.sqlite"
N_SAMPLES = 10


def get_spatialite_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.enable_load_extension(False)
    return conn


def main():
    conn = get_spatialite_connection(DB_PATH)
    cur = conn.cursor()

    # sanity: row count
    cur.execute("SELECT COUNT(*) FROM nuclei")
    total = cur.fetchone()[0]
    print(f"Total rows in nuclei table: {total:,}\n")

    if total == 0:
        print("Table is empty -- nothing to verify.")
        return

    # random sample -- ORDER BY RANDOM() is fine for a quick check on
    # sample sizes like 10; avoid it on huge tables for repeated/heavy use
    cur.execute(f"""
        SELECT
            id,
            object_type,
            classification_name,
            classification_color,
            is_locked,
            AsGeoJSON(geom)      AS geom_geojson,
            AsText(geom)         AS geom_wkt,
            AsGeoJSON(centroid)  AS centroid_geojson,
            IsValid(geom)        AS geom_is_valid,
            ST_NPoints(geom)     AS n_points
        FROM nuclei
        ORDER BY RANDOM()
        LIMIT ?
    """, (N_SAMPLES,))

    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    for row in rows:
        record = dict(zip(cols, row))

        print("=" * 70)
        print(f"id: {record['id']}")
        print(f"object_type: {record['object_type']}")
        print(f"classification: {record['classification_name']} "
              f"(colorRGB={record['classification_color']})")
        print(f"is_locked: {bool(record['is_locked'])}")
        print(f"geom valid: {bool(record['geom_is_valid'])}  "
              f"n_points: {record['n_points']}")

        centroid_geom = json.loads(record["centroid_geojson"])
        print(f"centroid (GeoJSON coords): {centroid_geom['coordinates']}")

        polygon_geom = json.loads(record["geom_geojson"])
        # print as a compact GeoJSON Feature, mirroring the original
        # save_poly() output structure for easy visual comparison
        feature = {
            "type": "Feature",
            "geometry": {
                "type": polygon_geom["type"],
                "coordinates": polygon_geom["coordinates"],
                "centroid": centroid_geom["coordinates"],
            },
            "properties": {
                "object_type": record["object_type"],
                "classification": {
                    "name": record["classification_name"],
                    "colorRGB": record["classification_color"],
                },
                "isLocked": bool(record["is_locked"]),
            },
        }
        # truncate long coordinate lists for readability in terminal
        coords = feature["geometry"]["coordinates"][0]
        preview_n = 4
        if len(coords) > preview_n * 2:
            coords_preview = (
                coords[:preview_n]
                + [f"... ({len(coords) - preview_n * 2} more points) ..."]
                + coords[-preview_n:]
            )
        else:
            coords_preview = coords
        feature_preview = json.loads(json.dumps(feature))
        feature_preview["geometry"]["coordinates"] = [coords_preview]

        print("GeoJSON feature (coords truncated for display):")
        print(json.dumps(feature_preview, indent=2))
        print()

    # quick aggregate checks, useful as an at-a-glance correctness signal
    cur.execute("SELECT COUNT(*) FROM nuclei WHERE IsValid(geom) = 0")
    n_invalid = cur.fetchone()[0]
    print("=" * 70)
    print(f"Invalid polygons in full table: {n_invalid:,} / {total:,}")

    cur.execute("""
        SELECT classification_name, COUNT(*)
        FROM nuclei
        GROUP BY classification_name
    """)
    print("\nClassification breakdown:")
    for name, count in cur.fetchall():
        print(f"  {name}: {count:,}")

    conn.close()


if __name__ == "__main__":
    main()