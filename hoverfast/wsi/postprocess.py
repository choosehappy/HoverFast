#!/usr/bin/env python3

from __future__ import annotations

from multiprocessing import Queue
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import ujson
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.measure import regionprops
from skimage.segmentation import watershed

from ..common.spatialite import (
    bulk_insert_nuclei_wkb,
    configure_for_bulk_load,
    get_spatialite_connection,
    point_to_wkb,
    poly_to_wkb,
)
from .image_utils import save_poly


def pre_watershed(
    output_mask: np.ndarray, maps: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Prepare for watershed segmentation by processing model output maps."""
    if np.all(output_mask == 0):
        return None, None, None

    h_dir = cv2.normalize(maps[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # type: ignore[call-overload]
    v_dir = cv2.normalize(maps[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # type: ignore[call-overload]

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=5)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=5)
    del h_dir, v_dir

    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # type: ignore[call-overload]
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # type: ignore[call-overload]
    overall = np.hypot(sobelh, sobelv)
    del sobelh, sobelv

    opening = output_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=2)  # type: ignore[assignment]

    np.subtract(overall, 1 - opening, out=overall)
    np.maximum(overall, 0, out=overall)

    dist = (1.0 - overall) * opening
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall = (overall >= 0.4).astype(np.int32)

    marker = opening - overall
    np.maximum(marker, 0, out=marker)
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker, _ = ndi.label(marker)
    del overall

    return dist, marker, opening


def _simplify_contour(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Approximate a contour with fewer vertices when a tolerance is set."""
    if tolerance == 0:
        return contour
    return cv2.approxPolyDP(contour, tolerance * cv2.arcLength(contour, True) / 1000, True)


def _repair_polygon(contour: np.ndarray) -> np.ndarray | None:
    """Return a valid single-ring contour, repairing self-intersections.

    ``None`` is returned when the geometry cannot be reduced to a polygon with
    a single boundary ring (the caller should skip such objects).
    """
    poly = Polygon(contour.squeeze())
    if poly.is_valid:
        return contour

    poly = make_valid(poly)
    for _ in range(10):
        if poly.geom_type == "Polygon":
            break
        poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
    else:
        return None

    bound = poly.boundary
    if bound.geom_type != "LineString":
        bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
    return np.array(bound.coords[:], int)


def _contour_centroid(contour: np.ndarray) -> tuple[int, int] | None:
    """Return the integer centroid of a contour, or ``None`` for zero area."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def centroid_in_valid_region(cx: int, cy: int, region_size: int, stride: int) -> bool:
    """Return whether a tile-local centroid belongs to this tile's valid region.

    Tiles overlap by ``stride`` pixels and are stepped by ``region_size - stride``.
    The valid region must be *half-open* so that adjacent tiles partition the
    plane exactly: a cell whose centroid lands on the shared margin is claimed by
    exactly one tile, preventing the same nucleus being emitted twice in the
    overlap. The previous closed interval kept boundary centroids in both tiles.
    """
    half = region_size // 2 - stride // 2
    center = region_size // 2
    return -half <= (cx - center) < half and -half <= (cy - center) < half


def _format_feature(
    coords: np.ndarray,
    centroid: np.ndarray,
    object_class: dict[str, Any],
    db_output_fname: str | None,
) -> Any:
    """Serialize one nucleus for the JSON or SpatiaLite backend."""
    if db_output_fname:
        return (
            "cell",
            object_class["name"],
            object_class["colorRGB"],
            False,
            poly_to_wkb(coords),
            point_to_wkb((float(centroid[0]), float(centroid[1]))),
        )
    return ujson.dumps(save_poly(coords, centroid, object_class))


def watershed_object(
    rg: Any,
    dist: np.ndarray,
    submarker: np.ndarray,
    opening: np.ndarray,
    offset: tuple[int, int],
    region_coord: np.ndarray,
    slide_data: dict[str, Any],
    db_output_fname: str | None = None,
    object_class: dict[str, Any] | None = None,
) -> list[Any]:
    """Perform watershed segmentation on detected objects."""
    if object_class is None:
        object_class = {"name": "Nuclei", "colorRGB": -65536}
    output: list[Any] = []
    vals = np.unique(submarker)
    vals = vals[np.nonzero(vals)]

    if vals.size > 1:
        label = watershed(dist, markers=submarker, mask=opening)  # type: ignore[no-untyped-call]
    else:
        label = rg.image.astype(np.uint8)
        vals = np.array([1])

    min_area = slide_data["threshold"] / slide_data["downfactor"] ** 2
    for val in vals:
        cell = np.uint8(label == val)
        contour = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0][0]  # type: ignore[call-overload]
        contour = _simplify_contour(contour, slide_data["poly_simplification"])

        if cv2.contourArea(contour) <= min_area:
            continue

        contour = _repair_polygon(contour)
        if contour is None:
            continue

        centroid = _contour_centroid(contour)
        if centroid is None:
            continue
        cx, cy = centroid

        if not centroid_in_valid_region(cx, cy, slide_data["region_size"], slide_data["stride"]):
            continue

        coords = (contour * slide_data["downfactor"]) + region_coord
        centroid_abs = slide_data["downfactor"] * np.array([cx, cy]) + region_coord
        output.append(_format_feature(coords, centroid_abs, object_class, db_output_fname))

    return output


def region_feature(
    output_mask: np.ndarray,
    region_coord: np.ndarray,
    dist: np.ndarray,
    marker: np.ndarray,
    opening: np.ndarray,
    slide_data: dict[str, Any],
    db_output_fname: str | None,
) -> list[Any]:
    """Extract features from each detected region."""
    output: list[Any] = []
    rgs = regionprops(ndi.label(output_mask)[0])  # type: ignore[no-untyped-call]

    for rg in rgs:
        if rg.area < slide_data["threshold"] / slide_data["downfactor"] ** 2:
            continue
        ymin, xmin, ymax, xmax = rg.bbox
        output += watershed_object(
            rg,
            dist[ymin:ymax, xmin:xmax],
            marker[ymin:ymax, xmin:xmax] * rg.image,
            opening[ymin:ymax, xmin:xmax],
            (xmin, ymin),
            region_coord,
            slide_data,
            db_output_fname,
        )

    return output


def post_processing_batch_task(
    output_tensor: torch.Tensor,
    maps_tensor: torch.Tensor,
    coords_tensor: torch.Tensor,
    slide_data: dict[str, Any],
    features_queue: Queue,
    db_output_fname: str | None = None,
) -> int:
    """Worker function for the multiprocessing pool to handle post-processing."""
    output_batch = output_tensor.numpy()
    maps_batch = maps_tensor.float().numpy()
    coords_batch = coords_tensor.numpy()

    batch_features: list[Any] = []

    # for i in range(len(output_batch)):
    #     maps = maps_tensor[i]
    #     output_mask = output_batch[i]
    #     region_coord = coords_batch[i]
    for output_mask, maps, region_coord in zip(output_batch, maps_batch, coords_batch):
        dist, marker, opening = pre_watershed(output_mask, maps)
        if marker is None:
            continue

        assert dist is not None and opening is not None
        features = region_feature(output_mask, region_coord, dist, marker, opening, slide_data, db_output_fname)
        if features:
            batch_features.extend(features)

    if batch_features:
        if db_output_fname:
            conn = get_spatialite_connection(db_output_fname)
            configure_for_bulk_load(conn)
            try:
                bulk_insert_nuclei_wkb(conn, batch_features, srid=0) 
            finally:
                conn.close()
        else:
            features_queue.put(batch_features)

    return len(batch_features)
