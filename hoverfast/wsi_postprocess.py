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

from .spatialite_utils import (
    bulk_insert_nuclei_wkb,
    get_spatialite_connection,
    point_to_wkb,
    poly_to_wkb,
)


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
    overall = np.sqrt(sobelh**2 + sobelv**2)
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
        vals = [1]

    for val in vals:
        cell = np.uint8(label == val)
        c = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0][0]  # type: ignore[call-overload]

        if slide_data["poly_simplification"] != 0:
            c = cv2.approxPolyDP(c, slide_data["poly_simplification"] * cv2.arcLength(c, True) / 1000, True)

        if cv2.contourArea(c) <= slide_data["threshold"] / slide_data["downfactor"] ** 2:
            continue

        poly = Polygon(c.squeeze())
        if not poly.is_valid:
            poly = make_valid(poly)
            while poly.geom_type != "Polygon":
                poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
            bound = poly.boundary
            if bound.geom_type != "LineString":
                bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
            c = np.array(bound.coords[:], int)

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if np.any(
            np.abs(np.array([cx, cy]) - slide_data["region_size"] // 2)
            > slide_data["region_size"] // 2 - slide_data["stride"] // 2
        ):
            continue

        coords = (c * slide_data["downfactor"]) + region_coord
        centroid = slide_data["downfactor"] * np.array([cx, cy]) + region_coord

        if db_output_fname:
            output.append(
                (
                    "cell",
                    object_class["name"],
                    object_class["colorRGB"],
                    False,
                    poly_to_wkb(coords),
                    point_to_wkb(centroid),
                )
            )
        else:
            from .wsi_image_utils import save_poly

            poly_feat = save_poly(coords, centroid, object_class)
            output.append(ujson.dumps(poly_feat))

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
            bulk_insert_nuclei_wkb(conn, batch_features, srid=0, batch_size=50_000)
        else:
            features_queue.put(batch_features)

    return len(batch_features)
