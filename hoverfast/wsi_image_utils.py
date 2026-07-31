#!/usr/bin/env python3

from __future__ import annotations

import gzip
import os
from multiprocessing import Queue
from typing import Any

import numpy as np
from PIL import Image


def magnification_from_mpp(mpp: float) -> float:
    """
    Find the magnification from the micron per pixel value.

    Parameters:
    mpp (float): Micron per pixel value.

    Returns:
    float: Calculated magnification.
    """
    return float(40 * 2 ** (np.round(np.log2(0.2425 / mpp))))


def rgba2rgb(img: Image.Image) -> Image.Image:
    """
    Convert an RGBA image to an RGB image by merging the alpha channel with a white background.

    Parameters:
    img (PIL.Image.Image): An RGBA image.

    Returns:
    PIL.Image.Image: An RGB image with the alpha channel merged with a white background.
    """
    bg_color = "#" + "ffffff"
    thumb = Image.new("RGB", img.size, bg_color)
    thumb.paste(img, None, img)
    return thumb


def save_poly(poly: np.ndarray, centroid: np.ndarray, object_class: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Serialize a polygon representing a detected object.

    This function converts a polygon and its associated features into a serialized format
    compatible with GeoJSON.

    Parameters:
    poly (numpy.ndarray): Coordinates of the polygon.
    centroid (numpy.ndarray): Centroid of the polygon.
    object_class (dict, optional): Classification information for the object.

    Returns:
    str: Serialized polygon in GeoJSON format.
    """
    if object_class is None:
        object_class = {"name": "Nuclei", "colorRGB": -65536}
    feature: dict[str, Any] = {}
    feature["geometry"] = {
        "type": "Polygon",
        "coordinates": (tuple(map(tuple, poly.squeeze())) + (tuple(poly[0].squeeze()),),),
    }
    feature["geometry"]["centroid"] = [int(coord) for coord in centroid]
    feature["properties"] = {"object_type": "cell", "classification": object_class, "isLocked": False}
    feature["type"] = "Feature"
    return feature


def writer(features_queue: Queue, output_path: str) -> None:
    """Write serialized features to a gzipped JSON file."""
    first = True

    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=5) as file:
        file.write("[\n")

        while True:
            feature_batch = features_queue.get()

            if feature_batch is None:  # Sentinel value
                break

            if feature_batch:
                formatted_chunk = (",\n" if not first else "") + ",\n".join(feature_batch)
                file.write(formatted_chunk)
                first = False

        file.write("\n]")


def preload_file_linux(file_path: str) -> int:
    """Signals the Linux page cache to asynchronously preload the file into RAM."""
    fd = os.open(file_path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)
    return 0
