#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import glob
import logging
import math
import multiprocessing
import os
import time
from typing import Any

import numpy as np
import openslide
import torch
from PIL import Image
from tqdm import tqdm

from .spatialite_utils import get_spatialite_connection, init_spatialite_db_deferred_index
from .wsi_image_utils import preload_file_linux, writer
from .wsi_model import WSIPatchDataset, load_model, predict_batch, predict_ihc_batch
from .wsi_postprocess import post_processing_batch_task, pre_watershed

# Backward-compatible re-exports for utils_roi.py and other consumers
__all__: list[str] = [
    "main_wsi",
    "infer_wsi",
    "get_slide",
    "find_regions",
    "load_model",
    "pre_watershed",
    "WSIPatchDataset",
]


def find_regions(mask_dir: str | None, slide_data: dict[str, Any]) -> np.ndarray[Any, Any]:
    """Find regions in a whole slide image (WSI) for inference."""
    if mask_dir is None:
        osh = openslide.open_slide(os.path.join(slide_data["fpath"], slide_data["sname"] + f".{slide_data['format']}"))
        level: int = np.argwhere(np.array(osh.level_downsamples) - 32 <= 10**-2).reshape(-1)[-1]
        upscale_factor: float = osh.level_downsamples[level]
        from .wsi_image_utils import rgba2rgb

        mask = rgba2rgb(
            osh.read_region(
                (slide_data["xb"], slide_data["yb"]),
                level,
                (round(slide_data["width"] / upscale_factor), round(slide_data["height"] / upscale_factor)),
            )
        ).convert("L")
        import cv2

        mask = cv2.adaptiveThreshold(  # type: ignore[assignment]
            np.asarray(mask), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        mask = Image.open(os.path.join(mask_dir, slide_data["sname"] + ".png")).convert("L")
        upscale_factor = round(slide_data["width"] / mask.size[0])

    pts = np.argwhere(np.asarray(mask))
    pts = pts * upscale_factor  # type: ignore[assignment]
    xbins_count: int = math.ceil((slide_data["width"] - slide_data["stride_at_base"]) / slide_data["tile_at_base"])
    ybins_count: int = math.ceil((slide_data["height"] - slide_data["stride_at_base"]) / slide_data["tile_at_base"])
    density: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    density, xedges, yedges = np.histogram2d(
        pts[:, 1],
        pts[:, 0],
        bins=[xbins_count, ybins_count],
        range=[
            [
                int(slide_data["stride_at_base"] // 2),
                xbins_count * slide_data["tile_at_base"] + int(slide_data["stride_at_base"] // 2),
            ],
            [
                int(slide_data["stride_at_base"] // 2),
                ybins_count * slide_data["tile_at_base"] + int(slide_data["stride_at_base"] // 2),
            ],
        ],
    )
    return np.argwhere(density) * slide_data["tile_at_base"]  # type: ignore[no-any-return]


def get_slide(
    sname: str,
    sformat: str,
    fpath: str,
    mag: float,
    kernel_size: int,
    region_size: int,
    threshold: float,
    outdir: str,
    poly_simplify_tolerance: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Gather data for a specific slide and set up parameters for processing."""
    slide_data: dict[str, Any] = {}
    slide_data["format"] = sformat
    slide_data["sname"] = sname
    slide_data["fpath"] = fpath

    osh = openslide.open_slide(os.path.join(slide_data["fpath"], slide_data["sname"] + f".{slide_data['format']}"))
    slide_data["xb"] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
    slide_data["yb"] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
    slide_data["width"] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, osh.level_dimensions[0][0]))
    slide_data["height"] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, osh.level_dimensions[0][1]))
    mpp_value: str | None = osh.properties.get("openslide.mpp-x")

    if mpp_value is None:
        mpp: float = 0.245
        logger.warning("WARNING. The MPP value was not found; using default value of 0.245.")
    else:
        mpp = float(mpp_value)

    from .wsi_image_utils import magnification_from_mpp

    base_mag: float = magnification_from_mpp(mpp)
    slide_data["base_mag"] = base_mag

    if base_mag < mag:
        raise ValueError(f"ERROR: Base magnification level lower than {mag}X detected.")
    slide_data["mpp"] = mpp
    downfactor: float = base_mag / mag
    slide_data["downfactor"] = downfactor

    level_downsamples = np.array(osh.level_downsamples, int)
    level = np.argwhere(level_downsamples <= downfactor).reshape(-1)[-1]
    slide_data["level"] = level
    slide_data["working_d"] = level_downsamples[level]

    slide_data["kernel_size"] = kernel_size
    slide_data["stride"] = slide_data["kernel_size"] // 2
    slide_data["stride_at_base"] = int(slide_data["stride"] * downfactor)
    slide_data["region_size"] = region_size
    slide_data["region_at_base"] = int(slide_data["region_size"] * downfactor)
    slide_data["tile_size"] = region_size - slide_data["stride"]
    slide_data["tile_at_base"] = int(slide_data["tile_size"] * downfactor)
    slide_data["poly_simplification"] = poly_simplify_tolerance
    slide_data["threshold"] = threshold / (mpp**2)
    slide_data["outdir"] = outdir

    return slide_data


def infer_wsi(
    sname: str,
    sformat: str,
    fpath: str,
    mask_dir: str | None,
    outdir: str,
    mag: float,
    batch_to_gpu: int,
    region_size: int,
    model: torch.nn.Module,
    device: torch.device,
    n_process: int,
    poly_simplify_tolerance: float,
    threshold: float,
    stain: str,
    logger: logging.Logger,
    db_output_fname: str | None,
) -> tuple[int, int]:
    """Perform nuclei detection on a whole slide image (WSI) using streaming DataLoader."""

    n_post_proc: int = max(1, n_process // 2)
    n_loader: int = max(1, n_process - n_post_proc)

    kernel_size: int = 256
    slide_data: dict[str, Any] = get_slide(
        sname, sformat, fpath, mag, kernel_size, region_size, threshold, outdir, poly_simplify_tolerance, logger
    )

    coords: np.ndarray = find_regions(mask_dir, slide_data)
    print(f"|- Computation starting on {len(coords)} patches.")

    dataset = WSIPatchDataset(coords, slide_data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_to_gpu,
        shuffle=False,
        num_workers=n_loader,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    if db_output_fname:
        conn = get_spatialite_connection(db_output_fname)
        init_spatialite_db_deferred_index(conn, srid=0)
    else:
        features_queue = multiprocessing.Manager().Queue()
        writer_process: multiprocessing.Process = multiprocessing.Process(
            target=writer, args=(features_queue, os.path.join(outdir, sname + ".json.gz"))
        )
        writer_process.start()

    post_proc_pool = multiprocessing.Pool(processes=n_post_proc)

    total_objects: int = 0
    async_results: list[Any] = []

    with torch.inference_mode():
        for batch_imgs, batch_coords_tensor in tqdm(loader, desc="Streaming Inference", leave=False):
            batch_imgs = batch_imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
            batch_imgs = batch_imgs.half().div(255.0)

            if stain == "ihc_dab":
                output_mask, maps = predict_ihc_batch(batch_imgs, model, device)
            else:
                output_mask, maps = predict_batch(batch_imgs, model)

            copy_stream: torch.cuda.Stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                output_cpu: torch.Tensor = output_mask.to("cpu", non_blocking=True)
                maps_cpu: torch.Tensor = maps.to("cpu", non_blocking=True)
                output_mask.record_stream(copy_stream)
                maps.record_stream(copy_stream)
            copy_event: torch.cuda.Event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            copy_event.record(copy_stream)

            coords_cpu: torch.Tensor = batch_coords_tensor.clone()
            copy_event.synchronize()

            output_cpu.share_memory_()  # type: ignore[no-untyped-call]
            maps_cpu.share_memory_()  # type: ignore[no-untyped-call]

            if db_output_fname:
                res: Any = post_proc_pool.apply_async(
                    post_processing_batch_task,
                    args=(output_cpu, maps_cpu, coords_cpu, slide_data, None, db_output_fname),
                )
            else:
                res = post_proc_pool.apply_async(
                    post_processing_batch_task, args=(output_cpu, maps_cpu, coords_cpu, slide_data, features_queue)
                )

            async_results.append(res)

    post_proc_pool.close()
    post_proc_pool.join()

    for res in async_results:
        total_objects += res.get()

    if db_output_fname:
        pass
    else:
        features_queue.put(None)
        writer_process.join()

    return len(coords), total_objects


def main_wsi(args: argparse.Namespace) -> None:
    """Main entry point for nuclei detection on whole slide images (WSI)."""
    print(args)
    slide_dirs: list[str] = args.slide_folder
    outdir: str = args.outdir
    mask_dir: str | None = args.binmask_dir
    mag: float = args.magnification
    batch_to_gpu: int = args.batch_gpu
    region_size: int = args.tile_size
    n_process: int | None = args.n_process
    model_path: str = args.model_path
    poly_simplify_tolerance: float = args.poly_simplify
    threshold: float = args.size_threshold
    stain: str = args.stain

    db_output: bool = args.db_output

    multiprocessing.set_start_method("fork", force=True)

    if n_process is None:
        n_process = os.cpu_count() or 1

    os.makedirs(outdir, exist_ok=True)

    logger: logging.Logger = logging.getLogger(
        f"{outdir}/HoverFast_log_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%Hh%M")
    )
    f_handler: logging.FileHandler = logging.FileHandler(
        f"{outdir}/HoverFast_log_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%Hh%M") + ".log"
    )
    c_handler: logging.StreamHandler = logging.StreamHandler()
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)
    c_format: logging.Formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    if len(slide_dirs) == 1:
        pattern: str = slide_dirs[0]

        slide_dirs = glob.glob(pattern) if glob.has_magic(pattern) else [pattern]

    if not slide_dirs:
        logger.error("No slides detected.")
        raise ValueError("No slides detected.")

    stats: dict[str, list[Any]] = {}

    preload_file_linux(slide_dirs[0])

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = load_model(model_path, device)

    for si, slide_dir in enumerate(slide_dirs):
        if si + 1 < len(slide_dirs):
            preload_file_linux(slide_dirs[si + 1])

        temp: tuple[str, str, str] = os.path.basename(slide_dir).rpartition(".")
        sname: str = temp[0]
        sformat: str = temp[-1]
        fpath: str = os.path.dirname(slide_dir)
        print(f"- Working on {sname}")
        stats[sname] = []

        db_output_fname: str | None = outdir + f"/{sname}.sqlite" if db_output else None

        try:
            start: float = time.time()
            n_patches: int
            n_objects: int
            n_patches, n_objects = infer_wsi(
                sname,
                sformat,
                fpath,
                mask_dir,
                outdir,
                mag,
                batch_to_gpu,
                region_size,
                model,
                device,
                n_process,
                poly_simplify_tolerance,
                threshold,
                stain,
                logger,
                db_output_fname,
            )
            stats[sname].append(n_patches)
            stats[sname].append(n_objects)
            stats[sname].append(time.time() - start)
            print(f"running time: {stats[sname][-1]:.2f}s, #patches {stats[sname][0]} and #objects {stats[sname][1]}")
        except Exception:
            logger.exception("File %s failed", sname)
