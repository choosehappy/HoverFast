#!/usr/bin/env python3

import datetime
import glob
import gzip
import json
import logging
import math
import multiprocessing
import os
import time

import cv2
import numpy as np
import openslide
import safetensors.torch
import scipy.ndimage as ndi
import torch
import ujson
from PIL import Image
from safetensors import safe_open
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.measure import regionprops
from skimage.segmentation import watershed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .hoverfast import HoverFast
from .spatialite_utils import (
    bulk_insert_nuclei_wkb,
    get_spatialite_connection,
    init_spatialite_db_deferred_index,
    point_to_wkb,
    poly_to_wkb,
)
from .utils_stain_deconv import *

# --- Helper Functions ---


def magnification_from_mpp(mpp): 
    """
    Find the magnification from the micron per pixel value.

    Parameters:
    mpp (float): Micron per pixel value.

    Returns:
    float: Calculated magnification.
    """
    return 40*2**(np.round(np.log2(0.2425/mpp)))

def load_model(model_path, device):
    """
    Load the pre-trained model from the given path.

    Parameters:
    model_path (str): Path to the pre-trained model.
    device (torch.device): Device to load the model on.

    Returns:
    torch.nn.Module: Loaded model ready for inference.
    """


    if not os.path.exists("unet_trt.ts"):
        print("not compiled - building")
        # 1. Inspect metadata without loading tensors into RAM
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
            config = json.loads(metadata["config"])

        # 2. Instantiate HoverFast using the extracted parameters
        model = HoverFast(**config).to(device, memory_format=torch.channels_last)

        # 3. Load weights directly into the instantiated model instance
        safetensors.torch.load_model(model, model_path)

        model = model.half()  # Convert the model to float16 (half precision) for faster inference
        model.eval()
    #---------
        import torch_tensorrt
        batch = torch.export.Dim("batch", min=1, max=7) #--- TODO: set to batch size

        example_input = torch.randn(7, 3, 1024, 1024, device="cuda", dtype=torch.float16)


        dynamic_shapes = {"x": {0: batch}}

        exp_program = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes=dynamic_shapes,  # match your forward()'s arg name
        )

        trt_model = torch_tensorrt.dynamo.compile(
            exp_program,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 1024, 1024),
                    opt_shape=(7, 3, 1024, 1024),  # pick whatever's most common
                    max_shape=(7, 3, 1024, 1024),
                    dtype=torch.half,
                )
            ],
            enabled_precisions={torch.half},
            optimization_level=5,
            workspace_size=8 << 30,   # 8 GB
            use_python_runtime=False,
        )
    #----------
        torch_tensorrt.save(trt_model, "unet_trt.ts", inputs=[example_input], output_format="torchscript")#, dynamic_shapes=dynamic_shapes)


    else:
        print("loading compiled..")
        
        import ctypes
        # 1. Manually open the missing TensorRT dependency into the global symbol table
        ctypes.CDLL(
            "/opt/conda/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.11",
            mode=ctypes.RTLD_GLOBAL,
        )

        # 2. Now load the torch_tensorrt C++ runtime
        torch.ops.load_library(
            "/opt/conda/lib/python3.11/site-packages/torch_tensorrt/lib/libtorchtrt_runtime.so"
        )
        
        trt_model = torch.jit.load("unet_trt.ts")
        

    print("returning model")
    return trt_model

def rgba2rgb(img):
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

def save_poly(poly, centroid, object_class=None):
    """
    Serialize a polygon representing a detected object.

    This function converts a polygon and its associated features into a serialized format
    compatible with GeoJSON.

    Parameters:
    poly (numpy.ndarray): Coordinates of the polygon.
    centroid (numpy.ndarray): Centroid of the polygon.
    object_class (dict, optional): Classification information for the object. Default is {'name': 'Nuclei', 'colorRGB': -65536}.

    Returns:
    str: Serialized polygon in GeoJSON format.
    """
    if object_class is None:
        object_class = {'name': 'Nuclei', 'colorRGB': -65536}
    feature = {}
    feature["geometry"] = {'type':'Polygon','coordinates':(tuple(map(tuple,poly.squeeze()))+(tuple(poly[0].squeeze()),),)}
    feature["geometry"]["centroid"] = [ int(coord) for coord in centroid]
    feature["properties"] = {'object_type': 'cell',
                                'classification': object_class,
                                'isLocked': False}
    feature["type"] = "Feature"
    return feature


def writer(features_queue, output_path):
    first = True
    
    with gzip.open(output_path, 'wt', encoding="utf-8", compresslevel=5) as file:
        file.write("[\n")
        
        while True:
            # feature_batch is now a list of feature strings from one worker batch
            feature_batch = features_queue.get()
            
            if feature_batch is None:  # Sentinel value
                break
                
            if feature_batch:
                formatted_chunk = (",\n" if not first else "") + ",\n".join(feature_batch)
                file.write(formatted_chunk)
                first = False
                
        file.write("\n]")

# --- Dataset Class ---

class WSIPatchDataset(Dataset):
    """
    PyTorch Dataset for lazy loading of WSI patches.
    
    This class handles the extraction of regions from a Whole Slide Image (WSI)
    on-the-fly during the DataLoader iteration. It ensures that OpenSlide objects
    are initialized within the worker processes to avoid pickling issues.
    """
    def __init__(self, coords, slide_data):
        """
        Initialize the dataset.

        Parameters:
        coords (numpy.ndarray): Array of coordinates (x, y) for the regions to extract.
        slide_data (dict): Dictionary containing slide metadata and parameters.
        """
        self.coords = coords
        self.slide_data = slide_data
        # We do NOT open the slide in __init__ because OpenSlide objects cannot be pickled
        # and passed to worker processes.
        self.slide = None 

    def __len__(self):
        """
        Return the total number of patches.
        """
        return len(self.coords)

    def __getitem__(self, idx):
        """
        Fetch a single patch.

        Parameters:
        idx (int): Index of the patch to retrieve.

        Returns:
        tuple: (tensor, coord)
            tensor (torch.Tensor): The image patch as a normalized float tensor (C, H, W).
            coord (numpy.ndarray): The (x, y) coordinate of the patch.
        """
        # Open the slide once per worker process
        if self.slide is None:
            self.slide = openslide.OpenSlide(
                os.path.join(self.slide_data['fpath'], 
                             self.slide_data['sname'] + f".{self.slide_data['format']}")
            )

        coord = self.coords[idx]
        
        # Read region
        region = self.slide.read_region(
            (self.slide_data['xb'] + coord[0], self.slide_data['yb'] + coord[1]), 
            self.slide_data['level'], 
            (int(self.slide_data['region_size']*(self.slide_data['downfactor']/self.slide_data['working_d'])),)*2
        )
        
        # Resize if necessary
        if self.slide_data['working_d'] != self.slide_data['downfactor']:
            region = region.resize((self.slide_data['region_size'],)*2)
        
        # Convert RGBA to RGB
        img = rgba2rgb(region)
        
        # Convert to Tensor and Normalize (0-1)
        # Permute to (C, H, W)
        img_np = np.array(img)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)# .float() / 255.0
        
        return tensor, coord

# --- Prediction Logic ---

def predict_ihc_batch(regions_gpu, model, device):
    """
    Perform nuclei detection with stain deconvolution on a batch of regions.

    Parameters:
    regions_gpu (torch.Tensor): Batch of images (B, C, H, W) normalized 0-1 on GPU.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on.

    Returns:
    tuple:
        output_processed (torch.Tensor): Boolean tensor of predicted masks.
        maps (torch.Tensor): Feature maps from the model.
    """
    # Convert to hed (expects tensor)
    hed_batch = rgb_to_hed_torch(regions_gpu, device)

    # Extract Hematoxylin channel
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)

    # Convert HED back to RGB
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)

    # Scale and transpose (reconstructed is usually BHWC, model needs BCHW)
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    # Model prediction
    output, maps = model(regions_gpu)
    
    # Post-processing prep
    output_processed = output.argmax(axis=1).type(torch.bool)
    
    return output_processed, maps

def predict_batch(regions_gpu, model):
    """
    Perform nuclei detection on a batch of regions.

    Parameters:
    regions_gpu (torch.Tensor): Batch of images (B, C, H, W) normalized 0-1 on GPU.
    model (torch.nn.Module): Pre-trained model for nuclei detection.

    Returns:
    tuple:
        output_processed (torch.Tensor): Boolean tensor of predicted masks.
        maps (torch.Tensor): Feature maps from the model.
    """
    # Model prediction
    output, maps = model(regions_gpu)
    
    # Post-processing prep
    output_processed = output.argmax(axis=1).type(torch.bool)

    maps_fp8 = maps.to(torch.float8_e4m3fn)

    return output_processed, maps_fp8

# --- Post Processing Logic ---

def pre_watershed(output_mask, maps):
    """
    Prepare for watershed segmentation by processing model output maps.

    This function normalizes the horizontal and vertical gradient maps, applies Sobel operations,
    and prepares the distance transform and marker image for watershed segmentation.

    Parameters:
    output_mask (numpy.ndarray): Binary mask indicating detected nuclei.
    maps (numpy.ndarray): Feature maps from the model output.

    Returns:
    tuple:
        dist (numpy.ndarray): Distance transform for watershed segmentation.
        marker (numpy.ndarray): Marker image for watershed segmentation.
        opening (numpy.ndarray): Processed mask for watershed segmentation.
    """
    if np.all(output_mask == 0):
        return None, None, None

    # Normalizing maps
    h_dir = cv2.normalize(maps[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(maps[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Sobel operations
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=5)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=5)
    del h_dir, v_dir

    # Normalizing sobel results
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    overall = np.sqrt(sobelh ** 2 + sobelv ** 2)
    del sobelh, sobelv

    # Morphology
    opening = output_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=2)

    np.subtract(overall, 1 - opening, out=overall)
    np.maximum(overall, 0, out=overall)

    # Preparing for watershed
    dist = (1.0 - overall) * opening
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall = (overall >= 0.4).astype(np.int32)

    marker = opening - overall
    np.maximum(marker, 0, out=marker)
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker, _ = ndi.label(marker)
    del overall

    return dist, marker, opening

def watershed_object(rg, dist, submarker, opening, offset, region_coord, slide_data, db_output_fname=None, object_class=None):
    """
    Perform watershed segmentation on detected objects.

    This function applies watershed segmentation to divide detected objects and extract their features.
    The features are then processed and saved.

    Parameters:
    rg (skimage.measure._regionprops.RegionProperties): Region properties of the detected object.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    submarker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    offset (tuple): Offset coordinates for the region.
    region_coord (numpy.ndarray): Coordinates of the region.
    slide_data (dict): Dictionary containing slide metadata and parameters.

    Returns:
    list: A list of serialized polygons representing detected nuclei.
    """
    if object_class is None:
        object_class = {'name': 'Nuclei', 'colorRGB': -65536}
    output = []
    vals = np.unique(submarker)
    vals = vals[np.nonzero(vals)]
    if vals.size > 1: 
        label = watershed(dist, markers=submarker, mask=opening)
    else:
        label = rg.image.astype(np.uint8)
        vals = [1]
    
    for val in vals:
        cell = np.uint8(label==val)
        c = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0][0]
        if slide_data['poly_simplification'] != 0:
            c = cv2.approxPolyDP(c,slide_data['poly_simplification']*cv2.arcLength(c,True)/1000,True)
        
        if cv2.contourArea(c) <= slide_data['threshold']/slide_data['downfactor']**2:
            continue

        poly = Polygon(c.squeeze())
        if not poly.is_valid:
            poly = make_valid(poly)
            while poly.geom_type != 'Polygon':
                poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
            bound = poly.boundary
            if bound.geom_type != 'LineString':
                bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
            c = np.array(bound.coords[:],int)

        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if np.any(np.abs(np.array([cx,cy])-slide_data['region_size']//2)>slide_data['region_size']//2-slide_data['stride']//2):
            continue


        coords = (c * slide_data['downfactor']) + region_coord
        centroid = slide_data['downfactor'] * np.array([cx, cy]) + region_coord

        if  db_output_fname:
            # Build directly as a DB-ready row: (geom_wkb, centroid_wkb, object_type,
            # classification_name, classification_color, is_locked)
            output.append((
                'cell',
                object_class['name'],
                object_class['colorRGB'],
                False,
                poly_to_wkb(coords),
                point_to_wkb(centroid),
            ))

        else:
            poly_feat = save_poly(coords, centroid, object_class)
            output.append(ujson.dumps(poly_feat))

            
    return output

def region_feature(output_mask, region_coord, dist, marker, opening, slide_data,db_output_fname):
    """
    Extract features from each detected region.

    This function isolates detected objects, performs watershed segmentation,
    and extracts features for each object.

    Parameters:
    output_mask (numpy.ndarray): Binary mask indicating detected nuclei.
    region_coord (numpy.ndarray): Coordinates of the region.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    marker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    slide_data (dict): Dictionary containing slide metadata and parameters.

    Returns:
    list: List of serialized feature strings.
    """
    output = []
    rgs = regionprops(ndi.label(output_mask)[0])
    
    for rg in rgs:
        if rg.area < slide_data['threshold']/slide_data['downfactor']**2:
            continue
        ymin,xmin,ymax,xmax = rg.bbox
        output += watershed_object(rg,dist[ymin:ymax,xmin:xmax],marker[ymin:ymax,xmin:xmax]*rg.image,opening[ymin:ymax,xmin:xmax],(xmin,ymin),region_coord,slide_data,db_output_fname)
    
    return output

def post_processing_batch_task(output_tensor, maps_tensor, coords_tensor, slide_data, features_queue , db_output_fname=None):
    """
    Worker function for the multiprocessing pool to handle post-processing.

    This function takes a batch of model predictions, performs watershed segmentation
    and feature extraction, and pushes the results to the writer queue.

    Parameters:
    batch_data (list): List of tuples (mask, maps, coord) from the inference loop.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    features_queue (multiprocessing.Queue): Queue to store the extracted features.

    Returns:
    int: Total number of features extracted in this batch.
    """

    # mask_shm = shared_memory.SharedMemory(name=mask_shm_name)
    # output_batch = np.ndarray(mask_shape, dtype=mask_dtype, buffer=mask_shm.buf)
    # maps_shm = shared_memory.SharedMemory(name=maps_shm_name)
    # maps_batch = np.ndarray(maps_shape, dtype=maps_dtype, buffer=maps_shm.buf)

#    try:

    output_batch = output_tensor.numpy()
    maps_batch = maps_tensor.float().numpy()
    coords_batch = coords_tensor.numpy()

    batch_features = []
    
    for output_mask, maps, region_coord in zip(output_batch, maps_batch, coords_batch):
        dist, marker, opening = pre_watershed(output_mask, maps)
        if marker is None:
            continue
            
        features = region_feature(output_mask, region_coord, dist, marker, opening, slide_data, db_output_fname)
        if features:
            batch_features.extend(features)  # Collect all features for the whole batch

    

    if batch_features:
        if db_output_fname:
            conn = get_spatialite_connection(db_output_fname)
            bulk_insert_nuclei_wkb(conn, batch_features, srid=0, batch_size=50_000)
        else:
            features_queue.put(batch_features)  # 1 Queue call per batch!
        
    return len(batch_features)


# --- Main Pipeline Logic ---

def find_regions(mask_dir, slide_data):
    """
    Find regions in a whole slide image (WSI) for inference.

    This function either creates a tissue mask from the WSI or loads a binary mask from the specified directory.
    It then identifies regions in the slide to be processed.

    Parameters:
    mask_dir (str or None): Directory containing quality control masks. If None, a tissue mask is created from the WSI.
    slide_data (dict): Dictionary containing slide metadata and parameters.

    Returns:
    numpy.ndarray: Array of coordinates for regions to be inferred on.
    """
    if mask_dir is None:
        osh = openslide.open_slide(os.path.join(slide_data['fpath'],slide_data['sname']+f".{slide_data['format']}"))
        level = np.argwhere(np.array(osh.level_downsamples)-32 <= 10**-2).reshape(-1)[-1]
        upscale_factor = osh.level_downsamples[level]
        mask = rgba2rgb(osh.read_region((slide_data['xb'], slide_data['yb']), level,(round(slide_data['width']/upscale_factor),round(slide_data['height']/upscale_factor)))).convert('L')
        mask = cv2.adaptiveThreshold(np.asarray(mask),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    else:
        mask = Image.open(os.path.join(mask_dir,slide_data['sname']+'.png')).convert('L')
        upscale_factor = round(slide_data['width'] / mask.size[0])
    
    pts = np.argwhere(mask)
    pts=pts*upscale_factor
    xbins=math.ceil((slide_data['width']-slide_data['stride_at_base'])/slide_data['tile_at_base'])
    ybins=math.ceil((slide_data['height']-slide_data['stride_at_base'])/slide_data['tile_at_base'])
    density,xbins,ybins=np.histogram2d(pts[:,1], pts[:,0], bins=[xbins,ybins],
                    range=[[int(slide_data['stride_at_base']//2), xbins*slide_data['tile_at_base']+int(slide_data['stride_at_base']//2)], [int(slide_data['stride_at_base']//2), ybins*slide_data['tile_at_base']+int(slide_data['stride_at_base']//2)]])
    return np.argwhere(density)*slide_data['tile_at_base']

def get_slide(sname,sformat,fpath,mag,kernel_size,region_size,threshold,outdir,poly_simplify_tolerance,logger):
    """
    Gather data for a specific slide and set up parameters for processing.

    This function retrieves metadata and initializes parameters for a given slide,
    preparing it for nuclei detection.

    Parameters:
    sname (str): Slide name.
    sformat (str): Slide format (e.g., 'tif').
    fpath (str): File path to the slide.
    mag (float): Target magnification.
    kernel_size (int): Size of the kernel for processing.
    region_size (int): Size of the region to be processed.
    threshold (float): Minimum size threshold for nuclei area in square micrometers.
    outdir (str): Output directory.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    logger (logging.Logger): Logger for logging messages.

    Returns:
    dict: A dictionary containing slide data and processing parameters.
    """
    slide_data = {}
    slide_data['format'] = sformat
    slide_data['sname'] = sname
    slide_data['fpath'] = fpath

    osh = openslide.open_slide(os.path.join(slide_data['fpath'],slide_data['sname']+f".{slide_data['format']}"))
    slide_data['xb'] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
    slide_data['yb'] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
    slide_data['width'] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, osh.level_dimensions[0][0]))
    slide_data['height'] = int(osh.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, osh.level_dimensions[0][1]))
    mpp_value = osh.properties.get('openslide.mpp-x')
    
    if mpp_value is None:
        mpp = 0.245
        logger.warning("WARNING. The MPP value was not found; using default value of 0.245.")
    else:
        mpp = float(mpp_value)

    slide_data['base_mag'] = base_mag = magnification_from_mpp(mpp)

    if base_mag < mag:
        raise ValueError(f"ERROR: Base magnification level lower than {mag}X detected.")
    slide_data['mpp'] = mpp
    slide_data['downfactor'] = downfactor = base_mag/mag

    level_downsamples = np.array(osh.level_downsamples,int)
    slide_data['level'] = level = np.argwhere(level_downsamples <= downfactor).reshape(-1)[-1]
    slide_data['working_d'] = level_downsamples[level]
    
    slide_data['kernel_size'] = kernel_size
    slide_data['stride'] = slide_data['kernel_size']//2
    slide_data['stride_at_base'] = int(slide_data['stride']*downfactor)
    slide_data['region_size'] = region_size
    slide_data['region_at_base'] = int(slide_data['region_size']*downfactor)
    slide_data['tile_size'] = region_size - slide_data['stride']
    slide_data['tile_at_base'] = int(slide_data['tile_size']*downfactor)
    slide_data['poly_simplification'] = poly_simplify_tolerance
    slide_data['threshold'] = threshold/(mpp**2)
    slide_data['outdir'] = outdir

    return slide_data

def infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain,logger,db_output_fname):
    """
    Perform nuclei detection on a whole slide image (WSI) using streaming DataLoader.

    This function processes a WSI for nuclei detection by streaming patches via DataLoader,
    performing inference using a pre-trained model, and saving the detected features.

    Parameters:
    sname (str): Slide name.
    sformat (str): Slide format (e.g., 'tif').
    fpath (str): File path to the slide.
    mask_dir (str or None): Directory containing quality control masks. If None, a tissue mask is created from the WSI.
    outdir (str): Output directory.
    mag (float): Target magnification.
    batch_to_gpu (int): Target batch size for GPU.
    region_size (int): Size of the region to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    n_process (int): Number of processes to use for multiprocessing.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    threshold (float): Minimum size threshold for nuclei area in square micrometers.
    stain (str): Staining type ('he' or 'ihc_dab').
    logger (logging.Logger): Logger for logging messages.

    Returns:
    tuple: Total number of regions processed and total number of nuclei detected.
    """
    
    n_post_proc = max(1, n_process // 2)   
    n_loader = max(1, n_process - n_post_proc)
    
    kernel_size = 256
    slide_data = get_slide(sname,sformat,fpath,mag,kernel_size,region_size,threshold,outdir,poly_simplify_tolerance,logger)
    
    coords = find_regions(mask_dir,slide_data)
    print(f"|- Computation starting on {len(coords)} patches.")

    # 1. Setup Dataset and DataLoader
    dataset = WSIPatchDataset(coords, slide_data)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_to_gpu, 
        shuffle=False, 
        num_workers=n_loader, 
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
         
    )

    # 2. Setup Output Queue and Writer
    if db_output_fname:
        conn = get_spatialite_connection(db_output_fname)
        init_spatialite_db_deferred_index(conn, srid=0)
    else: 
        features_queue = multiprocessing.Manager().Queue()
        writer_process = multiprocessing.Process(target=writer, args=(features_queue, os.path.join(outdir, sname + ".json.gz")))
        writer_process.start()

    # 3. Setup Post-Processing Pool
    post_proc_pool = multiprocessing.Pool(processes=n_post_proc)
    
    total_objects = 0
    async_results = []


    with torch.inference_mode():
        for batch_imgs, batch_coords_tensor in tqdm(loader, desc="Streaming Inference", leave=False):
            
            # --- GPU Inference ---
            batch_imgs = batch_imgs.to(device, memory_format=torch.channels_last, non_blocking=True) #move over as uint8 - faster than 32 and 16
            batch_imgs = batch_imgs.half().div(255.0)
            
            if stain == "ihc_dab":
                output_mask, maps = predict_ihc_batch(batch_imgs, model, device)
            else:
                output_mask, maps = predict_batch(batch_imgs, model)


            # --- Move to CPU for Post-Processing ---
            #---this works and is a more sophistocated than the regular sync - and seems to give almost no added value
            # consider reverting to a previous version in this pull request after benchmarking on the sever
            copy_stream = torch.cuda.Stream()

            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                output_cpu = output_mask.to("cpu", non_blocking=True)
                maps_cpu = maps.to("cpu", non_blocking=True)

                # Tell the allocator these source tensors are still in use by
                # copy_stream, so it won't recycle their memory early.
                output_mask.record_stream(copy_stream)
                maps.record_stream(copy_stream)
            copy_event = torch.cuda.Event()
            copy_event.record(copy_stream)

            coords_cpu = batch_coords_tensor.clone()

            copy_event.synchronize()

            output_cpu.share_memory_()
            maps_cpu.share_memory_()
            #-----  sync block completed

            if db_output_fname:
                res = post_proc_pool.apply_async(post_processing_batch_task,args=(output_cpu, maps_cpu, coords_cpu, slide_data, None, db_output_fname))
            else:
                res = post_proc_pool.apply_async(post_processing_batch_task,args=(output_cpu, maps_cpu, coords_cpu, slide_data, features_queue))

            async_results.append(res)

    # Wait for all post-processing to finish
    post_proc_pool.close()
    post_proc_pool.join()
    
    # Sum up results
    for res in async_results:
        total_objects += res.get()

    # Finish writing

    if db_output_fname:
        pass
        #conn = get_spatialite_connection(db_output_fname) 
        #build_spatial_indexes(conn) #start building the spatial indexes
    else:
        features_queue.put(None)
        writer_process.join()

    return len(coords), total_objects

def main_wsi(args) -> None:
    """
    Main entry point for nuclei detection on whole slide images (WSI).

    This function parses the command-line arguments, sets up the logger, loads the pre-trained model,
    and processes each slide for nuclei detection.

    Parameters:
    args (argparse.Namespace): Command-line arguments.
    """
    print(args)
    slide_dirs = args.slide_folder
    outdir = args.outdir
    mask_dir = args.binmask_dir
    mag = args.magnification
    batch_to_gpu = args.batch_gpu  
    # batch_mem is no longer needed with DataLoader streaming
    region_size = args.tile_size
    n_process = args.n_process
    model_path = args.model_path
    poly_simplify_tolerance = args.poly_simplify
    threshold = args.size_threshold
    stain = args.stain

    db_output = args.db_output

    multiprocessing.set_start_method('fork', force=True)
    
    if n_process is None:
        n_process = os.cpu_count()

    os.makedirs(outdir, exist_ok=True)

    logger = logging.getLogger(f"{outdir}/HoverFast_log_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%Hh%M"))
    f_handler = logging.FileHandler(f"{outdir}/HoverFast_log_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%Hh%M")+".log")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    
    if len(slide_dirs) == 1:
        pattern = slide_dirs[0]

        if glob.has_magic(pattern):
            slide_dirs = glob.glob(pattern)
        else:
            slide_dirs = [pattern]

    if not slide_dirs:
        logger.error("No slides detected.")
        raise ValueError("No slides detected.")

    stats={}

    def preload_file_linux(file_path: str):
        """Signals the Linux page cache to asynchronously preload the file into RAM."""
        fd = os.open(file_path, os.O_RDONLY)
        try:
            # Get total file size
            file_size = os.fstat(fd).st_size
            
            # POSIX_FADV_WILLNEED asks kernel to preload the range [0, file_size]
            os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(fd)
        return 0

    preload_file_linux(slide_dirs[0])


    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    #torch.backends.cudnn.benchmark=True
    model = load_model(model_path,device)

    for si,slide_dir in enumerate(slide_dirs):
        if si+1 < len(slide_dirs):
            preload_file_linux(slide_dirs[si+1]) #while processing this slide, start loading the next one

        temp = os.path.basename(slide_dir).rpartition('.')
        sname, sformat = temp[0],temp[-1]
        fpath = os.path.dirname(slide_dir)
        print(f"- Working on {sname}")
        stats[sname]=[]
        if db_output:
            db_output_fname=outdir+f"/{sname}.sqlite"  #TODO: use path object
        else:
            db_output_fname=None
        try:
            start=time.time()
            # Note: batch_mem argument removed from call
            n_patches, n_objects = infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain, logger,db_output_fname)
            stats[sname].append(n_patches)
            stats[sname].append(n_objects)
            stats[sname].append(time.time()-start)
            print(f"running time: {stats[sname][-1]:.2f}s, #patches {stats[sname][0]} and #objects {stats[sname][1]}")
        except Exception:
            logger.exception("File %s failed", sname)
            print(f"Error processing {sname}: {e}")

