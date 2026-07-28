from pathlib import Path
import numpy as np
import multiprocessing
import openslide
from shapely.geometry import Polygon
import gzip
import ujson
from functools import partial
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .hoverfast import HoverFast
import scipy.ndimage as ndi
import cv2
import math
from skimage.segmentation import watershed
from shapely.validation import make_valid
from tqdm import tqdm
import os
from skimage.measure import regionprops
import logging
import datetime
import glob
import time
from skimage.color import rgb2hed, hed2rgb
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
    checkpoint = torch.load(model_path, weights_only=False, map_location=lambda storage, loc: storage)
    model = HoverFast(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                      padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                      up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"], conv_block=checkpoint["conv_block"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_dict"])
    model = model.half()  # Convert the model to float16 (half precision) for faster inference
    model.eval()
    return model

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

def save_poly(poly,centroid,object_class = {'name': 'Nuclei', 'colorRGB': -65536}):
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
    feature = {}
    feature["geometry"] = {'type':'Polygon','coordinates':(tuple(map(tuple,poly.squeeze()))+(tuple(poly[0].squeeze()),),)}
    feature["geometry"]["centroid"] = [ int(coord) for coord in centroid]
    feature["properties"] = {'object_type': 'cell',
                                'classification': object_class,
                                'isLocked': False}
    feature["type"] = "Feature"
    return ujson.dumps(feature)

def writer(features_queue, output_path):
    """
    Save detected features to a JSON file.

    This function reads features from the provided queue and saves them to a compressed JSON file.

    Parameters:
    features_queue (multiprocessing.Queue): Queue containing the extracted features.
    output_path (str): Path to the output JSON file.
    """
    with gzip.open(output_path, 'wt', encoding="utf-8") as file:
            file.write('[')
            first = True
            while True:
                try:
                    feature = features_queue.get()
                except:
                    continue
                if feature is None: # end of analysis
                    break
                file.write(","*(not first)+'\n'+feature)
                first = False
            file.write('\n]')

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
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        
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
    
    return output_processed, maps

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

def watershed_object(rg, dist, submarker, opening, offset, region_coord, slide_data):
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
        output.append(save_poly((c*slide_data['downfactor'])+region_coord,slide_data['downfactor']*np.array([cx,cy])+region_coord))
    return output

def region_feature(output_mask, region_coord, dist, marker, opening, slide_data):
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
        output += watershed_object(rg,dist[ymin:ymax,xmin:xmax],marker[ymin:ymax,xmin:xmax]*rg.image,opening[ymin:ymax,xmin:xmax],(xmin,ymin),region_coord,slide_data)
    
    return output

def post_processing_batch_task(batch_data, slide_data, features_queue):
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
    total_features = 0
    for output_mask, maps, region_coord in batch_data:
        dist, marker, opening = pre_watershed(output_mask, maps)
        if marker is None:
            continue
        
        # Extract features
        features = region_feature(output_mask, region_coord, dist, marker, opening, slide_data)
        
        if features:
            # Join them into a string and put in queue
            features_queue.put(",\n".join(features))
            total_features += len(features)
            
    return total_features

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

def infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain,logger):
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
        prefetch_factor=2 
    )

    # 2. Setup Output Queue and Writer
    features_queue = multiprocessing.Manager().Queue()
    writer_process = multiprocessing.Process(target=writer, args=(features_queue, os.path.join(outdir, sname + ".json.gz")))
    writer_process.start()

    # 3. Setup Post-Processing Pool
    post_proc_pool = multiprocessing.Pool(processes=n_post_proc)
    
    total_objects = 0
    async_results = []

    with torch.no_grad():
        for batch_imgs, batch_coords_tensor in tqdm(loader, desc="Streaming Inference", leave=False):
            
            # --- GPU Inference ---
            batch_imgs = batch_imgs.to(device, memory_format=torch.channels_last).half()
            
            if stain == "ihc_dab":
                output_mask, maps = predict_ihc_batch(batch_imgs, model, device)
            else:
                output_mask, maps = predict_batch(batch_imgs, model)
            
            # --- Move to CPU for Post-Processing ---
            output_cpu = output_mask.cpu().numpy()
            maps_cpu = maps.cpu().numpy().astype(np.float32)
            coords_cpu = batch_coords_tensor.numpy()
            
            # Prepare batch data for the pool
            batch_data = []
            for i in range(len(output_cpu)):
                batch_data.append((output_cpu[i], maps_cpu[i], coords_cpu[i]))
            
            # --- Async Post-Processing ---
            res = post_proc_pool.apply_async(
                post_processing_batch_task, 
                args=(batch_data, slide_data, features_queue)
            )
            async_results.append(res)

    # Wait for all post-processing to finish
    post_proc_pool.close()
    post_proc_pool.join()
    
    # Sum up results
    for res in async_results:
        total_objects += res.get()

    # Finish writing
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

    multiprocessing.set_start_method('spawn', force=True)
    
    if n_process is None:
        n_process = os.cpu_count()

    os.makedirs(outdir, exist_ok=True)

    logger = logging.getLogger(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"))
    f_handler = logging.FileHandler(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")+".log")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=True
    model = load_model(model_path,device)

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

    for slide_dir in slide_dirs:
        temp = os.path.basename(slide_dir).rpartition('.')
        sname, sformat = temp[0],temp[-1]
        fpath = os.path.dirname(slide_dir)
        print(f"- Working on {sname}")
        stats[sname]=[]
        
        try:
            start=time.time()
            # Note: batch_mem argument removed from call
            n_patches, n_objects = infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain, logger)
            stats[sname].append(n_patches)
            stats[sname].append(n_objects)
            stats[sname].append(time.time()-start)
            print(f"running time: {stats[sname][-1]:.2f}s, #patches {stats[sname][0]} and #objects {stats[sname][1]}")
        except Exception as e:
            logger.error(f"File {sname} failed: {e}", exc_info=True)
            print(f"Error processing {sname}: {e}")

