import numpy as np
import multiprocessing
from shapely.geometry import Polygon
import gzip
import ujson
from functools import partial
from PIL import Image
import torch
from .hoverfast import HoverFast
import scipy.ndimage as ndi
import cv2
import math
from skimage.segmentation import watershed
from shapely.validation import make_valid
from tqdm import tqdm
import os
from skimage.measure import regionprops
from matplotlib.colors import to_rgb
import logging
import datetime
import glob
import time


def load_model(model_path,device):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = HoverFast(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"], conv_block=checkpoint["conv_block"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_dict"])
    model = model.half()  # Convert the model to float16
    model.eval()
    return model

int_coords = lambda x: np.array(x).round().astype(np.int32)

def divide_batch(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

def multiproc(function,arg_list,n_process,output=True):
    pool = multiprocessing.Pool(n_process)
    out = list(pool.imap(function,list(divide_batch(arg_list,int(np.ceil(len(arg_list)/n_process))))))
    pool.close()
    pool.join()
    if not output:
        return
    return sum(out,[]) if isinstance(out[0],list) else out

def load_roi(spaths):
    out = []
    for spath in spaths:
        region = np.asarray(Image.open(spath))
        out.append(np.copy(region))
    return out

def predict(regions, model, device):
    stride = 168

    # Transfer regions to GPU as a torch tensor
    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
    regions_shape = regions_gpu.shape
    # Padding operations on GPU
    regions_gpu = torch.nn.functional.pad(regions_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode='reflect')

    # Scale and transpose
    regions_gpu = regions_gpu.permute(0, 3, 1, 2) / 255

    # Model prediction
    output, maps = model(regions_gpu)
    # Post-processing
    output_processed = output[:, :, stride // 2:, stride // 2:][:, :, :regions_shape[1], :regions_shape[2]].argmax(axis=1).type(torch.bool)
    maps_processed = maps[:, :, stride // 2:, stride // 2:][:, :, :regions_shape[1], :regions_shape[2]]

    # Move the final results to CPU and convert to numpy
    output_cpu = output_processed.detach().cpu().numpy()
    maps_cpu = maps_processed.detach().cpu().numpy()
    maps_final = maps_cpu.astype(np.float32)

    return output_cpu,maps_final

def pre_watershed(output_mask,maps):
    if np.all(output_mask == 0):
        return None, None, None

    # Normalizing maps
    h_dir = cv2.normalize(maps[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(maps[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Sobel operations
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=5)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=5)
    del h_dir, v_dir

    # Normalizing sobel results and calculating overall gradient magnitude
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    overall = np.sqrt(sobelh ** 2 + sobelv ** 2)
    del sobelh, sobelv

    # Using the original kernel size
    opening = output_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=2)


    np.subtract(overall, 1 - opening, out=overall)  # In-place operation
    np.maximum(overall, 0, out=overall)  # In-place operation

    # Preparing for watershed
    dist = (1.0 - overall) * opening
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall = (overall >= 0.4).astype(np.int32)

    marker = opening - overall
    np.maximum(marker, 0, out=marker)  # In-place operation
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker, _ = ndi.label(marker)
    del overall

    return dist, marker, opening

def watershed_object(rg,dist,submarker,opening,offset,poly_simplify_tolerance,threshold):
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
        if poly_simplify_tolerance != 0:
            c = cv2.approxPolyDP(c,poly_simplify_tolerance*cv2.arcLength(c,True)/1000,True)
        if cv2.contourArea(c) <= threshold:
            continue
        poly = Polygon(c.squeeze())
        if not poly.is_valid:
            poly = make_valid(poly)
            while poly.geom_type != 'Polygon':
                poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
            bound = poly.boundary
            if bound.geom_type != 'LineString':
                bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
            c = np.array(bound.coords[:],int).squeeze()
        output.append(c.squeeze())
    return output

def region_feature(region,output_mask,dist, marker, opening,poly_simplify_tolerance, threshold,color,width,outdir,sname):
    rgs = regionprops(ndi.label(output_mask)[0])
    label = np.zeros_like(output_mask,np.int32)
    img = np.copy(region)
    color_rgb = (np.array(to_rgb(color))*255).astype(int).tolist()
    features = []
    for rg in rgs:
        if rg.area < threshold:
            continue
        ymin,xmin,ymax,xmax = rg.bbox
        temp = watershed_object(rg,dist[ymin:ymax,xmin:xmax],marker[ymin:ymax,xmin:xmax]*rg.image,opening[ymin:ymax,xmin:xmax],(xmin,ymin),poly_simplify_tolerance,threshold)
        for i in range(len(temp)):
            poly = temp[i]
            cv2.polylines(img, [poly],-1,color_rgb,width)
            cv2.fillPoly(label,[poly],len(features)+i)
        features += [save_poly(poly.astype(float)) for poly in temp]
    #save to Json
    with gzip.open(os.path.join(outdir,'json', sname + ".json.gz"), 'wt', encoding="ascii") as zipfile:
        if len(features)!=0:
            ujson.dump(features,zipfile)
    Image.fromarray(img).save(os.path.join(outdir,'overlay',sname+f'_overlay.png'))
    Image.fromarray(label).save(os.path.join(outdir,'label_mask',sname+f'_label_mask.png'))

def processing(regions,names,model,device,batch_to_gpu):
    arg_list1 = []
    for rgs in tqdm(divide_batch(regions,batch_to_gpu),desc="inner",leave=False, total = math.ceil(len(regions)/batch_to_gpu)):
        output_mask, maps = predict(rgs, model, device)
        arg_list1 += [(output_mask[j], maps[j], rgs[j]) for j in range(len(output_mask))]
    torch.cuda.empty_cache()
    return list(map(tuple.__add__, arg_list1, map(lambda x:(x,) ,names)))

def post_processing(data,outdir,threshold,poly_simplify_tolerance,color,width):
    for binmask,maps,region,sname in data:
        dist, marker, opening = pre_watershed(binmask,maps)
        region_feature(region,binmask,dist, marker, opening,poly_simplify_tolerance, threshold,color,width,outdir,sname)

def save_poly(poly,object_class = {'name': 'Nuclei', 'colorRGB': -65536}):
    feature = {}
    feature["geometry"] = {'type':'Polygon','coordinates':(tuple(map(tuple,poly.squeeze()))+(tuple(poly[0].squeeze()),),)}
    feature["properties"] = {'object_type': 'cell',
                                'classification': object_class,
                                'isLocked': False}
    feature["type"] = "Feature"
    return feature

def infer_roi(spaths,n_process,outdir,threshold,poly_simplify_tolerance,color,model,device,batch_to_gpu,width):
    regions = np.array(multiproc(load_roi,spaths,n_process))
    names = [os.path.basename(spath).rpartition('.')[0] for spath in spaths]
    arg_list = processing(regions, names,model,device,batch_to_gpu)
    multiproc(partial(post_processing,outdir=outdir,threshold=threshold,poly_simplify_tolerance=poly_simplify_tolerance,color=color,width=width),arg_list,n_process,output=False)

def main_roi(args) -> None:

    #get args

    slide_dirs = args.slide_folder
    outdir = args.outdir
    n_process = args.n_process
    model_path = args.model_path
    poly_simplify_tolerance = args.poly_simplify
    threshold = args.size_threshold
    batch_mem = args.batch_mem
    width = args.width
    batch_to_gpu = args.batch_gpu
    color = args.color
    if n_process is None:
        n_process = os.cpu_count()

    os.mkdir(outdir) if not os.path.exists(outdir) else None
    os.mkdir(os.path.join(outdir,'label_mask')) if not os.path.exists(os.path.join(outdir,'label_mask')) else None
    os.mkdir(os.path.join(outdir,'overlay')) if not os.path.exists(os.path.join(outdir,'overlay')) else None
    os.mkdir(os.path.join(outdir,'json')) if not os.path.exists(os.path.join(outdir,'json')) else None

    #config logger
    logger = logging.getLogger(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"))

    f_handler = logging.FileHandler(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")+".log")
    c_handler = logging.StreamHandler()

    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=True
    model = load_model(model_path,device)

    #get input files
    if len(slide_dirs)==1:
        #input is a glob pattern
        slide_dirs = glob.glob(slide_dirs[0])

    start=time.time()
    for spaths in tqdm(divide_batch(slide_dirs,batch_mem),desc="outer",leave=False,total=math.ceil(len(slide_dirs)/batch_mem)):
        try:
            infer_roi(spaths,n_process,outdir,threshold,poly_simplify_tolerance,color,model,device,batch_to_gpu,width)
        except Exception as e:
            logger.error(f"Batch {slide_dirs} failed: {e}", exc_info=True)
    end = time.time()-start
    print(f"running time: {end}, #patches {len(slide_dirs)}")
