import os
import math
import numpy as np
import tifffile
import zarr
from tqdm import tqdm
from aicsimageio import AICSImage

from skimage import filters, measure
import matplotlib.pyplot as plt
import warnings

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from segmentation_utils import *

def find_flex_files(directory):
    """Recursively searches for .flex files in the given directory."""
    flex_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".flex"):
                flex_files.append(os.path.join(root, file))
    return sorted(flex_files)


def read_flex_image(file_path):
    """Reads .flex file and extracts Z-stack images."""
    img    = AICSImage(file_path)
    stack  = img.get_image_data("ZYX", T=0, C=0)  # Extract as Z-stack
    images = [stack[z, :, :] for z in range(stack.shape[0])]  # Split by Z-sections
    return images


def estimate_flat_field_from_array(image_array):
    """Estimate flat field correction directly from a list of NumPy arrays."""
    height, width  = image_array[0].shape
    mean_intensity = np.mean(image_array, axis=0)
    flat_field     = mean_intensity / np.mean(mean_intensity)
    flat_field     = filters.gaussian(flat_field, sigma=64)
    return flat_field


def get_flat_fields(input_folder):
    """
    Read .flex files in the specified input folder and compute flat field correction.
    
    Parameters:
    - input_folder: str, path to the folder containing .flex files
    """  
    flex_files = find_flex_files(input_folder)  # Get all .flex files
    
    print("#====================================================#")
    print("Processing FLEX files from:", input_folder)
    print("#====================================================#")
    
    mean_dapi = []
    mean_fib  = []
    for flex_file in tqdm(flex_files): 
        
        images    = read_flex_image(flex_file)  # Read Z-stack images
        file_name = os.path.splitext(os.path.basename(flex_file))[0]
    
        totalImgs        = len(images)
        subplotsPerImage = 6  # 3 sections x (DAPI + Fibrillarin)
        numImg           = (totalImgs + subplotsPerImage - 1) // subplotsPerImage  
        
        dapi_images        = []
        fibrillarin_images = []
        
        for idx in range(numImg):
            # Merge sections 
            startIdx    = idx * subplotsPerImage 
            fibrillarin = np.maximum(images[startIdx], images[startIdx+2], images[startIdx+4])
            dapi        = np.maximum(images[startIdx+1], images[startIdx+3], images[startIdx+5])
            
            dapi_images.append(dapi)
            fibrillarin_images.append(fibrillarin)
        
        mean_dapi.append(np.mean(dapi_images, axis=0))
        mean_fib.append(np.mean(fibrillarin_images, axis=0))
        
    hoechst_flat_field = estimate_flat_field_from_array(mean_dapi)
    egfp_flat_field    = estimate_flat_field_from_array(mean_fib)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Display images
    axes[0].imshow(hoechst_flat_field, cmap="gray")
    axes[0].set_title("hoechst_flat_field")
    axes[0].axis("off")
    axes[1].imshow(egfp_flat_field, cmap="gray")
    axes[1].set_title("egfp_flat_field")
    axes[1].axis("off")
    # Show the plot
    plt.show()
    """
        
    return hoechst_flat_field, egfp_flat_field



def get_nuclei(flex_file, hoechst_flat_field, egfp_flat_field):
    """
    Read a .flex file and return interleaved crops:
    [EGFP_1, DAPI_1, EGFP_2, DAPI_2, ...]
    """  
    images = read_flex_image(flex_file)

    totalImgs        = len(images)
    subplotsPerImage = 6  # 3 sections x (DAPI + Fibrillarin)
    numImg           = (totalImgs + subplotsPerImage - 1) // subplotsPerImage  

    interleaved = []

    for idx in range(numImg):
        startIdx    = idx * subplotsPerImage 
        fibrillarin = np.maximum(images[startIdx], images[startIdx+2], images[startIdx+4])
        dapi        = np.maximum(images[startIdx+1], images[startIdx+3], images[startIdx+5])

        result = extract_nuclei(
            dapi, fibrillarin, 
            hoechst_flat_field=hoechst_flat_field, 
            egfp_flat_field=egfp_flat_field
        )
        if result is None:
            continue

        egfp_stack, dapi_stack = result
        N = egfp_stack.shape[0]

        for i in range(N):
            interleaved.append(egfp_stack[i])
            interleaved.append(dapi_stack[i])

    if len(interleaved) == 0:
        return None

    return interleaved
        
################################################################################### 
# HELPER FUNCTION TO READ ZIP FILES
################################################################################### 


def _has_key(g, key):
    try:
        _ = g[key]
        return True
    except Exception:
        return False

def load_from_zip(
    zip_filename,
    batch_size=1,
    start=0,
    read_all=False,
    channel="both",
):
    """
    Load images from a Zarr store inside a ZIP file.
    Dataset format:
       array_0, array_1, array_2, ...
       even index = EGFP
       odd index  = DAPI
       pairs = (0,1), (2,3), ...
    """

    ch_req = str(channel).upper()
    if ch_req == "GFP":
        ch_req = "EGFP"
    if ch_req not in {"BOTH", "DAPI", "EGFP"}:
        raise ValueError("channel must be both, DAPI, or EGFP")

    want_both = (ch_req == "BOTH")

    zstore = zarr.ZipStore(zip_filename, mode="r")
    try:
        root = zarr.open_group(zstore, mode="r")

        # detect array_i
        ids = []
        i = 0
        while _has_key(root, f"array_{i}"):
            ids.append(i)
            i += 1

        if len(ids) == 0:
            C = 2 if want_both else 1
            return np.empty((0, C, 96, 96)), 0, 0

        # =====================================================================
        # ✔ OPTION 1 : READ ALL IMAGES NOW — IGNORE start and batch_size
        # =====================================================================
        if read_all:
            pairs = []
            i = 0
            while i < len(ids):
                idx = ids[i]
                arr = root[f"array_{idx}"][:].astype(np.float32)
                arr = arr.reshape(1, 96, 96)

                if want_both:
                    # need a pair
                    if i + 1 >= len(ids):
                        break
                    arr2 = root[f"array_{idx+1}"][:].astype(np.float32)
                    arr2 = arr2.reshape(1, 96, 96)
                    pairs.append(np.stack([arr[0], arr2[0]], axis=0))
                    i += 2
                else:
                    # single channel mode
                    ch_here = "EGFP" if (idx % 2 == 0) else "DAPI"
                    if ch_here == ch_req:
                        pairs.append(arr)
                    i += 1

            if len(pairs) == 0:
                C = 2 if want_both else 1
                return np.empty((0, C, 96, 96)), 0, 0

            arr = np.stack(pairs, axis=0)
            return arr, 0, arr.shape[0]

        # =====================================================================
        # ✔ OPTION 2 : NORMAL BATCH LOADING (read_all=False)
        # =====================================================================
        pairs = []
        count = 0
        i = start

        while i < len(ids) and count < batch_size:
            idx = ids[i]
            arr = root[f"array_{idx}"][:].astype(np.float32)
            arr = arr.reshape(1, 96, 96)

            if want_both:
                if i + 1 >= len(ids):
                    break
                arr2 = root[f"array_{idx+1}"][:].astype(np.float32)
                arr2 = arr2.reshape(1, 96, 96)
                pairs.append(np.stack([arr[0], arr2[0]], axis=0))
                i += 2
            else:
                ch_here = "EGFP" if (idx % 2 == 0) else "DAPI"
                if ch_here == ch_req:
                    pairs.append(arr)
                i += 1

            count += 1

        if len(pairs) == 0:
            C = 2 if want_both else 1
            return np.empty((0, C, 96, 96)), 0, 0

        arr = np.stack(pairs, axis=0)
        return arr, 0, arr.shape[0]

    finally:
        zstore.close()




################################################################################### 
# HELPER FUNCTION TO READ ZIP FILES (END)
################################################################################### 



