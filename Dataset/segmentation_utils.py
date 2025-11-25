import os
import warnings
import random
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from tqdm import tqdm

import tifffile

from skimage import io
from skimage import util
from skimage import measure
from skimage import morphology 
from skimage import filters 
from skimage import segmentation 

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
    
    return hoechst_flat_field, egfp_flat_field


def estimate_flat_field(image_paths):
    """Estimate flat field from a given set of images. Flat field is estimated
    taking the piwel-wise mean than applying a gaussian filter.

    Parameters
    ----------
        image_paths : list of strings
            List of paths to images from which to estimate flat field.

    Returns
    -------
        flat_field : ndarray of shape (H, W)
            Flat field estimate. Same shape as input images.
    """
    # Extract height and width of images (assuming all provided images have 
    # the same size)
    height, width = io.imread(image_paths[0]).shape

    # Initialize mean image
    mean_intensity = np.zeros((height, width))

    # Compute mean
    for path in image_paths:
        img = util.img_as_float(io.imread(path))
        mean_intensity += img
    mean_intensity = mean_intensity/len(image_paths)

    # Normalize mean image by mean intensity of mean image
    flat_field = mean_intensity/np.mean(mean_intensity)

    # Apply gaussian filter
    flat_field = filters.gaussian(flat_field, sigma=64)

    return flat_field


def preprocess_image(image, flat_field=None, blur=False):
    """Preprocess the input image by applying flat field correction and optional 
    Gaussian blur. 

    Parameters
    ----------
        image : ndarray of shape (H, W)
            Input image to preprocess. 
        
        flat_field : ndarray of shape (H, W), optional (default: None)
            Flat field used for intensity correction. 
        
        blur : bool, optional (default: False)
            Whether to apply Gaussian blur for denoising. Default is False.
            Gaussian blur is used for denoising. 
      
    Returns
    -------
        preprocessed_image : ndarray
            Preprocessed image.

    Notes
    -----
    Converts image type to floats in the range [-1.0,1.0] or [0.0,1.0] 
    depending on the input type (signed or unsigned). 
    See skimage.util.img_as_float for more details.

    Todo
    ----
    To think about: Add some kind of intensity normalization to take into 
    account variability across well samples. How to differentiate intensity 
    variations due to random perturbations and/or experimental conditions from 
    intensity variations due to treatment action ?
    """
    # Convert image to floats. Assuming uint16 as input (unsigned int), 
    # the range of the new floating points image is [0.0, 1.0].
    prepocessed_image = util.img_as_float(image, force_copy=True)
    
    # Applies flat field correction if specified
    if flat_field is not None:
        prepocessed_image = prepocessed_image / flat_field
        prepocessed_image = np.clip(prepocessed_image, 0.0, 1.0) 

    # Apply Gaussian blur if specified 
    if blur:
        prepocessed_image = filters.gaussian(
            prepocessed_image, 
            sigma=2, 
            preserve_range=True,
        )

    return prepocessed_image


def segment_nuclei(image):
    """Segment nuclei in the input image (Hoechst-33342 channel) 
    using adaptive thresholding and apply morphological operations to clean 
    the obtained binary mask.

    Parameters
    ----------
        image : ndarray of shape (H, W) 
            Input image to segment (Hoechst-33342 channel).

    Returns
    -------
        nuclei_regions : integer ndarray of shape (H, W) 
            Labeled regions corresponding to the segmented nuclei.
    """
    # Compute the binary mask
    # - compute local thresholds (using gaussian kernel)
    adaptive_thresh = filters.threshold_local(image, block_size=201)
    # - compute local offsets (helps for removing noise from background)
    #offset = (np.percentile(image, q=50)/image)*np.percentile(image, q=1)
    std = np.std(image[image<np.percentile(image,q=50)])
    offset = std * (np.percentile(adaptive_thresh, q=50)/adaptive_thresh)
 
    # - retrieve binary mask
    binary_mask = image > adaptive_thresh + offset
 
    # Clean the binary mask
    # - remove small objects 
    cleaned_mask = morphology.remove_small_objects(
        binary_mask, 
        min_size=400,
    )
    # - fill holes
    cleaned_mask = morphology.remove_small_holes(
        cleaned_mask, 
        area_threshold=400,
    )
    # - remove objects on the border
    cleaned_mask = segmentation.clear_border(cleaned_mask)

    # Label each connected component (nuclei)
    nuclei_regions = measure.label(cleaned_mask)

    return nuclei_regions

def filter_nuclei(
    nuclei_regions, 
    hoechst_image, 
    egfp_image, 
    min_size=500, 
    max_size=5000,
    min_solidity=0.95, 
    max_eccentricity=0.8, 
    max_contrast_hoechst=0.9,
    min_contrast_egfp=0.3,
):
    """Filter the nuclei regions based on specified size, shape, and 
    intensity criteria. We want to keep only regular nuclei from cells in
    interphase. By "regular", we mean well defined nuclei in the hoechst-33342
    channel, which we expect to be of a plausible size, reasonably convex 
    (high solidity) and round (low eccentricity). These size and shape criteria 
    will also exclude nuclei that overlap (cf. Notes) and some nuclei from cells 
    that are not in interphase (indeed, in mitosis phase, the nuclear enveloppe 
    breaks down, which may lead to more or less irregular shapes in the 
    hoechst-33342 channel, depending on the stage of mitosis). Furthermore, 
    nuclei from cells in interphase should not exhibit strong DNA concentration 
    nor nucleoli dissolution (they should have low intensity contrast in the 
    hoechst-33342 channel but some contrast in the EGFP channel). We thus also 
    filter out nuclei based on a intensity contrast measure (cf. Parameters for 
    a precise formulation).
    

    Parameters
    ----------
        nuclei_regions : Integer ndarray of shape (H, W)
            Labeled regions representing candidate nuclei.
        
        hoechst_image: ndarray of shape (H, W)
            Hoechst-33342 channel image.
    
        egfp_image : ndarray of shape (H, W)
            EGFP channel image.

        min_size : int, optional (default: 500)
            Minimum expected size of the nuclei (in pixels).
        
        max_size : int, optional (default: 5000) 
            Maximum expected size of the nuclei (in pixels).

        min_solidity : float, optional (default : 0.95)
            Minimum expected solidity of the nuclei. Solidity is the ratio
            of the number of pixels contained in the nuclei region to the 
            number of pixels in the convex hull of this same region. 
            It measures convexity.

        max_eccentricity : float, optional (default: 0.8)
            Maximum expected eccentricity of the nuclei. 

        max_contrast_hoechst : float, optional (default: 0.9) 
            Maximum intensity contrast in the nuclei region for the
            Hoechst-33342 channel. Contrast is defined as the standard deviation 
            of pixel intensities in the nuclei region, normalized by the region 
            mean intensity (to obtain a contrast measure less dependent on the 
            intensity scale of images). 

        min_contrast_egfp : float, optional (default: 0.3) 
            Minimum intensity contrast in the nuclei region for the EFGP 
            channel. Contrast is defined as the standard deviation of pixel 
            intensities in the nuclei region, normalized by the region mean 
            intensity (to obtain a contrast measure less dependent on the 
            intensity scale of images). 

    Returns
    -------
        filtered_nuclei: 
            Labeled regions corresponding to the filtered nuclei.

        properties:

    Notes
    -----
    We could seperate overlapping nuclei with iterative thresholding (cf. work
    of Nicolas and Pascaline), a supplementary watershed segmentation step, 
    or other more elaborated methods but for now, we simply discard overlapping 
    nuclei (through solidity, eccentricity and size criteria). This eliminates 
    additional difficulties related to nuclei separation (over-fragmentation, 
    ambiguous boundaries, etc) and reduces computational load, while only 
    reducing slightly the amount of available data for downstream analysis.
    This seems like an acceptable trade-off because, in this case, we care more 
    about the quality of our segmentation than the completness (I assume we do,
    but it should perhaps be discussed/confirmed). 

    Similarly, we prefer to exclude some valid interphase nuclei rather than 
    accept irregular objects or nuclei from cells in mitosis. As a result, our 
    default criteria for defining a "regular" interphase nucleus are 
    intentionally restrictive.
    """
    # Measure nuclei regions geometric properties based on mask only.
    geometric_props = measure.regionprops_table(
        label_image=nuclei_regions,
        properties= ('label', 'area', 'solidity', 'eccentricity',),
    )
    geometric_props = pd.DataFrame(geometric_props)

    # Define contrast measure to be passed in the "extra_properties" argument 
    # of skimage.measure.regionprops_table for Hoechst-33343 and EGFP channels.
    def contrast(region_mask, intensity_image):
        """Compute region contrast as the intensity standard deviation divided
        by the intensity mean.
        """
        region_indexes = region_mask==1
        std = np.std(intensity_image[region_indexes])
        mean = np.mean(intensity_image[region_indexes])
        contrast = std/mean
        return contrast
    
    # Measure nuclei regions properties based on the Hoechst-33342 channel.
    hoechst_props = measure.regionprops_table(
        label_image=nuclei_regions,
        intensity_image=hoechst_image,
        properties=('label',),
        extra_properties =(contrast,),
    )
    hoechst_props = pd.DataFrame(hoechst_props)

    # Measure nuclei regions properties based the EGFP channel.
    egfp_props = measure.regionprops_table(
        label_image=nuclei_regions,
        intensity_image=egfp_image,
        properties=('label',),
        extra_properties= (contrast,),
    )
    egfp_props = pd.DataFrame(egfp_props)

    # Merge intensity properties from Hoechst-33342 and EGFP channels 
    intensity_props = pd.merge(
        hoechst_props,
        egfp_props,
        on='label',
        suffixes=['_hoechst', '_egfp']
    )

    # Merge geometric and intensity properties
    props = pd.merge(
        geometric_props, 
        intensity_props, 
        on='label'
    )
 
    # For each nucleus, evaluate each filtering criteria.
    filtering_criteria = {
        'min_size': props['area'] >= min_size,
        'max_size': props['area'] <= max_size,
        'min_solidity': props['solidity'] >= min_solidity,
        'max_eccentricity': props['eccentricity'] <= max_eccentricity,
        'max_contrast_hoechst': props['contrast_hoechst'] <= max_contrast_hoechst,
        'min_contrast_egfp': props['contrast_egfp'] >= min_contrast_egfp,
    }
    filtering_criteria = pd.DataFrame(filtering_criteria)

    # If all filtering criteria evaluate to True, then the nucleus is 
    # considered to be regular and in interphase (is_nucleus_valid = True).
    # If at least one of the criteria evaluates to False, the nucleus is 
    # filtered out (is_nucleus_valid = False).
    is_nucleus_valid = filtering_criteria.all(axis=1)
    
    # Compute new labeled mask for filtered nuclei. Takes input nuclei_regions 
    # and sets labels of rejected regions to zero. Other labels remain intact.
    filtered_nuclei = nuclei_regions.copy()
    rejected_labels = props['label'][~is_nucleus_valid]
    for label in rejected_labels:
        filtered_nuclei[nuclei_regions == label] = 0

    # Store results in case they are needed in downstream analysis
    props['is_valid'] = is_nucleus_valid
    filtering_criteria['is_valid'] = is_nucleus_valid

    return filtered_nuclei, props, filtering_criteria


def extract_nuclei(
    hoechst_image, 
    egfp_image, 
    pixel_height=96, 
    pixel_width=96,
    hoechst_flat_field=None, 
    egfp_flat_field=None):
    """Segment objects in Hoechst channel, filter nuclei and export 
    corresponding regions from the EGFP channel to individual images (nucleus 
    centered on black background).

    Parameters
    ----------
        hoechst_path : string
            Path to Hoechst-33342 channel image.

        egfp_path : string
            Path to EGFP channel image.

        pixel_height : int, optional (default = 96)
            Height in pixel of extracted image.

        pixel_width : int, optional (default = 96)
            Width in pixel of extracted image.

        hoechst_flat_field : ndarray of shape (H, W), optional (default = None)
            Flat field for intenisy correction in the Hoechst-33342 channel.
            Should be the same size as the image loaded from hoechst_path.

        egfp_flat_field : ndarray of shape (H, W), optional (default = None)
            Flat field for intenisy correction in the EGFP channel.
            Should be the same size as the image loaded from egfp_path.

        return_properties : bool, optional (default : False)
            Wether to return properties of extracted nuclei (as returned by
            filter_nuclei)

    Returns
    -------
        properties : pandas DataFrame, if return_properties == True
            Properties of extracted region, as returned by filter_nuclei
    """

    hoechst_preprocessed = preprocess_image(
        hoechst_image, 
        flat_field=hoechst_flat_field, 
        blur=True
    )

    egfp_preprocessed = preprocess_image(
        egfp_image, 
        flat_field=egfp_flat_field, 
        blur=False
    )

    candidate_nuclei = segment_nuclei(hoechst_preprocessed)

    filtered_nuclei, properties, _ = filter_nuclei(
        nuclei_regions=candidate_nuclei,
        hoechst_image=hoechst_preprocessed, 
        egfp_image=egfp_preprocessed
    )

    regions = measure.regionprops(
        label_image=filtered_nuclei,
        intensity_image=egfp_preprocessed,
    )

    egfp_out = []
    dapi_out = []

    for nucleus in regions:
        extracted_egfp = nucleus.image * nucleus.intensity_image

        h, w = extracted_egfp.shape
        if h > pixel_height or w > pixel_width:
            continue

        H, W = pixel_height, pixel_width
        cy, cx = H // 2, W // 2
        y0 = cy - h // 2
        x0 = cx - w // 2
        y1 = y0 + h
        x1 = x0 + w

        egfp_canvas = np.zeros((H, W), dtype=np.float64)
        egfp_canvas[y0:y1, x0:x1] = extracted_egfp

        minr, minc, maxr, maxc = nucleus.bbox
        hoechst_patch = hoechst_preprocessed[minr:maxr, minc:maxc]
        extracted_dapi = nucleus.image * hoechst_patch

        dapi_canvas = np.zeros((H, W), dtype=np.float64)
        dapi_canvas[y0:y1, x0:x1] = extracted_dapi

        egfp_out.append(egfp_canvas)
        dapi_out.append(dapi_canvas)

    if not egfp_out:
        warnings.warn("No nuclei found in the image.")
        return None

    egfp_stack = np.stack(egfp_out, axis=0)
    dapi_stack = np.stack(dapi_out, axis=0)
    return egfp_stack, dapi_stack




def save_nuclei(input_folder):
    """
    Read .flex files in the specified input folder and save nuclei in output_folder.
    
    Parameters:
    - input_folder: str, path to the folder containing .flex files
    """  
    flex_files = find_flex_files(input_folder)  # Get all .flex file
    flex_files[0:2]
    
    print("#====================================================#")
    print("Processing FLEX files from:", input_folder)
    print("#====================================================#")
    

    for flex_file in tqdm(flex_files):
        # In one iteration, one well is considered 
        images           = read_flex_image(flex_file)  # Read Z-stack images
        totalImgs        = len(images)
        subplotsPerImage = 6  # 3 sections x (DAPI + Fibrillarin)
        numImg           = (totalImgs + subplotsPerImage - 1) // subplotsPerImage  
        
        all_nuclei = []
        
        for idx in range(numImg):
            startIdx    = idx * subplotsPerImage 
            fibrillarin = np.maximum(images[startIdx], images[startIdx+2], images[startIdx+4])
            dapi        = np.maximum(images[startIdx+1], images[startIdx+3], images[startIdx+5])
            
            nuclei_stack = extract_nuclei(dapi, fibrillarin, 
                hoechst_flat_field=hoechst_flat_field, egfp_flat_field=egfp_flat_field)
            all_nuclei.append(nuclei_stack)
            
    return np.concatenate(all_nuclei, axis=0)


