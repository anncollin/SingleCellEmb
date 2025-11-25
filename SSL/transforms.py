from typing import Dict, Tuple, List, Optional
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as v2 


class Solarization(object):
    def __init__(self, p: float):
        self.p = p
        # v2.RandomSolarize works with torch.Tensor and PIL Image
        self.t = v2.RandomSolarize

    def __call__(self, img):
        if random.random() < self.p:
            # threshold is a hyperparameter; 0.5 is a common choice on [0,1]-scaled images
            return self.t(threshold=0.5)(img)
        else:
            return img
        
class ElasticDeformation(object):
    def __init__(self,
                 p: float,
                 alpha_range: Tuple[float, float],
                 sigma_range: Tuple[float, float]):
        self.p = p
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.t = v2.ElasticTransform

    def __call__(self, img):
        if random.random() < self.p:
            alpha = random.uniform(*self.alpha_range)
            sigma = random.uniform(*self.sigma_range)
            return self.t(alpha=alpha, sigma=sigma)(img)
        else:
            return img
        
class FloatEqualize:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return v2.functional.equalize(img)
        return img

class FloatContrast:
    def __init__(self, p: float, contrast_range: Tuple[float, float]):
        self.p = p
        self.low, self.high = contrast_range

    def __call__(self, img):
        if random.random() < self.p:
            c = random.uniform(self.low, self.high)
            return (img * c).clamp(0, 1)
        return img

#######################################################################################################
# MULTI-CROP TRANSFORM FOR DINO ON CELL IMAGES
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    image_size : int
#        Size of input images (assumed square, e.g. 96).
#
#    global_crops_scale : Tuple[float, float]
#        Relative scale range for global crops in RandomResizedCrop.
#
#    local_crops_scale : Tuple[float, float]
#        Relative scale range for local crops.
#
#    local_crops_number : int
#        Number of local crops to generate per image.
#
#    Returns:
#    ---------------------------
#    views : List[Tensor]
#        A list of augmented crops for the same input image. Each crop is a tensor (C, H, W).
#######################################################################################################


class MultiCropTransform:
    def __init__(
        self,
        image_size: int,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        cfg: Dict = None,
    ):
        self.local_crops_number = local_crops_number

        # ---------------------------------------------------------------------
        # 1. READ AUGMENTATION PARAMETERS FROM YAML
        # ---------------------------------------------------------------------
        aug = cfg.get("augment", {}) if cfg is not None else {}

        elastic_p = float(aug.get("elastic_p", 0.0))
        elastic_alpha_range = tuple(aug.get("elastic_alpha_range", [30.0, 50.0]))
        elastic_sigma_range = tuple(aug.get("elastic_sigma_range", [4.0, 6.0]))

        solarization_p = float(aug.get("solarization_p", 0.0))
        equalize_p = float(aug.get("equalize_p", 0.0))

        contrast_p = float(aug.get("contrast_p", 0.0))
        contrast_range = tuple(aug.get("contrast_range", [0.8, 1.2]))

        # rotation angles like [-180, 180] for full random rotation
        # or None if disabled
        rotation_degrees = aug.get("rotation_degrees", None)
        if rotation_degrees is not None:
            rotation_degrees = tuple(rotation_degrees)

        # ---------------------------------------------------------------------
        # 2. AUGMENTATIONS APPLIED TO BOTH GLOBAL AND LOCAL CROPS
        # ---------------------------------------------------------------------
        extra = []

        # Elastic deformation
        if elastic_p > 0:
            extra.append(ElasticDeformation(elastic_p, elastic_alpha_range, elastic_sigma_range))

        # Histogram equalization (float32-safe version)
        if equalize_p > 0:
            extra.append(FloatEqualize(equalize_p))

        # Contrast jitter (float32-safe version)
        if contrast_p > 0:
            extra.append(FloatContrast(contrast_p, contrast_range))

        # Solarization
        if solarization_p > 0:
            extra.append(Solarization(solarization_p))

        # Rootation
        if rotation_degrees is None:
            rotation = T.Identity()  
        else:
            rotation = T.RandomRotation(
                degrees=rotation_degrees,
                interpolation=T.InterpolationMode.BILINEAR,
            )

        # ---------------------------------------------------------------------
        # 4. GLOBAL CROP PIPELINE
        # ---------------------------------------------------------------------
        # Applied:
        # - RandomResizedCrop with global scale
        # - Random flips
        # - Rotation
        # - Gaussian blur
        # - *extra transforms (elastic, equalize, contrast, solarization)
        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=image_size,
                    scale=global_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                rotation,
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                *extra,
            ]
        )

        # ---------------------------------------------------------------------
        # 5. LOCAL CROP PIPELINE
        # ---------------------------------------------------------------------
        # Same as global, but:
        #   - Smaller size = image_size // 2
        #   - local_crops_scale for RandomResizedCrop
        local_size = image_size // 2

        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=local_size,
                    scale=local_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                rotation,
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                *extra,
            ]
        )

    # -------------------------------------------------------------------------
    # 6. APPLY TRANSFORMS
    # -------------------------------------------------------------------------
    def __call__(self, img):
        """
        Returns:
            [2 * global crops] + [local_crops_number * local crops]
        """
        views = []

        # Generate 2 global crops
        for _ in range(2):
            views.append(self.global_transform(img))

        # Generate N local crops
        for _ in range(self.local_crops_number):
            views.append(self.local_transform(img))

        return views
