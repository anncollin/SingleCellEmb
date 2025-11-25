from typing import Dict, Tuple, List, Optional
import random
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import scipy.ndimage as ndi


# -------------------------------------------------------------------------
# SOLARIZATION (manual version, compatible with Torchvision 0.12)
# -------------------------------------------------------------------------
class Solarization(object):
    def __init__(self, p: float, threshold: float = 0.5):
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.p:
            # img is a Tensor scaled to [0,1]
            return torch.where(img < self.threshold, img, 1 - img)
        return img


# -------------------------------------------------------------------------
# ELASTIC DEFORMATION (classic implementation used in DINO papers)
# -------------------------------------------------------------------------
class ElasticDeformation(object):
    def __init__(self, p: float,
                 alpha_range: Tuple[float, float],
                 sigma_range: Tuple[float, float]):

        self.p = p
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        # Convert torch tensor → numpy
        img_np = img.numpy().transpose(1, 2, 0)

        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)

        dx = ndi.gaussian_filter(
            (np.random.rand(*img_np.shape[:2]) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        dy = ndi.gaussian_filter(
            (np.random.rand(*img_np.shape[:2]) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha

        x, y = np.meshgrid(np.arange(img_np.shape[1]), np.arange(img_np.shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        deformed = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            deformed[..., c] = ndi.map_coordinates(
                img_np[..., c], indices, order=1, mode="reflect"
            ).reshape(img_np.shape[:2])

        # Convert back to tensor
        return torch.tensor(deformed.transpose(2, 0, 1), dtype=img.dtype)


# -------------------------------------------------------------------------
# FLOAT EQUALIZE (uses classic F.equalize)
# -------------------------------------------------------------------------
class FloatEqualize:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        # img is C×H×W float32
        c, h, w = img.shape
        out = torch.empty_like(img)

        for i in range(c):
            # convert to uint8
            ch = (img[i] * 255).byte()  # (H, W)

            # expand to (1, H, W) because torchvision equalize requires 3D
            ch_3d = ch.unsqueeze(0)

            # equalize
            ch_eq = F.equalize(ch_3d)

            # back to float
            out[i] = ch_eq.squeeze(0).float() / 255.0

        return out



# -------------------------------------------------------------------------
# FLOAT CONTRAST (simple manual implementation)
# -------------------------------------------------------------------------
class FloatContrast:
    def __init__(self, p: float, contrast_range: Tuple[float, float]):
        self.p = p
        self.low, self.high = contrast_range

    def __call__(self, img):
        if random.random() < self.p:
            c = random.uniform(self.low, self.high)
            return (img * c).clamp(0, 1)
        return img


# -------------------------------------------------------------------------
# MULTI-CROP TRANSFORM (DINO)
# -------------------------------------------------------------------------
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

        # -----------------------
        # 1. YAML parameters
        # -----------------------
        aug = cfg.get("augment", {}) if cfg is not None else {}

        elastic_p = float(aug.get("elastic_p", 0.0))
        elastic_alpha_range = tuple(aug.get("elastic_alpha_range", [30.0, 50.0]))
        elastic_sigma_range = tuple(aug.get("elastic_sigma_range", [4.0, 6.0]))

        solarization_p = float(aug.get("solarization_p", 0.0))
        equalize_p = float(aug.get("equalize_p", 0.0))
        contrast_p = float(aug.get("contrast_p", 0.0))
        contrast_range = tuple(aug.get("contrast_range", [0.8, 1.2]))

        rotation_degrees = aug.get("rotation_degrees", None)
        if rotation_degrees is not None:
            rotation_degrees = tuple(rotation_degrees)

        # -----------------------
        # 2. Extra transforms
        # -----------------------
        extra = []

        if elastic_p > 0:
            extra.append(ElasticDeformation(elastic_p, elastic_alpha_range, elastic_sigma_range))

        if equalize_p > 0:
            extra.append(FloatEqualize(equalize_p))

        if contrast_p > 0:
            extra.append(FloatContrast(contrast_p, contrast_range))

        if solarization_p > 0:
            extra.append(Solarization(solarization_p))

        if rotation_degrees is None:
            rotation = T.Identity()
        else:
            rotation = T.RandomRotation(
                degrees=rotation_degrees,
                interpolation=T.InterpolationMode.BILINEAR,
            )

        # -----------------------
        # 3. GLOBAL PIPELINE
        # -----------------------
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

        # -----------------------
        # 4. LOCAL PIPELINE
        # -----------------------
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

    def __call__(self, img):
        views = []

        for _ in range(2):
            views.append(self.global_transform(img))

        for _ in range(self.local_crops_number):
            views.append(self.local_transform(img))

        return views
