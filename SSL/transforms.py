from typing import Dict, Tuple
import torch
import torch.nn as nn
import kornia.augmentation as K


class RandomChannelBrightness(nn.Module):
    """
    Apply random multiplicative brightness to a specific channel.
    """
    def __init__(self, channel_idx: int, p: float, factor_range):
        super().__init__()
        self.channel_idx = channel_idx
        self.p    = float(p)
        self.low  = float(factor_range[0])
        self.high = float(factor_range[1])

    def forward(self, x):
        # x: (B, C, H, W)
        if self.p <= 0.0:
            return x

        if torch.rand(1, device=x.device) > self.p:
            return x

        factor = torch.empty(1, device=x.device).uniform_(self.low, self.high)

        x = x.clone()
        x[:, self.channel_idx] = (x[:, self.channel_idx] * factor).clamp(0.0, 1.0)
        return x



#######################################################################################################
# GPU MULTI-CROP TRANSFORM (KORNIA VERSION - CLEAN DINO STYLE)
#######################################################################################################
class KorniaMultiCropTransform(nn.Module):

    def __init__(
        self,
        image_size: int,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        cfg: Dict = None,
    ):
        super().__init__()

        self.local_crops_number = local_crops_number
        local_size = image_size // 2

        # -----------------------
        # 1. YAML PARAMETERS
        # -----------------------
        aug = cfg.get("augment", {}) if cfg is not None else {}

        elastic_p           = float(aug.get("elastic_p", 0.0))
        elastic_alpha_range = tuple(aug.get("elastic_alpha_range", [30.0, 50.0]))
        elastic_sigma_range = tuple(aug.get("elastic_sigma_range", [4.0, 6.0]))

        solarization_p = float(aug.get("solarization_p", 0.0))
        contrast_p     = float(aug.get("contrast_p", 0.0))
        contrast_range = tuple(aug.get("contrast_range", [0.8, 1.2]))

        brightness_dapi_p     = float(aug.get("brightness_dapi_p", 0.0))
        brightness_dapi_range = tuple(aug.get("brightness_dapi_range", [1.0, 1.0]))

        brightness_egfp_p     = float(aug.get("brightness_egfp_p", 0.0))
        brightness_egfp_range = tuple(aug.get("brightness_egfp_range", [1.0, 1.0]))


        rotation_degrees = aug.get("rotation_degrees", None)
        if rotation_degrees is not None:
            rotation_degrees = tuple(rotation_degrees)

        # -----------------------
        # 2. STRONG (GLOBAL-ONLY) EXTRA AUGS
        # -----------------------
        strong_extra = []

        if elastic_p > 0:
            strong_extra.append(
                K.RandomElasticTransform(
                    p=elastic_p,
                    alpha=elastic_alpha_range,
                    sigma=elastic_sigma_range
                )
            )

        if brightness_dapi_p > 0:
            strong_extra.append(
                RandomChannelBrightness(
                    channel_idx=1,   # DAPI
                    p=brightness_dapi_p,
                    factor_range=brightness_dapi_range,
                )
            )

        if brightness_egfp_p > 0:
            strong_extra.append(
                RandomChannelBrightness(
                    channel_idx=0,   # eGFP
                    p=brightness_egfp_p,
                    factor_range=brightness_egfp_range,
                )
            )

        if contrast_p > 0:
            strong_extra.append(
                K.RandomContrast(
                    contrast=contrast_range,
                    p=contrast_p,
                )
            )

        if solarization_p > 0:
            strong_extra.append(K.RandomSolarize(p=solarization_p))

        # -----------------------
        # 3. GEOMETRIC ROTATION (SHARED)
        # -----------------------
        if rotation_degrees is None:
            rotation = nn.Identity()
        else:
            rotation = K.RandomRotation(
                degrees=rotation_degrees,
                p=1.0,
                resample="bilinear",
            )

        # -----------------------
        # 4. GLOBAL PIPELINE (STRONG)
        # -----------------------
        self.global_transform = nn.Sequential(
            K.RandomResizedCrop(
                size=(image_size, image_size),
                scale=global_crops_scale,
                ratio=(1.0, 1.0),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            rotation,

            # STRONG BLUR
            K.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.5, 2.0),
                p=1.0,
            ),

            *strong_extra,
        )

        # -----------------------
        # 5. LOCAL PIPELINE (WEAK)
        # -----------------------
        self.local_transform = nn.Sequential(
            K.RandomResizedCrop(
                size=(local_size, local_size),
                scale=local_crops_scale,
                ratio=(1.0, 1.0),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            rotation,

            # WEAK BLUR
            K.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 0.5),
                p=1.0,
            ),
        )

    # -----------------------
    # 6. FORWARD (BATCHED GPU)
    # -----------------------
    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (B, C, H, W), already on GPU
        Returns:
            List of views: [global_1, global_2, local_1, ..., local_N]
        """
        views = []

        # Two global crops
        views.append(self.global_transform(x))
        views.append(self.global_transform(x))

        # Local crops
        for _ in range(self.local_crops_number):
            views.append(self.local_transform(x))

        return views
