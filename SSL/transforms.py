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
    
class CenteredRandomResizedCrop(nn.Module):
    def __init__(self, out_size, scale):
        super().__init__()
        self.out = out_size
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape

        # pick a random crop ratio (0.2 → 0.9)
        r = float(torch.empty(1, device=x.device).uniform_(*self.scale))

        # crop size = r * original image size
        crop_h = int(H * r)
        crop_w = int(W * r)

        # ensure crop is valid
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)

        # center coordinates
        top  = (H - crop_h) // 2
        left = (W - crop_w) // 2

        # centered crop manually
        x = x[:, :, top:top+crop_h, left:left+crop_w]

        # resize to DINO output size
        x = K.Resize((self.out, self.out), align_corners=False)(x)
        return x


#######################################################################################################
# GPU MULTI-CROP TRANSFORM (KORNIA VERSION - FULL YAML-DRIVEN)
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

        sharpness_p      = float(aug.get("sharpness_p", 0.0))
        sharpness_factor = float(aug.get("sharpness_factor", 5.0))

        brightness_dapi_p     = float(aug.get("brightness_dapi_p", 0.0))
        brightness_dapi_range = tuple(aug.get("brightness_dapi_range", [1.0, 1.0]))

        brightness_egfp_p     = float(aug.get("brightness_egfp_p", 0.0))
        brightness_egfp_range = tuple(aug.get("brightness_egfp_range", [1.0, 1.0]))

        rotation_degrees = aug.get("rotation_degrees", None)
        if rotation_degrees is not None:
            rotation_degrees = tuple(rotation_degrees)

        # ---- BLUR (GLOBAL / LOCAL)
        blur_global_p     = float(aug.get("blur_global_p", 0.0))
        blur_global_sigma = tuple(aug.get("blur_global_sigma", [0.5, 2.0]))

        blur_local_p     = float(aug.get("blur_local_p", 0.0))
        blur_local_sigma = tuple(aug.get("blur_local_sigma", [0.1, 0.5]))

        # ---- GAUSSIAN NOISE (GLOBAL / LOCAL)
        noise_global_p   = float(aug.get("noise_global_p", 0.0))
        noise_global_std = tuple(aug.get("noise_global_std", [0.0, 0.0]))

        noise_local_p   = float(aug.get("noise_local_p", 0.0))
        noise_local_std = tuple(aug.get("noise_local_std", [0.0, 0.0]))

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

        if sharpness_p > 0: 
            strong_extra.append(
                K.RandomSharpness(
                    sharpness=sharpness_factor,
                    p=sharpness_p,
                )
            )

        if solarization_p > 0:
            strong_extra.append(K.RandomSolarize(p=solarization_p))

        # -----------------------
        # 3. SHARED GEOMETRY
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
        # 4. GLOBAL BLUR + NOISE
        # -----------------------
        if blur_global_p > 0:
            global_blur = K.RandomGaussianBlur(
                kernel_size=(9, 9),
                sigma=blur_global_sigma,
                p=blur_global_p,
            )
        else:
            global_blur = nn.Identity()

        if noise_global_p > 0:
            global_noise = K.RandomGaussianNoise(
                mean=0.0,
                std=float(noise_global_std[1]),  
                p=noise_global_p,
            )
        else:
            global_noise = nn.Identity()


        # -----------------------
        # 5. LOCAL BLUR + NOISE
        # -----------------------
        if blur_local_p > 0:
            local_blur = K.RandomGaussianBlur(
                kernel_size=(5, 5),
                sigma=blur_local_sigma,
                p=blur_local_p,
            )
        else:
            local_blur = nn.Identity()

        if noise_local_p > 0:
            local_noise = K.RandomGaussianNoise(
                mean=0.0,
                std=float(noise_local_std[1]),  
                p=noise_local_p,
            )
        else:
            local_noise = nn.Identity()


        # -----------------------
        # 6. GLOBAL PIPELINE
        # -----------------------
        self.global_transform = nn.Sequential(
            rotation,
            CenteredRandomResizedCrop(image_size, global_crops_scale),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        
            global_blur,
            global_noise,

            *strong_extra,
        )

        # -----------------------
        # 7. LOCAL PIPELINE
        # -----------------------
        self.local_transform = nn.Sequential(
            rotation,
            K.RandomResizedCrop(
                size=(local_size, local_size),
                scale=local_crops_scale,
                ratio=(1.0, 1.0),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            
            local_blur,
            local_noise,
        )

    # -----------------------
    # 8. FORWARD (BATCHED GPU)
    # -----------------------
    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (B, C, H, W), already on GPU
        Returns:
            List of views: [global_1, global_2, local_1, ..., local_N]
        """
        views = []

        views.append(self.global_transform(x))
        views.append(self.global_transform(x))

        for _ in range(self.local_crops_number):
            views.append(self.local_transform(x))

        return views
