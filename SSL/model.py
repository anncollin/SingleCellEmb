from typing import Tuple

import timm
import torch.nn as nn
import torch.nn.functional as F


#######################################################################################################
# DINO PROJECTION HEAD
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    in_dim : int
#        Input feature dimension from the backbone (e.g. 384 for ViT-S).
#
#    out_dim : int
#        Output dimension (number of prototypes / logits).
#
#    hidden_dim : int
#        Hidden dimension inside the MLP.
#
#    bottleneck_dim : int
#        Bottleneck dimension before the final linear layer.
#
#    nlayers : int
#        Number of layers in the MLP (including bottleneck).
#
#    Returns:
#    ---------------------------
#    head : nn.Module
#        Projection head mapping backbone features to DINO logits.
#######################################################################################################
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ):
        super().__init__()
        layers = []
        dim_list = [in_dim] + [hidden_dim] * (nlayers - 1) + [bottleneck_dim]
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


#######################################################################################################
# DINO STUDENT/TEACHER WRAPPER
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    backbone : nn.Module
#        Vision transformer backbone returning a feature vector.
#
#    head : nn.Module
#        DINO projection head.
#
#    Returns:
#    ---------------------------
#    model : nn.Module
#        Model that returns DINO logits given an input crop.
#######################################################################################################
class DINOStudent(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


#######################################################################################################
# CREATE VIT-SMALL BACKBONE
# ----------------------------------------------------------------------------------------------------
# Accepts variable input image sizes (needed for multi-crop: 96x96 and 48x48).
#
#    Parameters:
#    ---------------------------
#    patch_size : int
#        ViT patch size. For DINO-8 use 8.
#
#    in_chans : int
#        Number of input channels (e.g. 2 for DAPI + fibrillarin).
#
#    Returns:
#    ---------------------------
#    backbone : nn.Module
#        ViT backbone model outputting feature vectors without a classifier head.
#######################################################################################################
def create_vit_small_backbone(
    patch_size: int = 8,
    in_chans: int = 2,
) -> nn.Module:
    # img_size=None allows multi-crop of any spatial size (48, 96, etc.)
    model = timm.create_model(
        "vit_small_patch8_224",
        img_size=None,
        dynamic_img_size=True,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=0,
    )
    return model
