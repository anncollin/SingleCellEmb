import os
from typing import Dict
import torch

from SSL.utils import compute_expert_annotation_metric
from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent


#######################################################################################################
# LOAD TRAINED STUDENT
#######################################################################################################
def load_trained_student(checkpoint_path: str, cfg: Dict, device: str = "cuda"):

    patch_size   = int(cfg.get("patch_size", 8))
    out_dim      = int(cfg.get("out_dim", 8192))
    architecture = str(cfg.get("architecture", "tiny"))

    in_channels_cfg = cfg.get("in_channels", "both")
    channel_map = {
        "egfp": 1,
        "dapi": 1,
        "both": 2,
    }
    in_chans = channel_map[in_channels_cfg]

    backbone = create_vit_small_backbone(
        architecture=architecture,
        patch_size=patch_size,
        in_chans=in_chans,
    )

    head = DINOHead(
        in_dim=backbone.num_features,
        out_dim=out_dim,
    )

    student = DINOStudent(backbone, head).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(ckpt["student_state_dict"], strict=True)
    student.eval()

    return student


#######################################################################################################
# EXPERT EVALUATION ENTRY POINT
#######################################################################################################
@torch.no_grad()
def evaluate_expert(cfg: Dict, annotations_csv: str, in_channels: str):

    assert in_channels in {"both", "egfp", "dapi"}, \
        f"Invalid in_channels={in_channels}"

    device          = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir        = os.getcwd()
    experiment_name = cfg["experiment_name"]
    ckpt_path       = f"{base_dir}/Results/{experiment_name}/checkpoints/final_weights.pth"

    student = load_trained_student(ckpt_path, cfg, device=device)

    score = compute_expert_annotation_metric(
        student=student,
        data_root=cfg["data_root"],
        annotations_csv=annotations_csv,
        in_channels=in_channels,
        device=device,
    )

    print(f"Expert annotation satisfaction: {score:.2f}%")
    return score
