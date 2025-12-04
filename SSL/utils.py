import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


#######################################################################################################
# EMA UPDATE FOR TEACHER PARAMETERS
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    student : nn.Module
#        Student model.
#
#    teacher : nn.Module
#        Teacher model to be updated in-place.
#
#    momentum : float
#        Exponential moving average momentum coefficient.
#
#    Returns:
#    ---------------------------
#    None
#######################################################################################################
@torch.no_grad()
def update_teacher(student, teacher, momentum: float) -> None:
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data * (1.0 - momentum))


#######################################################################################################
# ENSURE DIRECTORY EXISTS
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    path : str
#        Directory path to create if it does not exist.
#
#    Returns:
#    ---------------------------
#    path : str
#        The same path, for convenience.
#######################################################################################################
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


#######################################################################################################
# SAVE CHECKPOINT
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    checkpoint_path : str
#        Path to the checkpoint file to write.
#
#    state : Dict
#        Dictionary containing model and optimizer state.
#
#    Returns:
#    ---------------------------
#    None
#######################################################################################################
def save_checkpoint(checkpoint_path: str, state: Dict) -> None:
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save(state, checkpoint_path)


#######################################################################################################
# VISUALIZE 5 RANDOM IMAGES: 2 GLOBAL + N LOCAL CROPS EACH (GRID VIEW)
#######################################################################################################
@torch.no_grad()
def visualize_multicrop(
    dataset,
    gpu_transform,
    device="cuda",
    channel_display="rgb",
):
    """
    Displays 2 global and N local crops for 5 random dataset images
    in a grid: 5 rows x (2 + N) columns.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import random

    n_rows = 5
    sample_idxs = random.sample(range(len(dataset)), n_rows)

    # Get number of crops dynamically
    test_img = dataset[sample_idxs[0]].unsqueeze(0).to(device)
    test_crops = gpu_transform(test_img)
    n_cols = len(test_crops)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for row, sample_idx in enumerate(sample_idxs):

        img = dataset[sample_idx].unsqueeze(0).to(device)
        crops = gpu_transform(img)

        for col, crop in enumerate(crops):

            x = crop[0].detach().cpu().float().numpy()  # (C, H, W)

            if channel_display == "rgb":
                if x.shape[0] == 2:
                    rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
                    rgb[..., 1] = np.clip(x[0] * 3.0, 0.0, 1.0)  # eGFP -> G
                    rgb[..., 2] = np.clip(x[1] * 3.0, 0.0, 1.0)  # DAPI -> B
                    view = rgb
                else:
                    view = x.transpose(1, 2, 0)

            elif channel_display == "egfp":
                view = x[0]

            elif channel_display == "dapi":
                view = x[1]

            else:
                raise ValueError("channel_display must be 'rgb', 'egfp', or 'dapi'")

            axes[row, col].imshow(
                view,
                cmap="gray" if channel_display != "rgb" else None
            )
            axes[row, col].axis("off")

            if row == 0:
                if col < 2:
                    axes[row, col].set_title(f"Global {col+1}")
                else:
                    axes[row, col].set_title(f"Local {col-1}")

    plt.tight_layout()
    plt.show()

