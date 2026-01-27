import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#######################################################################################################
# COMPUTE EXPERT ANNOTATION METRIC
#######################################################################################################

class PopulationDataset(Dataset):
    """
    Dataset loading all .npy cells from a population folder.
    Applies the same channel selection as training.
    """

    def __init__(self, data_root: str, population_zip_path: str, in_channels: int):
        population_path = population_zip_path.replace(".zip", "")
        self.population_dir = os.path.join(data_root, population_path)
        self.in_channels = int(in_channels)

        if not os.path.isdir(self.population_dir):
            self.npy_files = []
        else:
            self.npy_files = sorted(
                [
                    os.path.join(self.population_dir, f)
                    for f in os.listdir(self.population_dir)
                    if f.endswith(".npy")
                ]
            )

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        arr = np.load(self.npy_files[idx])  # (C, H, W)

        if self.in_channels == 1:
            arr = arr[0:1]  # EGFP only
        elif self.in_channels == 2:
            pass
        else:
            raise ValueError(f"Unsupported in_channels={self.in_channels}")

        return torch.from_numpy(arr).float()


@torch.no_grad()
def compute_population_embedding(
    model,
    data_root: str,
    population_zip_path: str,
    in_channels: int,
    device="cuda",
    batch_size=128,
):
    """
    Population embedding = mean of all cell embeddings.
    """

    dataset = PopulationDataset(
        data_root=data_root,
        population_zip_path=population_zip_path,
        in_channels=in_channels,
    )

    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    feats = []

    for x in loader:
        x = x.to(device, non_blocking=True)
        h = model.backbone(x)
        feats.append(h.cpu())

    feats = torch.cat(feats, dim=0)
    return feats.mean(dim=0)


@torch.no_grad()
def compute_expert_annotation_metric(
    student,
    data_root: str,
    annotations_csv: str,
    in_channels: int,
    device="cuda",
):
    """
    Expert-annotation consistency metric (cases 1, 2, 3 only).
    """

    df = pd.read_csv(annotations_csv)

    total = 0
    valid = 0

    for _, row in df.iterrows():

        A = row["A"]
        B = row["B"]
        C = row["C"]

        q1 = str(row["q1_yesno"]).strip().lower()
        q2 = str(row["which"]).strip() if not pd.isna(row["which"]) else None
        q3 = str(row["both"]).strip() if not pd.isna(row["both"]) else None

        # ----------------------------------------------------------
        # Case selection (ONLY 1, 2, 3)
        # ----------------------------------------------------------
        if q1 == "yes" and q2 == "both" and q3 == "equal":
            case = 1
        elif q1 == "yes" and q2 == "B":
            case = 2
        elif q1 == "yes" and q2 == "C":
            case = 3
        else:
            continue

        # ----------------------------------------------------------
        # Population embeddings
        # ----------------------------------------------------------
        embA = compute_population_embedding(
            student, data_root, A, in_channels, device
        )
        embB = compute_population_embedding(
            student, data_root, B, in_channels, device
        )
        embC = compute_population_embedding(
            student, data_root, C, in_channels, device
        )

        if embA is None or embB is None or embC is None:
            continue

        dAB = torch.norm(embA - embB).item()
        dAC = torch.norm(embA - embC).item()
        dBC = torch.norm(embB - embC).item()

        # ----------------------------------------------------------
        # Validity rules
        # ----------------------------------------------------------
        if case == 1:
            is_valid = (dBC < dAB) and (dBC < dAC)
        elif case == 2:
            is_valid = (dAC < dAB) and (dAC < dBC)
        else:  # case == 3
            is_valid = (dAB < dAC) and (dAB < dBC)

        total += 1
        if is_valid:
            valid += 1

    if total == 0:
        return 0.0

    return 100.0 * valid / total

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
    Displays the original image + 2 global and N local crops
    for 5 random dataset images in a grid:
    5 rows x (1 + 2 + N) columns.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import random

    n_rows = 5
    sample_idxs = random.sample(range(len(dataset)), n_rows)

    # Determine number of crops dynamically
    test_img = dataset[sample_idxs[0]].unsqueeze(0).to(device)
    test_crops = gpu_transform(test_img)
    n_cols = 1 + len(test_crops)  # +1 for original image

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for row, sample_idx in enumerate(sample_idxs):

        img = dataset[sample_idx].unsqueeze(0).to(device)
        crops = gpu_transform(img)

        # ---- original image (column 0) ----
        x0 = img[0].detach().cpu().float().numpy()  # (C, H, W)

        if channel_display == "rgb":
            if x0.shape[0] == 2:
                rgb = np.zeros((x0.shape[1], x0.shape[2], 3), dtype=np.float32)
                rgb[..., 1] = np.clip(x0[0] * 3.0, 0.0, 1.0)
                rgb[..., 2] = np.clip(x0[1] * 3.0, 0.0, 1.0)
                view0 = rgb
            else:
                view0 = x0.transpose(1, 2, 0)
        elif channel_display == "egfp":
            view0 = x0[0]
        elif channel_display == "dapi":
            view0 = x0[1]
        else:
            raise ValueError("channel_display must be 'rgb', 'egfp', or 'dapi'")

        axes[row, 0].imshow(
            view0,
            cmap="gray" if channel_display != "rgb" else None
        )
        axes[row, 0].axis("off")

        if row == 0:
            axes[row, 0].set_title("Original")

        # ---- crops (columns 1..end) ----
        for col, crop in enumerate(crops, start=1):

            x = crop[0].detach().cpu().float().numpy()  # (C, H, W)

            if channel_display == "rgb":
                if x.shape[0] == 2:
                    rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
                    rgb[..., 1] = np.clip(x[0] * 3.0, 0.0, 1.0)
                    rgb[..., 2] = np.clip(x[1] * 3.0, 0.0, 1.0)
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
                if col == 1:
                    axes[row, col].set_title("Global 1")
                elif col == 2:
                    axes[row, col].set_title("Global 2")
                else:
                    axes[row, col].set_title(f"Local {col-2}")

    plt.tight_layout()
    plt.show()


