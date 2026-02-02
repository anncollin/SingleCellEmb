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

    def __init__(self, data_root: str, population_zip_path: str, in_channels: str):
        population_path = population_zip_path.replace(".zip", "")
        self.population_dir = os.path.join(data_root, population_path)

        self.in_channels = in_channels.lower()
        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        if self.in_channels not in self.channel_map:
            raise ValueError(f"Invalid in_channels={in_channels}")

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
        chans = self.channel_map[self.in_channels]
        arr = arr[chans]
        return torch.from_numpy(arr).float()


@torch.no_grad()
def compute_population_embedding(
    model,
    data_root: str,
    population_zip_path: str,
    in_channels: str,
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
    in_channels: str,
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

        if q1 == "yes" and q2 == "both" and q3 == "equal":
            case = 1
        elif q1 == "yes" and q2 == "B":
            case = 2
        elif q1 == "yes" and q2 == "C":
            case = 3
        else:
            continue

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

        if case == 1:
            is_valid = (dBC < dAB) and (dBC < dAC)
        elif case == 2:
            is_valid = (dAC < dAB) and (dAC < dBC)
        else:
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
def visualize_multicrop(dataset, gpu_transform, device="cuda", channel_display="both"):
    """
    Displays the original image + 2 global and N local crops
    for 5 random dataset images in a grid:
    5 rows x (1 + 2 + N) columns.
    """
    
    def make_view(x, mode):
        x = np.clip(x, 0.0, 1.0)

        if mode == "egfp":
            rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
            rgb[..., 1] = np.clip(x[0] * 3.0, 0.0, 1.0)
            return rgb

        if mode == "dapi":
            rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
            rgb[..., 2] = np.clip(x[0] * 3.0, 0.0, 1.0)
            return rgb

        if mode == "both":
            rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
            rgb[..., 1] = np.clip(x[0] * 3.0, 0.0, 1.0)  # EGFP
            rgb[..., 2] = np.clip(x[1] * 3.0, 0.0, 1.0)  # DAPI
            return rgb

        raise ValueError("channel_display must be 'egfp', 'dapi', or 'both'")

    n_rows = 5
    sample_idxs = random.sample(range(len(dataset)), n_rows)

    test_img = dataset[sample_idxs[0]].unsqueeze(0).to(device)
    test_crops = gpu_transform(test_img)
    n_cols = 1 + len(test_crops)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for row, sample_idx in enumerate(sample_idxs):

        img = dataset[sample_idx].unsqueeze(0).to(device)
        crops = gpu_transform(img)

        x0 = img[0].detach().cpu().float().numpy()
        view0 = make_view(x0, channel_display)

        axes[row, 0].imshow(view0)
        axes[row, 0].axis("off")

        if row == 0:
            axes[row, 0].set_title("Original")

        for col, crop in enumerate(crops, start=1):

            x = crop[0].detach().cpu().float().numpy()
            view = make_view(x, channel_display)

            axes[row, col].imshow(view)
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