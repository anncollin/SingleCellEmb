import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import torch

from Dataset.data_loader import PopulationDataset


#######################################################################################################
# MEAN EARTH MOVER DISTANCE BETWEEN TWO EMBEDDING SETS
# ----------------------------------------------------------------------------------------------------
# Computes the mean 1D Wasserstein (Earth Mover) distance between two sets of embeddings.
#
# Each embedding dimension is treated independently and the Wasserstein distance
# is computed between the corresponding distributions of the two populations.
# The final score is the average distance across all embedding dimensions.
#
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    emb_A : np.ndarray
#        Array of shape (N_A, D) containing embeddings for population A.
#
#    emb_B : np.ndarray
#        Array of shape (N_B, D) containing embeddings for population B.
#
#    normalize : bool
#        If True, embeddings are L2-normalized before computing distances.
#
#    Returns:
#    ---------------------------
#    distance : float
#        Mean Wasserstein distance across all embedding dimensions.
#######################################################################################################

def mean_emd_numpy(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
    normalize: bool = False,
) -> float:

    from scipy.stats import wasserstein_distance
    import torch.nn.functional as F

    if emb_A.size == 0 or emb_B.size == 0:
        return float("nan")

    if normalize:
        emb_A = F.normalize(torch.from_numpy(emb_A), p=2, dim=1).numpy()
        emb_B = F.normalize(torch.from_numpy(emb_B), p=2, dim=1).numpy()

    d = emb_A.shape[1]
    score = 0.0

    for i in range(d):
        score += wasserstein_distance(emb_A[:, i], emb_B[:, i])

    return score / float(d)

@torch.no_grad()
def compute_expert_annotation_metric(
    student,
    data_root: str,
    annotations_csv: str,
    in_channels: str,
    device="cuda",
    metric: str = "prototype",
):

    df = pd.read_csv(annotations_csv)

    dataset = PopulationDataset(
        root_dir=data_root,
        wells_csv="./Dataset/unique_drugs.csv",
        in_channels=in_channels,
    )

    well_index = {}
    for i, (path, drug) in enumerate(dataset.samples):
        plate = os.path.basename(os.path.dirname(path))
        well  = os.path.basename(path).replace(".npy", ".zip")
        key   = f"{plate}/{well}"
        well_index[key] = i

    cache = {}

    def get_population_embeddings(pop_path):

        if pop_path not in cache:

            idx = well_index.get(pop_path, None)
            if idx is None:
                return None

            tensor, _ = dataset[idx]
            tensor = tensor.to(device)

            feats = []
            batch_size = 512

            for i in range(0, tensor.shape[0], batch_size):
                batch = tensor[i:i+batch_size]
                z = student.backbone(batch)
                feats.append(z.cpu())

            feats = torch.cat(feats, dim=0)

            if metric == "prototype":
                cache[pop_path] = feats.mean(dim=0)
            elif metric == "emd":
                cache[pop_path] = feats.numpy()
            else:
                raise ValueError("metric must be 'prototype' or 'emd'")

        return cache[pop_path]

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

        embA = get_population_embeddings(A)
        embB = get_population_embeddings(B)
        embC = get_population_embeddings(C)

        if embA is None or embB is None or embC is None:
            continue

        if metric == "prototype":

            dAB = torch.norm(embA - embB).item()
            dAC = torch.norm(embA - embC).item()
            dBC = torch.norm(embB - embC).item()

        else:

            dAB = mean_emd_numpy(embA, embB)
            dAC = mean_emd_numpy(embA, embC)
            dBC = mean_emd_numpy(embB, embC)

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


#######################################################################################################
# VISUALIZE RANDOM CELLS FROM POPULATION DATASET
#######################################################################################################

@torch.no_grad()
def compute_dapi_ratio(dataset):

    dmso_pixels = 0
    dmso_total  = 0

    scr_pixels = 0
    scr_total  = 0

    for i, (_, drug) in enumerate(dataset.samples):

        tensor, _ = dataset[i]        # (N,C,H,W)
        dapi = tensor[:, 1].numpy()   # DAPI channel

        nonzero = (dapi > 0).sum()
        total   = dapi.size

        if drug == "DMSO":
            dmso_pixels += nonzero
            dmso_total  += total

        elif drug == "SCR":
            scr_pixels += nonzero
            scr_total  += total

    dmso_mean = dmso_pixels / dmso_total
    scr_mean  = scr_pixels  / scr_total

    print("DAPI non-zero area")
    print("DMSO:", dmso_mean)
    print("SCR :", scr_mean)
    print("DMSO/SCR ratio:", dmso_mean / scr_mean)
    

@torch.no_grad()
def visualize_population_samples(dataset, cells_per_row=10, channel_display="both"):

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
            rgb[..., 1] = np.clip(x[0] * 3.0, 0.0, 1.0)
            rgb[..., 2] = np.clip(x[1] * 3.0, 0.0, 1.0)
            return rgb

    dmso_wells = []
    scr_wells  = []

    # FAST: do not load .npy files
    for i, (_, drug) in enumerate(dataset.samples):
        if drug == "DMSO":
            dmso_wells.append(i)
        elif drug == "SCR":
            scr_wells.append(i)

    fig, axes = plt.subplots(4, cells_per_row, figsize=(3 * cells_per_row, 12))

    # ---------------- DMSO ----------------
    dmso_ratios = []

    for row in range(2):
        tensor, _ = dataset[random.choice(dmso_wells)]
        idxs = random.sample(range(tensor.shape[0]), cells_per_row)

        for col, idx in enumerate(idxs):

            x = tensor[idx].numpy()
            dapi = x[1]                     # DAPI channel
            dmso_ratios.append((dapi > 0).mean())

            view = make_view(x, channel_display)
            axes[row, col].imshow(view)
            axes[row, col].axis("off")


    # ---------------- SCR ----------------
    scr_ratios = []

    for row in range(2):
        tensor, _ = dataset[random.choice(scr_wells)]
        idxs = random.sample(range(tensor.shape[0]), cells_per_row)

        for col, idx in enumerate(idxs):

            x = tensor[idx].numpy()
            dapi = x[1]
            scr_ratios.append((dapi > 0).mean())

            view = make_view(x, channel_display)
            axes[row + 2, col].imshow(view)
            axes[row + 2, col].axis("off")


    # ---------------- ratio ----------------
    dmso_mean = np.mean(dmso_ratios)
    scr_mean  = np.mean(scr_ratios)

    print("DAPI non-zero area")
    print("DMSO:", dmso_mean)
    print("SCR :", scr_mean)
    print("DMSO/SCR ratio:", dmso_mean / scr_mean)

    # ---------------- labels ----------------
    axes[0,0].set_title("DMSO", fontsize=12)
    axes[2,0].set_title("SCR", fontsize=12)

    plt.tight_layout()
    plt.show()