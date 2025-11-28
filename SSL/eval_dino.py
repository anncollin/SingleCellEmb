import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.utils import ensure_dir


#######################################################################################################
# TRAINED STUDENT LOADING
#######################################################################################################
def load_trained_student(checkpoint_path: str, cfg: Dict, device: str = "cuda"):

    patch_size = int(cfg.get("patch_size", 8))
    in_chans   = int(cfg.get("in_channels", 2))
    out_dim    = int(cfg.get("out_dim", 8192))

    backbone = create_vit_small_backbone(patch_size=patch_size, in_chans=in_chans)
    head     = DINOHead(in_dim=backbone.num_features, out_dim=out_dim)
    student  = DINOStudent(backbone, head).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(ckpt["student_state_dict"], strict=True)
    student.eval()

    return student


#######################################################################################################
# DRUG LIST
#######################################################################################################
def load_drug_list(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 2].astype(str).str.strip().tolist()


#######################################################################################################
# FIND FOLDER OF NPY FILES FOR A DRUG
#######################################################################################################
def get_npy_folder_from_drug(drug_name: str, cfg: Dict) -> str:

    df = pd.read_csv(cfg["label_path"], header=None, dtype=str)

    matches = df[df.iloc[:, 2].str.strip() == drug_name.strip()]
    if len(matches) == 0:
        return None

    plate = matches.iloc[0, 0].strip()
    well  = matches.iloc[0, 1].strip()

    folder = os.path.join(cfg["data_root"], plate, well)
    return folder if os.path.isdir(folder) else None


#######################################################################################################
# STREAMING EMBEDDING EXTRACTION (USES ALL CELLS, BOUNDED MEMORY)
#######################################################################################################
@torch.no_grad()
def compute_embeddings_for_drug_folder(
    student,
    folder: str,
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Streams all .npy files from disk in mini-batches.
    Uses ALL cells without ever stacking the full dataset in RAM or GPU.
    """

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    )

    if len(files) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    out = []
    N = len(files)

    for i in range(0, N, batch_size):

        batch_files = files[i:i + batch_size]

        batch = np.stack(
            [np.load(os.path.join(folder, f)).astype(np.float32)
             for f in batch_files],
            axis=0
        )  # (B, 2, 96, 96)

        batch = torch.from_numpy(batch).to(device, non_blocking=True)

        z = student.backbone(batch)
        out.append(z.cpu().numpy())

        del batch
        torch.cuda.empty_cache()

    return np.concatenate(out, axis=0)


#######################################################################################################
# 1D MULTI-THREADED MARGINAL WASSERSTEIN DISTANCE
#######################################################################################################
def marginal_wasserstein_multithread(
    A: np.ndarray,
    B: np.ndarray,
    normalize: bool = False,
    n_threads: int = 16,
) -> float:

    if normalize:
        A = F.normalize(torch.from_numpy(A), p=2, dim=1).numpy()
        B = F.normalize(torch.from_numpy(B), p=2, dim=1).numpy()

    d = A.shape[1]

    if d <= 4 or n_threads <= 1:
        return sum(wasserstein_distance(A[:, i], B[:, i]) for i in range(d)) / d

    def w1_dim(i):
        return wasserstein_distance(A[:, i], B[:, i])

    max_workers = min(n_threads, d)
    total = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(w1_dim, i) for i in range(d)]
        for f in as_completed(futures):
            total += f.result()

    return total / float(d)


#######################################################################################################
# PAIRWISE WASSERSTEIN MATRIX (FULL DATASET, STREAMED)
#######################################################################################################
def compute_pairwise_emd_matrix(
    student,
    drugs: List[str],
    cfg: Dict,
    device: str = "cuda",
    batch_size: int = 128,
    n_threads: int = 16,
    normalize: bool = False,
):

    valid_drugs = []
    folder_list = []

    # Resolve drug -> folder
    for drug in drugs:
        folder = get_npy_folder_from_drug(drug, cfg)
        if folder is not None:
            valid_drugs.append(drug)
            folder_list.append(folder)

    D = len(valid_drugs)
    M = np.zeros((D, D), dtype=np.float32)

    for i in tqdm(range(D), desc="Outer loop"):

        # Embed drug i (ALL cells, streamed)
        A = compute_embeddings_for_drug_folder(
            student,
            folder_list[i],
            device=device,
            batch_size=batch_size,
        )

        for j in range(i, D):

            B = compute_embeddings_for_drug_folder(
                student,
                folder_list[j],
                device=device,
                batch_size=batch_size,
            )

            dist = marginal_wasserstein_multithread(
                A, B, normalize=normalize, n_threads=n_threads
            )

            M[i, j] = M[j, i] = dist

            del B
            torch.cuda.empty_cache()

        del A
        torch.cuda.empty_cache()

    return valid_drugs, M


#######################################################################################################
# SAVE CSV
#######################################################################################################
def save_distance_matrix(matrix, labels, output_csv):

    with open(output_csv, "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            f.write(label + "," + ",".join(str(float(x)) for x in row) + "\n")


#######################################################################################################
# MAIN ENTRY POINT
#######################################################################################################
def evaluate_dino_experiment(cfg: Dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = str(cfg.get("results_root", "./Results/DINO_default"))
    ensure_dir(results_root)

    ckpt = os.path.join(results_root, "checkpoints", "final_weights.pth")

    drugs = load_drug_list(cfg["label_path"])

    print(f"[Eval] Loading student from {ckpt}")
    student = load_trained_student(ckpt, cfg, device=device)

    labels, matrix = compute_pairwise_emd_matrix(
        student=student,
        drugs=drugs,
        cfg=cfg,
        device=device,
        batch_size=128,   # GPU memory bounded here
        n_threads=16,
        normalize=False,
    )

    out_csv = os.path.join(results_root, "drug_emd_distance_matrix.csv")
    save_distance_matrix(matrix, labels, out_csv)

    print(f"[Eval] Saved distance matrix to {out_csv}")

    return matrix, labels
