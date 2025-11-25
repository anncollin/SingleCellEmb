import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import pandas as pd 

from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.utils import ensure_dir

#########################################################################################
# IMPORT load_from_zip (the ONLY allowed reader)
#########################################################################################
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATASET_UTILS_DIR = os.path.join(PROJECT_ROOT, "dataset")
sys.path.append(DATASET_UTILS_DIR)

from createDB_utils import load_from_zip


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
# ZIP PATH RESOLUTION
#######################################################################################################
def get_zip_path_from_drug(drug_name):
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "labels", "unique_drugs.csv")
    )

    df = pd.read_csv(csv_path, header=None)
    matches = df[df.iloc[:, 2].astype(str).str.strip().str.lower() ==
                 drug_name.strip().lower()]

    if matches.empty:
        print(f"[Eval] WARNING: drug {drug_name} not found")
        return None

    base_dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "MyDB")
    )

    plate = str(matches.iloc[0][0]).strip()
    well  = str(matches.iloc[0][1]).strip().zfill(6)

    zp = os.path.join(base_dataset_path, plate, f"{well}.zip")
    return zp if os.path.exists(zp) else None


#######################################################################################################
# LOAD ALL CELLS FOR ONE DRUG
#######################################################################################################
def load_all_cells_for_drug(zip_path: str, normalize_16bit: bool = True):
    imgs, _, _ = load_from_zip(
        zip_filename=zip_path,
        batch_size=1,
        start=0,
        read_all=True,
        channel="both",
    )
    if imgs.shape[0] == 0:
        return torch.empty(0, 2, 96, 96)

    if normalize_16bit and imgs.max() > 1.0:
        imgs = imgs / 65535.0

    return torch.from_numpy(imgs.astype(np.float32))


#######################################################################################################
# EMBEDDING EXTRACTION (BATCHED)
#######################################################################################################
@torch.no_grad()
def compute_embeddings_for_images(student, imgs: torch.Tensor, device="cuda", batch_size=128):
    if imgs.numel() == 0:
        return np.zeros((0, 1), dtype=np.float32)

    imgs = imgs.to(device)
    out  = []
    N    = imgs.shape[0]

    for i in range(0, N, batch_size):
        z = student.backbone(imgs[i:i+batch_size])
        out.append(z.cpu().numpy())

    return np.concatenate(out, axis=0)


#######################################################################################################
# 1D MULTI-THREADED MARGINAL WASSERSTEIN DISTANCE
#######################################################################################################
def marginal_wasserstein_multithread(A: np.ndarray,
                                     B: np.ndarray,
                                     normalize: bool = False,
                                     n_threads: int = 16) -> float:
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
# PAIRWISE COMPUTATION — MEMORY SAFE
#######################################################################################################
def compute_pairwise_emd_matrix(student,
                                drugs: List[str],
                                device="cuda",
                                batch_size=128,
                                n_threads=16,
                                normalize=False):

    valid_drugs = []
    zp_list     = []

    # resolve all drugs
    for drug in drugs:
        zp = get_zip_path_from_drug(drug)
        if zp is not None:
            valid_drugs.append(drug)
            zp_list.append(zp)

    D = len(valid_drugs)
    M = np.zeros((D, D), dtype=np.float32)

    # pairwise computation
    for i in tqdm(range(D), desc="Outer loop"):
        imgs_i = load_all_cells_for_drug(zp_list[i])
        A = compute_embeddings_for_images(student, imgs_i, device=device, batch_size=batch_size)

        for j in range(i, D):
            imgs_j = load_all_cells_for_drug(zp_list[j])
            B = compute_embeddings_for_images(student, imgs_j, device=device, batch_size=batch_size)

            dist = marginal_wasserstein_multithread(A, B,
                                                    normalize=normalize,
                                                    n_threads=n_threads)
            M[i, j] = M[j, i] = dist

            del imgs_j, B

        del imgs_i, A
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
# MAIN ENTRY POINT TO BE CALLED BY YOUR SCRIPT
#######################################################################################################
def evaluate_dino_experiment(cfg: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = str(cfg.get("results_root", "./Results/DINO_default"))
    ensure_dir(results_root)

    ckpt = os.path.join(results_root, "checkpoints", "final_weights.pth")

    # load drug list
    drugs_csv = os.path.join(PROJECT_ROOT, "dataset", "labels", "unique_drugs.csv")
    drugs = load_drug_list(drugs_csv)

    print(f"[Eval] Loading student from {ckpt}")
    student = load_trained_student(ckpt, cfg, device=device)

    labels, matrix = compute_pairwise_emd_matrix(
        student=student,
        drugs=drugs,
        device=device,
        batch_size=128,
        n_threads=16,
        normalize=False
    )

    out_csv = os.path.join(results_root, "drug_emd_distance_matrix.csv")
    save_distance_matrix(matrix, labels, out_csv)
    print(f"[Eval] Saved distance matrix to {out_csv}")

    return matrix, labels
