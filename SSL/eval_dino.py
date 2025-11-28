import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import pandas as pd
import pickle
from scipy.stats import wasserstein_distance

from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.utils import ensure_dir


#######################################################################################################
# LOAD TRAINED STUDENT
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
# MAP DRUG -> FOLDER
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
# STREAM ALL CELLS -> EMBEDDINGS (FULL DATASET, GPU SAFE)
#######################################################################################################
@torch.no_grad()
def compute_embeddings_for_drug_folder(
    student,
    folder: str,
    device: str = "cuda",
    batch_size: int = 128,
) -> torch.Tensor:

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))

    if len(files) == 0:
        return torch.empty(0, 1)

    out = []

    for i in range(0, len(files), batch_size):

        batch_files = files[i:i + batch_size]

        batch = np.stack(
            [np.load(os.path.join(folder, f)).astype(np.float32)
             for f in batch_files],
            axis=0
        )

        batch = torch.from_numpy(batch).to(device, non_blocking=True)
        z = student.backbone(batch)
        out.append(z.cpu())

        del batch
        torch.cuda.empty_cache()

    return torch.cat(out, dim=0)


#######################################################################################################
# MEAN MARGINAL EMD (SCIPY VERSION)
#######################################################################################################
def mean_emd(emb_A: torch.Tensor,
             emb_B: torch.Tensor,
             normalize: bool = False) -> float:

    d = emb_A.shape[1]

    if normalize:
        emb_A = F.normalize(emb_A, p=2, dim=1).numpy()
        emb_B = F.normalize(emb_B, p=2, dim=1).numpy()
    else:
        emb_A = emb_A.numpy()
        emb_B = emb_B.numpy()

    emd_score = 0.0
    for i in range(d):
        emd_score += wasserstein_distance(emb_A[:, i], emb_B[:, i])

    return emd_score / float(d)


#######################################################################################################
# SINGLE PAIR DISTANCE
#######################################################################################################
def compute_pair_distance(args_tuple):

    cls1, cls2, embeddings, q_cls, normalize = args_tuple

    q_cls_np = np.asarray(q_cls)

    mask1 = (q_cls_np == cls1)
    mask2 = (q_cls_np == cls2)

    emb_A = embeddings[mask1]
    emb_B = embeddings[mask2]

    distance = mean_emd(emb_A, emb_B, normalize=normalize)
    return (cls1, cls2), distance


#######################################################################################################
# EVALUATION ENTRY POINT (CALLED FROM MAIN)
#######################################################################################################
def evaluate_dino_experiment(cfg: Dict):

    multiprocessing.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = str(cfg.get("results_root", "./Results/DINO_default"))
    ensure_dir(results_root)
    ckpt = os.path.join(results_root, "checkpoints", "final_weights.pth")

    student = load_trained_student(ckpt, cfg, device=device)

    drugs = load_drug_list(cfg["label_path"])

    valid_drugs = []
    folder_list = []

    for drug in drugs:
        folder = get_npy_folder_from_drug(drug, cfg)
        if folder is not None:
            valid_drugs.append(drug)
            folder_list.append(folder)

    D = len(valid_drugs)

    all_embeddings = []
    q_cls = []

    for i, folder in enumerate(tqdm(folder_list, desc="Embedding drugs")):
        Z = compute_embeddings_for_drug_folder(
            student,
            folder,
            device=device,
            batch_size=128,
        )
        all_embeddings.append(Z)
        q_cls.extend([i] * Z.shape[0])

    embeddings = torch.cat(all_embeddings, dim=0)
    q_cls = np.array(q_cls)

    unique_classes = list(range(D))
    all_class_pairs = list(combinations(unique_classes, 2))

    assigned_pairs = all_class_pairs

    worker_args = [
        (cls1, cls2, embeddings, q_cls, False)
        for cls1, cls2 in assigned_pairs
    ]

    dist_dict = {(cls, cls): 0.0 for cls in unique_classes}

    with Pool(processes=10) as pool:
        results = list(
            tqdm(
                pool.imap(compute_pair_distance, worker_args),
                total=len(worker_args),
                desc="Computing EMD"
            )
        )

    for (cls1, cls2), distance in results:
        dist_dict[(cls1, cls2)] = distance
        dist_dict[(cls2, cls1)] = distance

    out_file = os.path.join(results_root, "emd_distances_not_normalized_part0.pkl")

    with open(out_file, "wb") as f:
        pickle.dump(dist_dict, f)

    return dist_dict, valid_drugs
