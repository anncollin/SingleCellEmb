import os
import time
import pickle
import multiprocessing
from typing import Dict, List
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.utils import ensure_dir


# =====================================================================================================
# GLOBAL SHARED MEMORY FOR MULTIPROCESSING
# =====================================================================================================
GLOBAL_EMB = None
GLOBAL_QCLS = None


def init_worker(embeddings, q_cls):
    global GLOBAL_EMB, GLOBAL_QCLS
    GLOBAL_EMB = embeddings
    GLOBAL_QCLS = q_cls


# =====================================================================================================
# LOAD TRAINED STUDENT
# =====================================================================================================
def load_trained_student(checkpoint_path: str, cfg: Dict, device: str = "cuda"):

    patch_size = int(cfg.get("patch_size", 8))
    in_chans = int(cfg.get("in_channels", 2))
    out_dim = int(cfg.get("out_dim", 8192))

    backbone = create_vit_small_backbone(patch_size=patch_size, in_chans=in_chans)
    head = DINOHead(in_dim=backbone.num_features, out_dim=out_dim)
    student = DINOStudent(backbone, head).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(ckpt["student_state_dict"], strict=True)
    student.eval()

    return student


# =====================================================================================================
# LOAD DRUG LIST
# =====================================================================================================
def load_drug_list(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 2].astype(str).str.strip().tolist()


# =====================================================================================================
# MAP DRUG -> FOLDER
# =====================================================================================================
def get_npy_folder_from_drug(drug_name: str, cfg: Dict) -> str:

    df = pd.read_csv(cfg["label_path"], header=None, dtype=str)

    matches = df[df.iloc[:, 2].str.strip() == drug_name.strip()]
    if len(matches) == 0:
        return None

    plate = matches.iloc[0, 0].strip()
    well = matches.iloc[0, 1].strip()

    folder = os.path.join(cfg["data_root"], plate, well)
    return folder if os.path.isdir(folder) else None


# =====================================================================================================
# EMBEDDING COMPUTATION
# =====================================================================================================
@torch.no_grad()
def compute_embeddings_for_drug_folder(student, folder, device="cuda", batch_size=128):

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))

    if len(files) == 0:
        return torch.empty(0, 1)

    out = []

    for i in range(0, len(files), batch_size):

        batch_files = files[i:i + batch_size]

        batch = np.stack(
            [np.load(os.path.join(folder, f)).astype(np.float32) for f in batch_files],
            axis=0
        )

        batch = torch.from_numpy(batch).to(device, non_blocking=True)
        z = student.backbone(batch)
        out.append(z.cpu())

        del batch
        torch.cuda.empty_cache()

    return torch.cat(out, dim=0)


# =====================================================================================================
# MEAN MARGINAL EMD
# =====================================================================================================
def mean_emd(emb_A, emb_B, normalize=False):

    d = emb_A.shape[1]

    if normalize:
        emb_A = F.normalize(emb_A, p=2, dim=1).numpy()
        emb_B = F.normalize(emb_B, p=2, dim=1).numpy()
    else:
        emb_A = emb_A.numpy()
        emb_B = emb_B.numpy()

    score = 0.0
    for i in range(d):
        score += wasserstein_distance(emb_A[:, i], emb_B[:, i])

    return score / float(d)


# =====================================================================================================
# SINGLE PAIR DISTANCE (MULTIPROCESS-SAFE)
# =====================================================================================================
def compute_pair_distance(args):

    cls1, cls2 = args

    mask1 = (GLOBAL_QCLS == cls1)
    mask2 = (GLOBAL_QCLS == cls2)

    emb_A = GLOBAL_EMB[mask1]
    emb_B = GLOBAL_EMB[mask2]

    dist = mean_emd(emb_A, emb_B, normalize=False)
    return (cls1, cls2), dist


# =====================================================================================================
# MAIN EVALUATION
# =====================================================================================================
def evaluate_dino_experiment(cfg: Dict):

    multiprocessing.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = str(cfg.get("results_root", "./Results/DINO_default"))
    ensure_dir(results_root)

    ckpt = os.path.join(results_root, "checkpoints", "final_weights.pth")
    emb_cache_path = os.path.join(results_root, "cached_embeddings.pkl")

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

    # ----------------------------------------------------------------------------------
    # EMBEDDING CACHE
    # ----------------------------------------------------------------------------------
    if os.path.isfile(emb_cache_path):

        print("Loading cached embeddings...")
        with open(emb_cache_path, "rb") as f:
            cache = pickle.load(f)

        embeddings = cache["embeddings"]
        q_cls = cache["q_cls"]

    else:

        print("Computing embeddings (first run)...")

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

        with open(emb_cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "q_cls": q_cls}, f)

        print("Cached embeddings saved.")

    # ----------------------------------------------------------------------------------
    # PAIRWISE EMD (MULTIPROCESSING)
    # ----------------------------------------------------------------------------------
    unique_classes = list(range(D))
    all_class_pairs = list(combinations(unique_classes, 2))

    worker_args = [(cls1, cls2) for cls1, cls2 in all_class_pairs]

    dist_dict = {(cls, cls): 0.0 for cls in unique_classes}

    hostname = os.uname().nodename
    if "orion" in hostname:
        cpu_count = os.cpu_count()   # full parallel
        cpu_count = 6
    else:
        cpu_count = 1  

    print(f"Using {cpu_count} CPU processes for EMD")

    print("Starting EMD computation...")

    """
    t0 = time.perf_counter()

    with Pool(
        processes=cpu_count,
        initializer=init_worker,
        initargs=(embeddings, q_cls),
    ) as pool:

        results = list(
            tqdm(
                pool.imap(compute_pair_distance, worker_args, chunksize=20),
                total=len(worker_args),
                desc="Computing EMD"
            )
        )

    t1 = time.perf_counter()

    total_time = t1 - t0
    mean_time = total_time / float(len(worker_args))

    print(f"Total EMD computation time: {total_time:.3f} s")
    print(f"Mean time per distance: {mean_time:.6f} s")
    """

    cpu_times = {}

    for cpu_count in range(1, 13):

        print(f"\nBenchmarking with {cpu_count} CPU(s)...")
        
        t0 = time.perf_counter()

        with Pool(
            processes=cpu_count,
            initializer=init_worker,
            initargs=(embeddings, q_cls),
        ) as pool:

            results = list(
                tqdm(
                    pool.imap(compute_pair_distance, worker_args, chunksize=20),
                    total=len(worker_args),
                    desc=f"Computing EMD ({cpu_count} CPU)",
                )
            )

        t1 = time.perf_counter()

        total_time = t1 - t0
        mean_time = total_time / float(len(worker_args))

        cpu_times[cpu_count] = {
            "total_time": total_time,
            "mean_time": mean_time,
        }

        print(f"CPUs: {cpu_count:2d} | Total time: {total_time:.3f} s | Mean: {mean_time:.6f} s")


    # ----------------------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------------------
    print("\n================ CPU BENCHMARK SUMMARY ================\n")

    best_cpu = None
    best_time = float("inf")

    for cpu_count in sorted(cpu_times):
        t = cpu_times[cpu_count]["total_time"]
        m = cpu_times[cpu_count]["mean_time"]

        print(
            f"CPUs: {cpu_count:2d} | "
            f"Total: {t:8.3f} s | "
            f"Mean: {m:.6f} s"
        )

        if t < best_time:
            best_time = t
            best_cpu = cpu_count

    print("\n======================================================")
    print(f"Best CPU count: {best_cpu}")
    print(f"Best total time: {best_time:.3f} s")
    print("======================================================\n")




    for (cls1, cls2), dist in results:
        dist_dict[(cls1, cls2)] = dist
        dist_dict[(cls2, cls1)] = dist

    out_file = os.path.join(results_root, "emd_distances_not_normalized_part0.pkl")

    with open(out_file, "wb") as f:
        pickle.dump(dist_dict, f)

    return dist_dict, valid_drugs
