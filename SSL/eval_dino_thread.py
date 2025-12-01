import os
import time
import pickle
from typing import Dict, List
from itertools import combinations  # kept in case you want to reuse, but not needed now

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# LOAD DRUG LIST
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
# EMBEDDING COMPUTATION
#######################################################################################################
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


#######################################################################################################
# MEAN MARGINAL EMD (NUMPY VERSION)
#######################################################################################################
def mean_emd_numpy(emb_A: np.ndarray,
                   emb_B: np.ndarray,
                   normalize: bool = False) -> float:

    d = emb_A.shape[1]

    if normalize:
        emb_A_t = F.normalize(torch.from_numpy(emb_A), p=2, dim=1).numpy()
        emb_B_t = F.normalize(torch.from_numpy(emb_B), p=2, dim=1).numpy()
    else:
        emb_A_t = emb_A
        emb_B_t = emb_B

    score = 0.0
    for i in range(d):
        score += wasserstein_distance(emb_A_t[:, i], emb_B_t[:, i])

    return score / float(d)


#######################################################################################################
# MAIN EVALUATION (SINGLE PROCESS + MULTITHREADING)
#######################################################################################################
def evaluate_dino_experiment(cfg: Dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = str(cfg.get("results_root", "./Results/DINO_default"))
    ensure_dir(results_root)

    ckpt = os.path.join(results_root, "checkpoints", "final_weights.pth")
    emb_cache_path = os.path.join(results_root, "cached_embeddings.pkl")

    student = load_trained_student(ckpt, cfg, device=device)

    drugs = load_drug_list(cfg["callibration_path"])

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
    # BUILD PER-CLASS EMBEDDINGS (ONE BLOCK PER DRUG)
    # ----------------------------------------------------------------------------------
    print("Building per-class embedding blocks for threading...")
    emb_np = embeddings.numpy()  # embeddings are on CPU already

    unique_classes = list(range(D))
    class_embeddings = {}

    for cls in unique_classes:
        mask = (q_cls == cls)
        class_embeddings[cls] = emb_np[mask]

    print(f"Built {len(class_embeddings)} class embedding blocks.")

    # ----------------------------------------------------------------------------------
    # THREADED PAIRWISE EMD (ONE THREAD PER 'ROW' OF THE MATRIX)
    # ----------------------------------------------------------------------------------
    def compute_row_distances(cls1: int):
        emb_A = class_embeddings[cls1]
        row_results = []

        for cls2 in unique_classes:
            if cls2 <= cls1:
                continue
            emb_B = class_embeddings[cls2]
            dist = mean_emd_numpy(emb_A, emb_B, normalize=False)
            row_results.append(((cls1, cls2), dist))

        return row_results


    max_threads = min(12, os.cpu_count() or 1)
    thread_times = {}

    print(f"\nBenchmarking thread counts from 1 to {max_threads}...\n")

    for n_threads in range(1, max_threads + 1):

        print(f"Benchmarking with {n_threads} thread(s)...")

        t0 = time.perf_counter()
        results_tmp = []

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_cls = {
                executor.submit(compute_row_distances, cls1): cls1
                for cls1 in unique_classes
            }

            for future in as_completed(future_to_cls):
                results_tmp.extend(future.result())

        t1 = time.perf_counter()

        total_time = t1 - t0
        mean_time = total_time / (D * (D - 1) / 2)

        thread_times[n_threads] = (total_time, mean_time)

        print(
            f"Threads: {n_threads:2d} | "
            f"Total: {total_time:8.3f} s | "
            f"Mean: {mean_time:.6f} s"
        )


    # ----------------------------------------------------------------------------------
    # SELECT BEST THREAD COUNT
    # ----------------------------------------------------------------------------------
    print("\n================ THREAD BENCHMARK SUMMARY ================\n")

    best_threads = None
    best_time = float("inf")

    for n_threads in thread_times:
        t, m = thread_times[n_threads]
        print(f"Threads: {n_threads:2d} | Total: {t:8.3f} s | Mean: {m:.6f} s")

        if t < best_time:
            best_time = t
            best_threads = n_threads

    print("\n========================================================")
    print(f"Best thread count: {best_threads}")
    print(f"Best total time:   {best_time:.3f} s")
    print("========================================================\n")


    # ----------------------------------------------------------------------------------
    # FINAL RUN WITH BEST THREAD COUNT
    # ----------------------------------------------------------------------------------
    print(f"Running final EMD computation with {best_threads} threads...")

    dist_dict = {(cls, cls): 0.0 for cls in unique_classes}

    t0 = time.perf_counter()
    final_results = []

    with ThreadPoolExecutor(max_workers=best_threads) as executor:
        future_to_cls = {
            executor.submit(compute_row_distances, cls1): cls1
            for cls1 in unique_classes
        }

        for future in tqdm(as_completed(future_to_cls),
                        total=len(future_to_cls),
                        desc="Computing EMD (final)"):
            final_results.extend(future.result())

    t1 = time.perf_counter()

    total_time = t1 - t0
    mean_time = total_time / (D * (D - 1) / 2)

    print(f"\nFINAL EMD TIME: {total_time:.3f} s")
    print(f"FINAL MEAN PER DISTANCE: {mean_time:.6f} s\n")

    for (cls1, cls2), dist in final_results:
        dist_dict[(cls1, cls2)] = dist
        dist_dict[(cls2, cls1)] = dist
