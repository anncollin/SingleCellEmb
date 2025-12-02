import os
import time
import pickle
import multiprocessing
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
            [np.load(os.path.join(folder, f)).astype(np.float32)
             for f in batch_files],
            axis=0
        )

        batch = torch.from_numpy(batch).to(device, non_blocking=True)
        z = student.backbone(batch)
        out.append(z.cpu())

        del batch

    return torch.cat(out, dim=0)


#######################################################################################################
# MEAN MARGINAL EMD
#######################################################################################################
def mean_emd_numpy(emb_A: np.ndarray,
                   emb_B: np.ndarray,
                   normalize: bool = False) -> float:

    d = emb_A.shape[1]

    if normalize:
        emb_A = F.normalize(torch.from_numpy(emb_A), p=2, dim=1).numpy()
        emb_B = F.normalize(torch.from_numpy(emb_B), p=2, dim=1).numpy()

    score = 0.0
    for i in range(d):
        score += wasserstein_distance(emb_A[:, i], emb_B[:, i])

    return score / float(d)


#######################################################################################################
# HYBRID WORKER (ONE PROCESS + MULTI-THREADING) — GLOBAL PROGRESS VERSION
#######################################################################################################
def hybrid_worker(proc_id,
                  n_processes,
                  class_embeddings,
                  unique_classes,
                  results_root,
                  normalize,
                  n_threads,
                  global_counter):

    def compute_row(cls1: int):
        emb_A = class_embeddings[cls1]
        row_results = []

        for cls2 in unique_classes:
            if cls2 <= cls1:
                continue
            emb_B = class_embeddings[cls2]
            dist = mean_emd_numpy(emb_A, emb_B, normalize=normalize)
            row_results.append(((cls1, cls2), dist))

        return row_results

    assigned_rows = [
        cls for i, cls in enumerate(unique_classes)
        if i % n_processes == proc_id
    ]

    print(f"[Process {proc_id}] Started ({len(assigned_rows)} rows)")

    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:

        futures = [
            executor.submit(compute_row, cls)
            for cls in assigned_rows
        ]

        for future in as_completed(futures):
            res = future.result()
            results.extend(res)

            # Global distance count update
            with global_counter.get_lock():
                global_counter.value += len(res)

    out_file = os.path.join(results_root, f"emd_block_proc{proc_id}.pkl")

    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    print(
        f"[Process {proc_id}] Finished | "
        f"Saved {len(results)} distances"
    )


#######################################################################################################
# MAIN EVALUATION ENTRY POINT
#######################################################################################################
def evaluate_dino_experiment(cfg: Dict):

    multiprocessing.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    experiment_name = str(cfg.get("experiment_name", "DINO_experiment"))
    results_root    = ensure_dir(f"./Results/{experiment_name}")
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

        n_folders = len(folder_list)
        print(f"\nEmbedding drugs: {n_folders} folders")

        t0_embed = time.perf_counter()

        for i, folder in enumerate(folder_list):

            if len(drugs) > 50:
                percent = int(100.0 * (i + 1) / n_folders)
                prev_percent = int(100.0 * i / n_folders) if i > 0 else -1

                if percent != prev_percent:
                    elapsed = time.perf_counter() - t0_embed
                    print(
                        f"[Embedding] {i+1}/{n_folders} folders "
                        f"({percent}%) | Elapsed: {elapsed:.1f} s"
                    )

            Z = compute_embeddings_for_drug_folder(
                student,
                folder,
                device=device,
                batch_size=128,
            )

            all_embeddings.append(Z)
            q_cls.extend([i] * Z.shape[0])


        with open(emb_cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "q_cls": q_cls}, f)

        print("Cached embeddings saved.")

    # ----------------------------------------------------------------------------------
    # BUILD PER-CLASS EMBEDDINGS
    # ----------------------------------------------------------------------------------
    emb_np = embeddings.numpy()
    unique_classes = list(range(D))
    class_embeddings = {}

    for cls in unique_classes:
        mask = (q_cls == cls)
        class_embeddings[cls] = emb_np[mask]

    # ----------------------------------------------------------------------------------
    # RUN HYBRID COMPUTATION (HARD-CODED 3 PROCESSES × 3 THREADS)
    # ----------------------------------------------------------------------------------
    n_processes = 3
    n_threads   = 3
    normalize   = False

    print("\n================ HYBRID RUN (3 PROCESSES × 3 THREADS) ================\n")

    total_pairs = int(D * (D - 1) / 2)
    global_counter = multiprocessing.Value("i", 0)

    t0 = time.perf_counter()
    processes = []

    for pid in range(n_processes):
        p = multiprocessing.Process(
            target=hybrid_worker,
            args=(
                pid,
                n_processes,
                class_embeddings,
                unique_classes,
                results_root,
                normalize,
                n_threads,
                global_counter,
            )
        )
        p.start()
        processes.append(p)

    # ---- GLOBAL % OF DISTANCES ----
    last_percent = -1

    while any(p.is_alive() for p in processes):

        with global_counter.get_lock():
            done = global_counter.value

        if total_pairs > 0:
            percent = int(100.0 * done / total_pairs)
        else:
            percent = 100

        if percent != last_percent:
            print(
                f"[Hybrid] Distances computed: {done}/{total_pairs} "
                f"({percent}%)"
            )
            last_percent = percent

        time.sleep(5)

    for p in processes:
        p.join()

    # ----------------------------------------------------------------------------------
    # MERGE PARTIAL DISTANCE BLOCKS INTO FULL MATRIX AND SAVE AS CSV
    # ----------------------------------------------------------------------------------
    print("Merging distance blocks into full CSV matrix...")

    Dmat = np.zeros((D, D), dtype=np.float32)

    for pid in range(n_processes):
        block_path = os.path.join(results_root, f"emd_block_proc{pid}.pkl")

        if not os.path.isfile(block_path):
            raise RuntimeError(f"Missing block file: {block_path}")

        with open(block_path, "rb") as f:
            block = pickle.load(f)

        for (i, j), dist in block:
            Dmat[i, j] = dist
            Dmat[j, i] = dist

    np.fill_diagonal(Dmat, 0.0)

    csv_path = os.path.join(results_root, "emd_full_matrix.csv")
    df_mat = pd.DataFrame(Dmat, index=valid_drugs, columns=valid_drugs)
    df_mat.to_csv(csv_path)

    print(f"Full distance matrix saved to: {csv_path}")

    # ----------------------------------------------------------------------------------
    # CLEAN UP PARTIAL BLOCK FILES
    # ----------------------------------------------------------------------------------

    for pid in range(n_processes):
        block_path = os.path.join(results_root, f"emd_block_proc{pid}.pkl")

        if os.path.isfile(block_path):
            os.remove(block_path)
        else:
            print(f"Warning: block file not found: {block_path}")


    # ----------------------------------------------------------------------------------
    # FINAL TIMING
    # ----------------------------------------------------------------------------------
    t1 = time.perf_counter()

    total_time = t1 - t0
    mean_time  = total_time / total_pairs if total_pairs > 0 else float("nan")

    print("\n================ HYBRID RUN FINISHED ================\n")
    print(f"Processes: 3 | Threads: 3")
    print(f"Total time: {total_time:.3f} s")
    print(f"Mean time per pair: {mean_time:.6f} s")

    return total_time, mean_time, valid_drugs
