import os
import time
import math
import multiprocessing
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed

from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.utils import ensure_dir


#######################################################################################################
# TIME FORMATTER (HOURS / MINUTES / SECONDS)
#######################################################################################################
def hms(seconds: float) -> str:
    """
    Safely format a duration (in seconds) into 'Hh Mm Ss.s' format.
    Handles None, negative, NaN, and infinite values.
    """

    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "0h 0m 0.0s"

    if not math.isfinite(seconds):
        return "0h 0m 0.0s"

    if seconds < 0:
        seconds = 0.0

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h}h {m}m {s:.1f}s"


#######################################################################################################
# PROGRESS MONITOR
#######################################################################################################
def monitor_progress(global_counter, total_pairs, start_time, refresh=5, tag=""):
    last = -1
    while True:
        time.sleep(refresh)
        with global_counter.get_lock():
            done = global_counter.value

        if done == last:
            continue
        last = done

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total_pairs - done) / rate if rate > 0 else float("inf")
        pct = 100.0 * done / total_pairs if total_pairs > 0 else 100.0

        prefix = f"[{tag}] " if tag else ""
        print(
            f"{prefix}Distances computed: {done}/{total_pairs} "
            f"({pct:.0f}%) | Elapsed: {hms(elapsed)} | Remaining: {hms(remaining)}",
            flush=True,
        )

        if done >= total_pairs:
            break


#######################################################################################################
# LOAD TRAINED STUDENT
#######################################################################################################
def load_trained_student(checkpoint_path: str, cfg: Dict, device: str = "cuda"):

    patch_size = int(cfg.get("patch_size", 8))

    in_channels = str(cfg.get("in_channels", "both")).lower()
    in_chans_map = {
        "egfp": 1,
        "dapi": 1,
        "both": 2,
    }
    if in_channels not in in_chans_map:
        raise ValueError(f"Invalid in_channels={in_channels}")
    in_chans = in_chans_map[in_channels]

    out_dim = int(cfg.get("out_dim", 8192))
    architecture = str(cfg.get("architecture", "tiny"))

    backbone = create_vit_small_backbone(
        architecture=architecture,
        patch_size=patch_size,
        in_chans=in_chans,
    )

    head = DINOHead(
        in_dim=backbone.num_features,
        out_dim=out_dim,
    )

    student = DINOStudent(backbone, head).to(device)

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
def get_npy_folder_from_drug(drug_name: str, cfg: Dict) -> Optional[str]:

    df = pd.read_csv(cfg["label_path"], header=None, dtype=str)

    matches = df[df.iloc[:, 2].str.strip() == drug_name.strip()]
    if len(matches) == 0:
        return None

    plate = matches.iloc[0, 0].strip()
    well = matches.iloc[0, 1].strip()

    folder = os.path.join(cfg["data_root"], plate, well)
    return folder if os.path.isdir(folder) else None


#######################################################################################################
# EMBEDDING COMPUTATION
#######################################################################################################
@torch.no_grad()
def compute_embeddings_for_drug_folder(
    student,
    folder,
    in_channels: str,
    device="cuda",
    batch_size=128,
):

    in_channels = in_channels.lower()
    channel_map = {
        "egfp": [0],
        "dapi": [1],
        "both": [0, 1],
    }
    if in_channels not in channel_map:
        raise ValueError(f"Invalid in_channels={in_channels}")

    chans = channel_map[in_channels]

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))

    if len(files) == 0:
        return torch.empty(0, student.backbone.num_features)

    out = []

    for i in range(0, len(files), batch_size):

        batch_files = files[i:i + batch_size]
        batch_list = []

        for f in batch_files:
            path = os.path.join(folder, f)
            try:
                arr = np.load(path).astype(np.float32)  # (2, H, W)
                arr = arr[chans]                       # (C, H, W)
                batch_list.append(arr)
            except Exception:
                continue

        if len(batch_list) == 0:
            continue

        batch = torch.from_numpy(np.stack(batch_list, axis=0))
        batch = batch.to(device, non_blocking=True)

        z = student.backbone(batch)
        out.append(z.cpu())

        del batch

    if len(out) == 0:
        return torch.empty(0, student.backbone.num_features)

    return torch.cat(out, dim=0)


#######################################################################################################
# DISTANCE METRICS
#######################################################################################################
def mean_emd_numpy(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
    normalize: bool = False,
) -> float:

    if emb_A.size == 0 or emb_B.size == 0:
        return float("nan")

    d = emb_A.shape[1]

    if normalize:
        emb_A = F.normalize(torch.from_numpy(emb_A), p=2, dim=1).numpy()
        emb_B = F.normalize(torch.from_numpy(emb_B), p=2, dim=1).numpy()

    score = 0.0
    for i in range(d):
        score += wasserstein_distance(emb_A[:, i], emb_B[:, i])

    return score / float(d)


def prototype_distance_numpy(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
    normalize: bool = False,
) -> float:

    if emb_A.size == 0 or emb_B.size == 0:
        return float("nan")

    proto_A = emb_A.mean(axis=0)
    proto_B = emb_B.mean(axis=0)

    if normalize:
        proto_A = F.normalize(torch.from_numpy(proto_A[None, :]), p=2, dim=1).numpy()[0]
        proto_B = F.normalize(torch.from_numpy(proto_B[None, :]), p=2, dim=1).numpy()[0]

    diff = proto_A - proto_B
    return float(np.sqrt(np.sum(diff * diff)))


#######################################################################################################
# HYBRID WORKER (ONE PROCESS + MULTI-THREADING)
#######################################################################################################
def hybrid_worker(
    proc_id,
    n_processes,
    class_embeddings,
    unique_classes,
    results_root,
    normalize,
    n_threads,
    global_counter,
    metric,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    metric = str(metric).lower().strip()
    if metric not in {"emd", "prototype"}:
        raise ValueError(f"Invalid metric={metric}")

    def compute_row(cls1: int) -> np.ndarray:
        emb_A = class_embeddings[cls1]
        row_list = []

        for cls2 in unique_classes:
            if cls2 <= cls1:
                continue

            emb_B = class_embeddings[cls2]
            if metric == "emd":
                dist = mean_emd_numpy(emb_A, emb_B, normalize=normalize)
            else:
                dist = prototype_distance_numpy(emb_A, emb_B, normalize=normalize)

            row_list.append((cls1, cls2, dist))

        if len(row_list) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return np.asarray(row_list, dtype=np.float32)

    assigned_rows = [
        cls for i, cls in enumerate(unique_classes)
        if i % n_processes == proc_id
    ]

    results_chunks = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(compute_row, cls) for cls in assigned_rows]

        for future in as_completed(futures):
            res = future.result()
            if res.size > 0:
                results_chunks.append(res)
                with global_counter.get_lock():
                    global_counter.value += res.shape[0]

    if len(results_chunks) > 0:
        results = np.concatenate(results_chunks, axis=0)
    else:
        results = np.zeros((0, 3), dtype=np.float32)

    out_file = os.path.join(results_root, f"dist_block_{metric}_proc{proc_id}.npy")
    tmp_file = out_file + ".tmp.npy"

    np.save(tmp_file, results)
    os.replace(tmp_file, out_file)


#######################################################################################################
# INTERNAL HELPERS
#######################################################################################################
def _get_base_dir_from_file() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if "/Todo_List" in base_dir:
        base_dir = base_dir.replace("/Todo_List", "")
    return base_dir


def _select_subset_drugs(cfg: Dict, subset: str) -> Tuple[List[str], str]:
    subset = str(subset).lower().strip()
    if subset in {"callibration", "calibration", "callib"}:
        if "callibration_path" not in cfg:
            raise KeyError("cfg['callibration_path'] is required for subset='callibration'")
        subset_drugs = load_drug_list(cfg["callibration_path"])
        suffix = "_callibration"
    elif subset in {"all", "entire", "full", "labels", "label"}:
        subset_drugs = load_drug_list(cfg["label_path"])
        suffix = ""
    else:
        raise ValueError(f"Invalid subset={subset}")
    return subset_drugs, suffix


def _metric_tag(metric: str) -> str:
    metric = str(metric).lower().strip()
    if metric == "emd":
        return "EMD"
    if metric == "prototype":
        return "PROTO"
    raise ValueError(f"Invalid metric={metric}")


#######################################################################################################
# MAIN ENTRY POINT
#######################################################################################################
def evaluate_computeDistanceMatrix(
    cfg: Dict,
    subset: str = "all",
    metric: str = "emd",
    normalize: bool = False,
    n_processes: int = 3,
    n_threads: int = 3,
    batch_size: int = 128,
    refresh: int = 5,
):

    multiprocessing.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_name = str(cfg.get("experiment_name", "DINO_experiment"))
    base_dir = _get_base_dir_from_file()
    results_root = ensure_dir(f"{base_dir}/Results/{experiment_name}")

    ckpt = f"{base_dir}/Results/{experiment_name}/checkpoints/final_weights.pth"

    in_channels = str(cfg.get("in_channels", "both")).lower()
    if in_channels not in {"egfp", "dapi", "both"}:
        raise ValueError(f"Invalid in_channels={in_channels}")

    metric = str(metric).lower().strip()
    if metric not in {"emd", "prototype"}:
        raise ValueError(f"Invalid metric={metric}")

    subset_drugs, subset_suffix = _select_subset_drugs(cfg, subset)
    metric_tag = _metric_tag(metric)

    student = load_trained_student(ckpt, cfg, device=device)

    emb_cache_path = os.path.join(results_root, f"cached_embeddings_{in_channels}.pt")

    # ======================================================================
    # 1) BUILD ALL VALID DRUGS (FROM LABELS, WITH EXISTING FOLDERS)
    # ======================================================================
    if subset == "callibration":
        all_label_drugs = subset_drugs
    else:
        all_label_drugs = load_drug_list(cfg["label_path"])

    all_valid_drugs = []
    folder_list = []
    for drug in all_label_drugs:
        folder = get_npy_folder_from_drug(drug, cfg)
        if folder is not None:
            all_valid_drugs.append(drug)
            folder_list.append(folder)

    # ======================================================================
    # 2) SUBSET FILTERING
    # ======================================================================
    valid_drugs = [d for d in subset_drugs if d in all_valid_drugs]
    D = len(valid_drugs)

    if D == 0:
        raise RuntimeError("No valid drugs found for the requested subset.")

    # ======================================================================
    # 3) EMBEDDING CACHE (FOR ALL VALID DRUGS)
    # ======================================================================

    print("Computing embeddings...")   

    cache_ok = False
    embeddings = None
    q_cls = None

    if os.path.isfile(emb_cache_path):
        cache = torch.load(emb_cache_path, map_location="cpu")
        embeddings = cache.get("embeddings", None)
        q_cls = cache.get("q_cls", None)
        cached_drugs = cache.get("all_valid_drugs", None)

        if embeddings is not None and q_cls is not None and cached_drugs is not None:
            if list(cached_drugs) == list(all_valid_drugs):
                cache_ok = True

    if not cache_ok:
        all_embeddings = []
        q_cls_list = []

        for global_idx, folder in enumerate(folder_list):
            Z = compute_embeddings_for_drug_folder(
                student,
                folder,
                in_channels=in_channels,
                device=device,
                batch_size=batch_size,
            )
            all_embeddings.append(Z)
            q_cls_list.extend([global_idx] * Z.shape[0])

        if len(all_embeddings) == 0:
            raise RuntimeError("No embeddings computed (no .npy files found).")

        embeddings = torch.cat(all_embeddings, dim=0).cpu()
        q_cls = np.asarray(q_cls_list, dtype=np.int32)

        torch.save(
            {"embeddings": embeddings, "q_cls": q_cls, "all_valid_drugs": all_valid_drugs},
            emb_cache_path,
        )

    # ======================================================================
    # 4) BUILD CLASS EMBEDDINGS FOR SUBSET
    # ======================================================================
    emb_np = embeddings.numpy()
    subset_indices = [all_valid_drugs.index(d) for d in valid_drugs]
    unique_classes = list(range(len(subset_indices)))

    class_embeddings = {}
    for local_cls, global_cls in enumerate(subset_indices):
        mask = (q_cls == global_cls)
        class_embeddings[local_cls] = emb_np[mask]

    # ======================================================================
    # 5) PAIRWISE DISTANCES (HYBRID)
    # ======================================================================
    total_pairs = int(D * (D - 1) / 2)
    global_counter = multiprocessing.Value("i", 0)
    start_time = time.time()

    processes = []
    for pid in range(int(n_processes)):
        p = multiprocessing.Process(
            target=hybrid_worker,
            args=(
                pid,
                int(n_processes),
                class_embeddings,
                unique_classes,
                results_root,
                bool(normalize),
                int(n_threads),
                global_counter,
                metric,
            ),
        )
        p.start()
        processes.append(p)

    monitor_progress(
        global_counter,
        total_pairs,
        start_time,
        refresh=int(refresh),
        tag=f"{metric_tag}{subset_suffix}",
    )

    for p in processes:
        p.join()

    # ======================================================================
    # 6) MERGE BLOCKS
    # ======================================================================
    Dmat = np.zeros((D, D), dtype=np.float32)

    for pid in range(int(n_processes)):
        block_path = os.path.join(results_root, f"dist_block_{metric}_proc{pid}.npy")
        if not os.path.isfile(block_path):
            continue
        block = np.load(block_path)
        for i, j, dist in block:
            i = int(i)
            j = int(j)
            Dmat[i, j] = float(dist)
            Dmat[j, i] = float(dist)

    np.fill_diagonal(Dmat, 0.0)

    csv_name = f"DINO_{experiment_name}_{metric_tag}_DIST{subset_suffix}.csv"
    csv_path = os.path.join(results_root, csv_name)

    df_mat = pd.DataFrame(Dmat, index=valid_drugs, columns=valid_drugs)
    df_mat.to_csv(csv_path)

    return csv_path