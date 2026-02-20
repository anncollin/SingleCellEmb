import os
from typing import Dict

import pandas as pd
import torch

from SSL.utils import compute_expert_annotation_metric
from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.eval_computeDistanceMatrix import (
    load_trained_student,
    compute_embeddings_for_drug_folder,
    mean_emd_numpy,
)

@torch.no_grad()
def evaluate_expertAnnotation(
    cfg: Dict,
    metric: str = "prototype",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name = cfg["experiment_name"]
    in_channels = cfg.get("in_channels", "both")

    base_dir = os.getcwd().replace("/Todo_List", "")
    results_root = os.path.join(base_dir, "Results", experiment_name)

    ckpt_path = os.path.join(results_root, "checkpoints", "final_weights.pth")
    student = load_trained_student(ckpt_path, cfg, device=device)

    # ---------------------------------------------------------
    # PROTOTYPE CASE (already implemented)
    # ---------------------------------------------------------
    if metric == "prototype":

        score = compute_expert_annotation_metric(
            student=student,
            data_root=cfg["data_root"],
            annotations_csv="./SSL/annotations.csv",
            in_channels=in_channels,
            device=device,
        )

        print("\n" + "-" * 60)
        print(f"Expert annotation satisfaction (PROTOTYPE): {score:.2f}%")
        print("-" * 60)
        return score

    # ---------------------------------------------------------
    # EMD CASE
    # ---------------------------------------------------------
    elif metric == "emd":

        annotations = pd.read_csv("./SSL/annotations.csv")

        total = 0
        valid = 0

        cache = {}

        def get_embeddings(pop_path):
            if pop_path not in cache:
                folder = os.path.join(
                    cfg["data_root"],
                    pop_path.replace(".zip", "")
                )
                Z = compute_embeddings_for_drug_folder(
                    student,
                    folder,
                    in_channels=in_channels,
                    device=device,
                )
                cache[pop_path] = Z.cpu().numpy()
            return cache[pop_path]

        for _, row in annotations.iterrows():

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

            embA = get_embeddings(A)
            embB = get_embeddings(B)
            embC = get_embeddings(C)

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

        score = 0.0 if total == 0 else 100.0 * valid / total

        print("\n" + "-" * 60)
        print(f"Expert annotation satisfaction (EMD): {score:.2f}%")
        print("-" * 60)
        return score

    else:
        raise ValueError("metric must be 'prototype' or 'emd'")