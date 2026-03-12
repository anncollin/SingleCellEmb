import os
from typing import Dict

import torch

from SSL.eval_computeDistanceMatrix import load_trained_student
from SSL.utils import compute_expert_annotation_metric


@torch.no_grad()
def evaluate_expertAnnotation(
    cfg: Dict,
    metric: str = "prototype",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_name = cfg["experiment_name"]
    in_channels = cfg.get("in_channels", "both")

    # ------------------------------------------------------------------
    # visualize some population samples
    # ------------------------------------------------------------------
    from Dataset.data_loader import PopulationDataset
    from SSL.utils import visualize_population_samples
    dataset = PopulationDataset(
        root_dir=cfg["data_root"],
        wells_csv=cfg["label_path"],
        in_channels=in_channels,
    )

    visualize_population_samples(dataset)
    # ------------------------------------------------------------------

    base_dir = os.getcwd().replace("/Todo_List", "")
    results_root = os.path.join(base_dir, "Results", experiment_name)

    ckpt_path = os.path.join(results_root, "checkpoints", "final_weights.pth")

    student = load_trained_student(ckpt_path, cfg, device=device)

    score = compute_expert_annotation_metric(
        student=student,
        data_root=cfg["data_root"],
        annotations_csv="./SSL/annotations.csv",
        in_channels=in_channels,
        device=device,
        metric=metric,
    )

    print("-" * 60)
    print(f"Expert annotation satisfaction: {score:.2f}%")
    print("-" * 60)

    return score