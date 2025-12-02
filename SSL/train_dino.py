import math
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import random
import numpy as np
from sklearn.metrics import silhouette_score
import wandb

from Dataset.data_loader import CellDataset
from SSL.transforms import KorniaMultiCropTransform
from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.loss import DINOLoss
from SSL.utils import update_teacher, ensure_dir, save_checkpoint


#######################################################################################################
# COMPUTE SILHOUETTE SCORE (FAIL-SAFE VERSION)
#######################################################################################################
@torch.no_grad()
def compute_silhouette(student, dataset, gpu_transform, device="cuda", num_samples=500):

    try:
        student.eval()

        N = min(num_samples, len(dataset))
        if N < 2:
            print("Warning: Not enough samples for silhouette score.")
            return float("nan")

        idxs = random.sample(range(len(dataset)), N)

        embeddings = []
        labels = []

        for idx in idxs:

            img = dataset[idx].unsqueeze(0).to(device)

            crops = gpu_transform(img)
            img = crops[0]

            z = student.backbone(img).squeeze(0).cpu().numpy()
            embeddings.append(z)

            path = dataset.npy_files[random.randrange(len(dataset.npy_files))]
            labels.append(1 if "siRNA" in path else 0)

        embeddings = np.stack(embeddings)
        labels = np.array(labels)

        # Silhouette requires at least 2 distinct labels
        if len(np.unique(labels)) < 2:
            print("Warning: Only one class present in silhouette labels.")
            return float("nan")

        return silhouette_score(embeddings, labels)

    except Exception as e:
        print(f"Warning: Silhouette computation failed: {str(e)}")
        return float("nan")



#######################################################################################################
# COMPUTE ALL METRICS (GPU MULTI-CROP COMPATIBLE)
#######################################################################################################
@torch.no_grad()
def compute_all_metrics(student, teacher, dataloader, gpu_transform, device="cuda", num_batches=5):

    student.eval()
    teacher.eval()

    student_entropies = []
    teacher_entropies = []

    student_embeddings = []
    teacher_embeddings = []

    cosine_sims = []
    proto_indices = []

    for batch_idx, images in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = images.to(device, non_blocking=True)

        crops = gpu_transform(images)
        x = crops[0]

        z_s = student(x)
        z_t = teacher(x)
        h_s = student.backbone(x)
        h_t = teacher.backbone(x)

        p_s = torch.softmax(z_s, dim=-1) + 1e-12
        p_t = torch.softmax(z_t, dim=-1) + 1e-12

        student_entropies.append(float(-(p_s * p_s.log()).sum(dim=1).mean()))
        teacher_entropies.append(float(-(p_t * p_t.log()).sum(dim=1).mean()))

        student_embeddings.append(h_s.cpu())
        teacher_embeddings.append(h_t.cpu())

        z_s_norm = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-12)
        z_t_norm = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-12)
        cosine_sims.append(float((z_s_norm * z_t_norm).sum(dim=-1).mean()))

        proto_idx = z_s.argmax(dim=-1).cpu()
        proto_indices.append(proto_idx)

    student_entropy = float(np.mean(student_entropies))
    teacher_entropy = float(np.mean(teacher_entropies))

    student_embeddings = torch.cat(student_embeddings, dim=0)
    teacher_embeddings = torch.cat(teacher_embeddings, dim=0)

    student_feature_var = float(student_embeddings.var(dim=0).mean())
    teacher_feature_var = float(teacher_embeddings.var(dim=0).mean())

    student_teacher_cosine = float(np.mean(cosine_sims))

    proto_indices = torch.cat(proto_indices, dim=0)
    out_dim = proto_indices.max().item() + 1
    counts = torch.bincount(proto_indices, minlength=out_dim).float()

    active_dim_fraction = float((counts > 0).sum().item() / counts.numel())
    max_dim_frequency = float(counts.max().item() / proto_indices.numel())

    return {
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "student_feature_variance": student_feature_var,
        "teacher_feature_variance": teacher_feature_var,
        "student_teacher_cosine": student_teacher_cosine,
        "active_dimension_fraction": active_dim_fraction,
        "max_dimension_frequency": max_dim_frequency,
    }


#######################################################################################################
# TRAIN ONE EPOCH OF DINO (HARD-CODED 10% DATASET)
#######################################################################################################
def train_one_epoch(
    student,
    teacher,
    dino_loss: DINOLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    gpu_transform,
    use_wandb: bool, 
    device: str = "cuda",
    base_momentum: float = 0.996,
    max_momentum: float = 1.0,
) -> float:

    student.train()
    teacher.eval()

    total_loss = 0.0
    n_batches = 0

    # ---- HARD-CODE: USE ONLY 10% OF DATASET PER EPOCH ----
    total_batches = len(dataloader)
    max_batches = max(1, int(0.1 * total_batches))
    # --------------------------------------------------------

    dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100, disable=use_wandb)

    for it, images in enumerate(dataloader):

        if it >= max_batches:
            break

        images = images.to(device, non_blocking=True)

        crops = gpu_transform(images)

        student_outputs = [student(c) for c in crops]
        teacher_outputs = [teacher(crops[0]), teacher(crops[1])]

        loss = dino_loss(student_outputs, teacher_outputs, epoch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        momentum = max_momentum - (max_momentum - base_momentum) * (
            math.cos(math.pi * (epoch + it / max_batches)) + 1.0
        ) / 2.0

        update_teacher(student, teacher, momentum)

        total_loss += float(loss.item())
        n_batches += 1

        dataloader.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, n_batches)


#######################################################################################################
# RUN DINO EXPERIMENT (CLEAN PIPELINE, HARD-CODED 10%)
#######################################################################################################
def run_dino_experiment(cfg: Dict):

    use_wandb = cfg.get("use_wandb", True)

    if use_wandb:
        wandb.finish()
        wandb.init(project="DINO", name=cfg.get("experiment_name", "DINO_run"))
        wandb.config.update(cfg, allow_val_change=True)
    else:
        print("Running WITHOUT wandb logging.")

    data_root = cfg["data_root"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = int(cfg.get("image_size", 96))
    in_chans   = int(cfg.get("in_channels", 2))
    patch_size = int(cfg.get("patch_size", 8))
    out_dim    = int(cfg.get("out_dim", 8192))

    batch_size   = int(cfg.get("batch_size", 64))
    epochs       = int(cfg.get("epochs", 50))
    lr           = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))

    num_workers        = int(cfg.get("num_workers", 4))
    local_crops_number = int(cfg.get("local_crops_number", 6))

    global_crops_scale = tuple(float(v) for v in cfg.get("global_crops_scale", [0.4, 1.0]))
    local_crops_scale  = tuple(float(v) for v in cfg.get("local_crops_scale", [0.1, 0.4]))

    base_momentum = float(cfg.get("base_momentum", 0.996))
    max_momentum  = float(cfg.get("max_momentum", 1.0))

    experiment_name = str(cfg.get("experiment_name", "DINO_experiment"))
    results_root    = ensure_dir(f"./Results/{experiment_name}")
    checkpoints_dir = ensure_dir(f"{results_root}/checkpoints")

    gpu_transform = KorniaMultiCropTransform(
        image_size=image_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        local_crops_number=local_crops_number,
        cfg=cfg
    ).to(device)

    dataset = CellDataset(root_dir=data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    backbone_student = create_vit_small_backbone(
        patch_size=patch_size,
        in_chans=in_chans,
    )
    backbone_teacher = create_vit_small_backbone(
        patch_size=patch_size,
        in_chans=in_chans,
    )

    embed_dim = backbone_student.num_features
    head_student = DINOHead(in_dim=embed_dim, out_dim=out_dim)
    head_teacher = DINOHead(in_dim=embed_dim, out_dim=out_dim)

    student = DINOStudent(backbone_student, head_student).to(device)
    teacher = DINOStudent(backbone_teacher, head_teacher).to(device)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(
        out_dim=out_dim,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        nepochs=epochs,
        student_temp=0.1,
        center_momentum=0.9,
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    for epoch in range(epochs):

        avg_loss = train_one_epoch(
            student=student,
            teacher=teacher,
            dino_loss=dino_loss,
            dataloader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            gpu_transform=gpu_transform,
            use_wandb=use_wandb,
            device=device,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )

        metrics = compute_all_metrics(
            student, teacher, dataloader, gpu_transform, device=device, num_batches=5
        )

        if use_wandb:
            wandb.log(metrics, step=epoch)
            wandb.log({"loss": avg_loss}, step=epoch)

        if (epoch + 1) % 5 == 0:
            sil = compute_silhouette(
                student, dataset, gpu_transform, device=device, num_samples=500
            )
            if use_wandb:
                wandb.log({"silhouette": sil}, step=epoch)
            print(f"Silhouette score at epoch {epoch+1}: {sil:.4f}")

        print(f"[{experiment_name}] Epoch {epoch+1}/{epochs} - DINO loss: {avg_loss:.4f}")

    final_ckpt = f"{checkpoints_dir}/final_weights.pth"
    save_checkpoint(
        final_ckpt,
        {
            "epoch": epochs,
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
    )

    print(f"[{experiment_name}] Train finished. Checkpoints saved to {checkpoints_dir}.")
    print(f"[{experiment_name}] Saved final weights to {final_ckpt}.")

    return student, teacher
