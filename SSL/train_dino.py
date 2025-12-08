import math
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import wandb

from Dataset.data_loader import CellDataset
from SSL.utils import visualize_multicrop
from SSL.transforms import KorniaMultiCropTransform
from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.loss import DINOLoss
from SSL.utils import update_teacher, ensure_dir, save_checkpoint


#######################################################################################################
# TIME FORMATTER
#######################################################################################################
def hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"


#######################################################################################################
# COMPUTE BASIC METRICS
#######################################################################################################
@torch.no_grad()
def compute_all_metrics(student, teacher, dataloader, gpu_transform, device="cuda", num_batches=5):

    student.eval()
    teacher.eval()

    student_ent = []
    teacher_ent = []
    student_emb = []
    teacher_emb = []
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

        ps = torch.softmax(z_s, dim=-1) + 1e-12
        pt = torch.softmax(z_t, dim=-1) + 1e-12

        student_ent.append(float(-(ps * ps.log()).sum(dim=1).mean()))
        teacher_ent.append(float(-(pt * pt.log()).sum(dim=1).mean()))

        student_emb.append(h_s.cpu())
        teacher_emb.append(h_t.cpu())

        zs_n = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-12)
        zt_n = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-12)
        cosine_sims.append(float((zs_n * zt_n).sum(dim=-1).mean()))

        idx = z_s.argmax(dim=-1).cpu()
        proto_indices.append(idx)

    student_emb = torch.cat(student_emb, dim=0)
    teacher_emb = torch.cat(teacher_emb, dim=0)

    proto_indices = torch.cat(proto_indices, dim=0)
    out_dim = proto_indices.max().item() + 1
    counts = torch.bincount(proto_indices, minlength=out_dim).float()

    return {
        "student_entropy": float(np.mean(student_ent)),
        "teacher_entropy": float(np.mean(teacher_ent)),
        "student_feature_variance": float(student_emb.var(dim=0).mean()),
        "teacher_feature_variance": float(teacher_emb.var(dim=0).mean()),
        "student_teacher_cosine": float(np.mean(cosine_sims)),
        "active_dimension_fraction": float((counts > 0).sum().item() / counts.numel()),
        "max_dimension_frequency": float(counts.max().item() / proto_indices.numel()),
    }


#######################################################################################################
# TRAIN ONE EPOCH
#######################################################################################################
def train_one_epoch(
    student,
    teacher,
    dino_loss: DINOLoss,
    dataloader: DataLoader,
    optimizer,
    epoch: int,
    gpu_transform,
    use_wandb: bool,
    device="cuda",
    base_momentum=0.996,
    max_momentum=1.0,
):
    student.train()
    teacher.eval()

    total_loss = 0.0
    n_batches  = 0

    total_batches = len(dataloader)
    max_batches   = max(1, int(0.1 * total_batches))  # hardcoded 10%

    data_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100, disable=use_wandb)

    for it, images in enumerate(data_iter):

        if it >= max_batches:
            break

        images = images.to(device, non_blocking=True)

        crops           = gpu_transform(images)
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

        data_iter.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, n_batches)


#######################################################################################################
# MAIN DINO TRAINING
#######################################################################################################
def run_dino_experiment(cfg: Dict):

    use_wandb = cfg.get("use_wandb", True)

    if use_wandb:
        wandb.finish()
        wandb.init(project="DINO", name=cfg.get("experiment_name", "DINO_run"))
        wandb.config.update(cfg, allow_val_change=True)

    data_root = cfg["data_root"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size   = int(cfg.get("image_size", 96))
    in_chans     = int(cfg.get("in_channels", 2))
    patch_size   = int(cfg.get("patch_size", 8))
    out_dim      = int(cfg.get("out_dim", 768))
    architecture = str(cfg.get("architecture", "tiny"))

    batch_size   = int(cfg.get("batch_size", 64))
    epochs       = int(cfg.get("epochs", 50))
    lr           = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))

    num_workers        = int(cfg.get("num_workers", 4))
    local_crops_number = int(cfg.get("local_crops_number", 6))

    global_crops_scale = tuple(cfg.get("global_crops_scale", [0.4, 1.0]))
    local_crops_scale  = tuple(cfg.get("local_crops_scale", [0.1, 0.4]))

    base_momentum = float(cfg.get("base_momentum", 0.996))
    max_momentum  = float(cfg.get("max_momentum", 1.0))

    experiment_name = cfg.get("experiment_name", "DINO_experiment")
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

    # ------------------------------ visualize DA ------------------------------
    visualize_multicrop(dataset, gpu_transform, device)
    # --------------------------------------------------------------------------

    backbone_student = create_vit_small_backbone(
        architecture=architecture,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    backbone_teacher = create_vit_small_backbone(
        architecture=architecture,
        patch_size=patch_size,
        in_chans=in_chans,
    )

    embed_dim = backbone_student.num_features

    head_student = DINOHead(embed_dim, out_dim)
    head_teacher = DINOHead(embed_dim, out_dim)

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
        betas=(0.9, 0.999),
    )

    t0 = time.perf_counter()

    for epoch in range(epochs):

        # Time estimation
        elapsed = time.perf_counter() - t0
        avg_epoch = elapsed / (epoch + 1)
        remaining = avg_epoch * (epochs - epoch - 1)
        print(
            f"[Time] Epoch {epoch+1}/{epochs} | "
            f"Elapsed: {hms(elapsed)} | Remaining: {hms(remaining)}"
        )

        avg_loss = train_one_epoch(
            student,
            teacher,
            dino_loss,
            dataloader,
            optimizer,
            epoch,
            gpu_transform,
            use_wandb,
            device,
            base_momentum,
            max_momentum,
        )

        metrics = compute_all_metrics(student, teacher, dataloader, gpu_transform, device)

        if use_wandb:
            wandb.log(metrics, step=epoch)
            wandb.log({"loss": avg_loss}, step=epoch)

        print(f"[{experiment_name}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    save_checkpoint(
        f"{checkpoints_dir}/final_weights.pth",
        {
            "epoch": epochs,
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
    )

    print(f"[Done] Saved final checkpoint.")
    return student, teacher
