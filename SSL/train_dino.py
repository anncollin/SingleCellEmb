import math
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import random
import numpy as np
from sklearn.metrics import silhouette_score
import wandb

from Dataset.data_loader import CellDataset
from SSL.transforms import MultiCropTransform
from SSL.model import create_vit_small_backbone, DINOHead, DINOStudent
from SSL.loss import DINOLoss
from SSL.utils import update_teacher, ensure_dir, save_checkpoint


#######################################################################################################
# COMPUTE SILHOUETTE SCORE FOR STUDENT BACKBONE REPRESENTATIONS
#######################################################################################################
@torch.no_grad()
def compute_silhouette(student, dataset, device="cuda", num_samples=500):
    N = min(num_samples, len(dataset))
    idxs = random.sample(range(len(dataset)), N)

    embeddings = []
    labels = []

    for idx in idxs:
        # ---- Load image normally (only first global crop is used) ----
        crops = dataset[idx]
        img = crops[0].unsqueeze(0).to(device)

        # ---- Extract embedding from student backbone ----
        z = student.backbone(img).squeeze(0).cpu().numpy()
        embeddings.append(z)

        # ---- Infer the domain label (siRNA vs non-siRNA) ----
        # Note: __getitem__ does not use idx, so we reproduce sampling.
        zip_path = dataset.zip_files[random.randrange(len(dataset.zip_files))]

        if "siRNA" in zip_path:
            labels.append(1)
        else:
            labels.append(0)

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    return silhouette_score(embeddings, labels)

#######################################################################################################
# COMPUTE ALL METRICS IN A SINGLE PASS (FAST)
# ----------------------------------------------------------------------------------------------------
#    Computes:
#        - student_entropy
#        - teacher_entropy
#        - student_feature_variance
#        - teacher_feature_variance
#        - student_teacher_cosine
#        - active_dimension_fraction
#        - max_dimension_frequency
#
#    This replaces 4 different loops and is ~4x faster.
#######################################################################################################
@torch.no_grad()
def compute_all_metrics(student, teacher, dataloader, device="cuda", num_batches=5):

    student.eval()
    teacher.eval()

    # ---------- storage ----------
    student_entropies = []
    teacher_entropies = []

    student_embeddings = []
    teacher_embeddings = []

    cosine_sims = []

    proto_indices = []

    # ---------- iterate ONCE ----------
    for batch_idx, crops in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        x = crops[0].to(device)

        # ---- forward passes ----
        z_s = student(x)          # (B, out_dim)
        z_t = teacher(x)          # (B, out_dim)
        h_s = student.backbone(x) # (B, D)
        h_t = teacher.backbone(x) # (B, D)

        # ========== ENTROPY ==========
        p_s = torch.softmax(z_s, dim=-1) + 1e-12
        p_t = torch.softmax(z_t, dim=-1) + 1e-12

        ent_s = -(p_s * p_s.log()).sum(dim=1).mean().item()
        ent_t = -(p_t * p_t.log()).sum(dim=1).mean().item()

        student_entropies.append(ent_s)
        teacher_entropies.append(ent_t)

        # ========== FEATURE VARIANCE (collect embeddings) ==========
        student_embeddings.append(h_s.detach().cpu())
        teacher_embeddings.append(h_t.detach().cpu())

        # ========== STUDENT–TEACHER COSINE SIMILARITY ==========
        z_s_norm = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-12)
        z_t_norm = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-12)
        cos_sim = (z_s_norm * z_t_norm).sum(dim=-1).mean().item()
        cosine_sims.append(cos_sim)

        # ========== ACTIVE DIMENSIONS / DEAD DIMENSIONS ==========
        proto_idx = z_s.argmax(dim=-1).cpu()
        proto_indices.append(proto_idx)

    # ---------- aggregate all metrics ----------

    # Entropy
    student_entropy = float(sum(student_entropies) / len(student_entropies))
    teacher_entropy = float(sum(teacher_entropies) / len(teacher_entropies))

    # Variance
    if len(student_embeddings) > 0:
        student_embeddings = torch.cat(student_embeddings, dim=0)
        teacher_embeddings = torch.cat(teacher_embeddings, dim=0)
        student_feature_var = float(student_embeddings.var(dim=0).mean().item())
        teacher_feature_var = float(teacher_embeddings.var(dim=0).mean().item())
    else:
        student_feature_var = 0.0
        teacher_feature_var = 0.0

    # Cosine similarity
    student_teacher_cosine = float(sum(cosine_sims) / len(cosine_sims)) if len(cosine_sims) else 0.0

    # Active dimensions / max-dimensional frequency
    if len(proto_indices) > 0:
        proto_indices = torch.cat(proto_indices, dim=0)
        num_samples = proto_indices.numel()
        out_dim = z_s.shape[1]

        counts = torch.bincount(proto_indices, minlength=out_dim).float()
        active_dims = (counts > 0).sum().item()

        active_dim_fraction = float(active_dims / out_dim)
        max_dim_frequency = float(counts.max().item() / num_samples)
    else:
        active_dim_fraction = 0.0
        max_dim_frequency = 0.0

    # ---------- return everything ----------
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
# DINO COLLATE FUNCTION
#######################################################################################################
def dino_collate(batch: List) -> List[torch.Tensor]:
    n_crops = len(batch[0])
    crops: List[torch.Tensor] = []
    for i in range(n_crops):
        crops_i = torch.stack([sample[i] for sample in batch], dim=0)
        crops.append(crops_i)
    return crops


#######################################################################################################
# TRAIN ONE EPOCH OF DINO
#######################################################################################################
def train_one_epoch(
    student,
    teacher,
    dino_loss: DINOLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cuda",
    base_momentum: float = 0.996,
    max_momentum: float = 1.0,
) -> float:
    student.train()
    teacher.eval()

    total_loss = 0.0
    n_batches = 0

    dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)

    for it, crops in enumerate(dataloader):
        if it == 10:
            break

        crops = [c.to(device, non_blocking=True) for c in crops]

        student_outputs = [student(c) for c in crops]
        teacher_outputs = [teacher(crops[0]), teacher(crops[1])]

        loss = dino_loss(student_outputs, teacher_outputs, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum = max_momentum - (max_momentum - base_momentum) * (
            math.cos(math.pi * (epoch + it / len(dataloader))) + 1.0
        ) / 2.0
        update_teacher(student, teacher, momentum)

        total_loss += loss.item()
        n_batches += 1

        dataloader.set_postfix(loss=float(loss.item()))

    return total_loss / max(n_batches, 1)


#######################################################################################################
# RUN DINO EXPERIMENT FROM CONFIG
#######################################################################################################
def run_dino_experiment(cfg: Dict):

    use_wandb = cfg.get("use_wandb", True)

    if use_wandb:
        wandb.init(project="DINO", name=cfg.get("experiment_name", "DINO_run"))
        wandb.config.update(cfg)
    else:
        print("Running WITHOUT wandb logging.")


    data_root = cfg["data_root"]
    device    = "cuda" if torch.cuda.is_available() else "cpu"

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

    results_root    = str(cfg.get("results_root", "./Results/DINO_default"))
    checkpoints_dir = ensure_dir(f"{results_root}/checkpoints")
    experiment_name = str(cfg.get("experiment_name", "DINO_experiment"))

    transform = MultiCropTransform(
        image_size=image_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        local_crops_number=local_crops_number,
        cfg=cfg
    )

    dataset = CellDataset(root_dir=data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dino_collate,
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
            device=device,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )

        metrics = compute_all_metrics(student, teacher, dataloader, device=device, num_batches=5)
        wandb.log(metrics, step=epoch)
        wandb.log({"loss": avg_loss}, step=epoch)


        if (epoch + 1) % 5 == 0:
            sil = compute_silhouette(student, dataset, device=device, num_samples=500)
            wandb.log({"silhouette": sil}, step=epoch)
            print(f"Silhouette score at epoch {epoch+1}: {sil:.4f}")

        print(f"[{experiment_name}] Epoch {epoch+1}/{epochs} - DINO loss: {avg_loss:.4f}")

    print(f"[{experiment_name}] Train finished. Checkpoints saved to {checkpoints_dir}.")

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
    print(f"[{experiment_name}] Saved final weights to {final_ckpt}.")

    return student, teacher
