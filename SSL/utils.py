import os
from typing import Dict

import torch


#######################################################################################################
# EMA UPDATE FOR TEACHER PARAMETERS
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    student : nn.Module
#        Student model.
#
#    teacher : nn.Module
#        Teacher model to be updated in-place.
#
#    momentum : float
#        Exponential moving average momentum coefficient.
#
#    Returns:
#    ---------------------------
#    None
#######################################################################################################
@torch.no_grad()
def update_teacher(student, teacher, momentum: float) -> None:
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data * (1.0 - momentum))


#######################################################################################################
# ENSURE DIRECTORY EXISTS
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    path : str
#        Directory path to create if it does not exist.
#
#    Returns:
#    ---------------------------
#    path : str
#        The same path, for convenience.
#######################################################################################################
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


#######################################################################################################
# SAVE CHECKPOINT
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    checkpoint_path : str
#        Path to the checkpoint file to write.
#
#    state : Dict
#        Dictionary containing model and optimizer state.
#
#    Returns:
#    ---------------------------
#    None
#######################################################################################################
def save_checkpoint(checkpoint_path: str, state: Dict) -> None:
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save(state, checkpoint_path)
