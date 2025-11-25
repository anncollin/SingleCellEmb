from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


#######################################################################################################
# DINO LOSS MODULE
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    out_dim : int
#        Number of prototypes (output dimension of the head).
#
#    warmup_teacher_temp : float
#        Initial temperature for the teacher softmax.
#
#    teacher_temp : float
#        Final temperature for the teacher softmax (after warmup).
#
#    warmup_teacher_temp_epochs : int
#        Number of epochs over which to linearly increase teacher temperature.
#
#    nepochs : int
#        Total number of train epochs (for scheduling).
#
#    student_temp : float
#        Softmax temperature for student outputs.
#
#    center_momentum : float
#        Momentum used in the update of the center buffer.
#
#    Returns:
#    ---------------------------
#    loss_value : Tensor
#        Scalar DINO loss value.
#######################################################################################################
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = torch.full((nepochs,), teacher_temp)
        if warmup_teacher_temp_epochs > 0:
            self.teacher_temp_schedule[:warmup_teacher_temp_epochs] = torch.linspace(
                warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
            )

    def forward(
        self,
        student_output: List[torch.Tensor],
        teacher_output: List[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        student_out = torch.cat(student_output, dim=0)
        teacher_out = torch.cat(teacher_output, dim=0)

        student_out = student_out / self.student_temp
        student_out = torch.log_softmax(student_out, dim=-1)

        temp = self.teacher_temp_schedule[epoch].to(teacher_out.device)
        teacher_out = (teacher_out - self.center) / temp
        teacher_out = torch.softmax(teacher_out, dim=-1).detach()

        n_teacher = len(teacher_output)
        n_student = len(student_output)
        batch_size = teacher_output[0].shape[0]

        loss = 0.0
        n_loss_terms = 0
        for iq in range(n_teacher):
            for v in range(n_student):
                if v == iq:
                    continue
                t = teacher_out[iq * batch_size : (iq + 1) * batch_size]
                s = student_out[v * batch_size : (v + 1) * batch_size]
                loss += torch.sum(-t * s, dim=-1).mean()
                n_loss_terms += 1
        loss /= n_loss_terms

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
