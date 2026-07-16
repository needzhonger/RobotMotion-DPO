from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DiffusionDPOLoss(nn.Module):
    """DPO over four per-sample diffusion denoising losses.

    Diffusion log probability is approximated as -0.5 * denoising_loss.
    Reference losses are detached defensively. All returned losses are [B] so
    that the trainer can apply timestep importance weights before reduction.
    """

    def __init__(
        self,
        beta: float = 20.0,
        lambda_dpo: float = 1.0,
        lambda_sft: float = 2.0,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.lambda_dpo = lambda_dpo
        self.lambda_sft = lambda_sft

    def forward(
        self,
        policy_winner_loss: Tensor,
        policy_loser_loss: Tensor,
        reference_winner_loss: Tensor,
        reference_loser_loss: Tensor,
    ) -> Dict[str, Tensor]:
        shapes = {
            tuple(policy_winner_loss.shape),
            tuple(policy_loser_loss.shape),
            tuple(reference_winner_loss.shape),
            tuple(reference_loser_loss.shape),
        }
        if len(shapes) != 1 or policy_winner_loss.ndim != 1:
            raise ValueError("all inputs must have the same [B] shape")

        reference_winner_loss = reference_winner_loss.detach()
        reference_loser_loss = reference_loser_loss.detach()

        policy_logratio = -0.5 * (policy_winner_loss - policy_loser_loss)
        reference_logratio = -0.5 * (
            reference_winner_loss - reference_loser_loss
        )
        logits = self.beta * (policy_logratio - reference_logratio)

        dpo_loss = -F.logsigmoid(logits)
        sft_loss = policy_winner_loss
        total_loss = self.lambda_dpo * dpo_loss + self.lambda_sft * sft_loss

        return {
            "loss": total_loss,
            "dpo_loss": dpo_loss,
            "sft_loss": sft_loss,
            "logits": logits,
            "policy_logratio": policy_logratio,
            "reference_logratio": reference_logratio,
            "preference_accuracy": (logits > 0).float(),
        }
