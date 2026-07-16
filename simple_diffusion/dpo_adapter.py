"""Adapter connecting simple_diffusion to generic_diffusion_dpo."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from generic_diffusion_dpo.interfaces import Condition, DiffusionDPOAdapter

from .models import GaussianDiffusion


class SimpleDiffusionDPOAdapter(DiffusionDPOAdapter):
    """Expose GaussianDiffusion through the generic Diffusion-DPO contract.

    Conditions are dictionaries containing:

    - ``condition``: normalized human pose, shape ``[T, 69]``;
    - ``mask``: valid-frame mask, shape ``[T]``.

    Generated and preference motions stay normalized with shape ``[T, 36]``.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        motion_dim: int = 36,
        sampling_steps: int = 20,
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.motion_dim = motion_dim
        self.sampling_steps = sampling_steps

    @property
    def device(self) -> torch.device:
        return next(self.diffusion.parameters()).device

    @torch.no_grad()
    def generate_motion(self, condition: Condition, num_samples: int) -> Tensor:
        human_condition = torch.as_tensor(condition["condition"], dtype=torch.float32)
        human_condition = human_condition.to(self.device)
        human_condition = human_condition.unsqueeze(0).repeat(num_samples, 1, 1)
        return self.diffusion.sample(
            human_condition,
            motion_dim=self.motion_dim,
            sampling_steps=self.sampling_steps,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(self.diffusion.num_steps, (batch_size,), device=device)

    def add_noise(
        self,
        clean_motion: Tensor,
        timesteps: Tensor,
        noise: Tensor,
        condition: Condition,
    ) -> Tensor:
        return self.diffusion.add_noise(clean_motion, timesteps, noise=noise)

    def denoise(
        self,
        noisy_motion: Tensor,
        timesteps: Tensor,
        condition: Condition,
        shared_state: Mapping[str, Any],
    ) -> Tensor:
        return self.diffusion.denoiser(
            noisy_motion,
            timesteps,
            condition["condition"],
        )

    def training_target(
        self,
        clean_motion: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        # simple_diffusion directly predicts clean x0.
        return clean_motion

    def loss_mask(self, condition: Condition) -> Optional[Tensor]:
        return condition.get("mask")

    def per_sample_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        squared_error = (prediction - target).square()
        if mask is None:
            return squared_error.flatten(1).mean(1)

        weights = mask.to(squared_error.dtype).unsqueeze(-1)
        numerator = (squared_error * weights).sum(dim=(1, 2))
        denominator = (weights.sum(dim=(1, 2)) * squared_error.shape[-1]).clamp_min(1)
        return numerator / denominator
