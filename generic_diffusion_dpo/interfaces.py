from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn


Condition = Any


def _collate(values: List[Any]) -> Any:
    """Default collation for tensors, dictionaries, scalars and strings."""
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values)
    if isinstance(first, Mapping):
        return {key: _collate([value[key] for value in values]) for key in first}
    if isinstance(first, (float, int)):
        return torch.as_tensor(values)
    return values


def _repeat_batch(value: Any, repeats: int) -> Any:
    """Repeat a collated batch in block order: [batch, batch, ...]."""
    if torch.is_tensor(value):
        return value.repeat((repeats,) + (1,) * (value.ndim - 1))
    if isinstance(value, Mapping):
        return {key: _repeat_batch(item, repeats) for key, item in value.items()}
    if isinstance(value, list):
        return value * repeats
    if isinstance(value, tuple):
        return value * repeats
    return value


class DiffusionDPOAdapter(nn.Module, ABC):
    """The only model-specific layer required by the generic DPO pipeline.

    The adapter must own the trainable diffusion model as a registered submodule.
    `generate_motion` is used offline. The remaining methods form a differentiable
    one-step denoising interface used during DPO training.
    """

    @abstractmethod
    def generate_motion(self, condition: Condition, num_samples: int) -> Tensor:
        """Return candidate clean motions with shape [K, ...]."""

    @abstractmethod
    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Return integer diffusion timesteps with shape [B]."""

    @abstractmethod
    def add_noise(
        self,
        clean_motion: Tensor,
        timesteps: Tensor,
        noise: Tensor,
        condition: Condition,
    ) -> Tensor:
        """Implement q(x_t | x_0), returning a tensor shaped like clean_motion."""

    @abstractmethod
    def denoise(
        self,
        noisy_motion: Tensor,
        timesteps: Tensor,
        condition: Condition,
        shared_state: Mapping[str, Any],
    ) -> Tensor:
        """Differentiable model forward; return x0/epsilon/v prediction."""

    @abstractmethod
    def training_target(
        self,
        clean_motion: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Return the target matching the model prediction type."""

    @abstractmethod
    def per_sample_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Return an unreduced denoising loss with shape [B]."""

    def collate_conditions(self, conditions: List[Condition]) -> Condition:
        return _collate(conditions)

    def repeat_condition(self, condition: Condition, repeats: int) -> Condition:
        return _repeat_batch(condition, repeats)

    def move_condition_to_device(self, condition: Condition, device: torch.device) -> Condition:
        if torch.is_tensor(condition):
            return condition.to(device)
        if isinstance(condition, Mapping):
            return {
                key: self.move_condition_to_device(value, device)
                for key, value in condition.items()
            }
        return condition

    def make_shared_forward_state(
        self,
        pair_batch_size: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Create randomness shared by winner/loser and policy/reference.

        Override this for classifier-free dropout or other stochastic model
        layers. Returned tensors should already have leading size 2*B.
        """
        return {}

    def timestep_weights(self, timesteps: Tensor) -> Tensor:
        """Optional importance weights for non-uniform timestep sampling."""
        return torch.ones_like(timesteps, dtype=torch.float32)

    def loss_mask(self, condition: Condition) -> Optional[Tensor]:
        """Return an optional mask for per-sample denoising loss.

        The returned mask must already match the doubled winner/loser batch.
        Models without padded sequences can keep the default ``None``.
        """
        return None
