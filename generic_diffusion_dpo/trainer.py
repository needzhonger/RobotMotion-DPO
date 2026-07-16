from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import Tensor

from .interfaces import DiffusionDPOAdapter
from .loss import DiffusionDPOLoss


@dataclass
class DPOTrainingConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    beta: float = 20.0
    lambda_dpo: float = 1.0
    lambda_sft: float = 2.0
    max_steps: int = 5_000
    grad_clip: float = 1.0
    log_every: int = 20
    save_every: int = 200
    output_dir: str = "dpo_output"


class DPOTrainer:
    def __init__(
        self,
        policy: DiffusionDPOAdapter,
        config: DPOTrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)

        # The reference is an immutable snapshot of the initial policy.
        self.reference = copy.deepcopy(policy).to(self.device)
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

        self.objective = DiffusionDPOLoss(
            beta=config.beta,
            lambda_dpo=config.lambda_dpo,
            lambda_sft=config.lambda_sft,
        )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.policy.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        winner = batch["winner"].to(self.device)
        loser = batch["loser"].to(self.device)
        pair_batch_size = winner.shape[0]

        condition = self.policy.collate_conditions(batch["conditions"])
        condition = self.policy.move_condition_to_device(condition, self.device)
        doubled_condition = self.policy.repeat_condition(condition, repeats=2)

        timesteps = self.policy.sample_timesteps(pair_batch_size, self.device)
        timestep_weights = self.policy.timestep_weights(timesteps)
        doubled_timesteps = torch.cat([timesteps, timesteps], dim=0)

        # Same noise for the winner and loser of each pair.
        pair_noise = torch.randn_like(winner)
        clean_motion = torch.cat([winner, loser], dim=0)
        doubled_noise = torch.cat([pair_noise, pair_noise], dim=0)
        noisy_motion = self.policy.add_noise(
            clean_motion,
            doubled_timesteps,
            doubled_noise,
            doubled_condition,
        )
        target = self.policy.training_target(
            clean_motion,
            doubled_noise,
            doubled_timesteps,
        )

        # The adapter must encode CFG/dropout masks in this shared state.
        shared_state = self.policy.make_shared_forward_state(
            pair_batch_size,
            self.device,
        )

        # Match PhysMoDPO: policy/reference have the same train/eval mode and
        # consume identical internal randomness (e.g. Transformer dropout).
        self.reference.train(self.policy.training)
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        policy_prediction = self.policy.denoise(
            noisy_motion,
            doubled_timesteps,
            doubled_condition,
            shared_state,
        )
        cpu_rng_after = torch.get_rng_state()
        cuda_rng_after = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        policy_losses = self.policy.per_sample_loss(policy_prediction, target, mask=None)
        policy_winner_loss, policy_loser_loss = policy_losses.chunk(2)

        with torch.no_grad():
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, self.device)
            reference_prediction = self.reference.denoise(
                noisy_motion,
                doubled_timesteps,
                doubled_condition,
                shared_state,
            )
            reference_losses = self.reference.per_sample_loss(
                reference_prediction,
                target,
                mask=None,
            )
            reference_winner_loss, reference_loser_loss = reference_losses.chunk(2)

        # Advance the global RNG as if only the policy forward had happened.
        torch.set_rng_state(cpu_rng_after)
        if cuda_rng_after is not None:
            torch.cuda.set_rng_state(cuda_rng_after, self.device)

        terms = self.objective(
            policy_winner_loss,
            policy_loser_loss,
            reference_winner_loss,
            reference_loser_loss,
        )
        terms["weighted_loss"] = (terms["loss"] * timestep_weights).mean()
        return terms

    def train(self, dataloader: Iterable[Dict[str, Any]]) -> DiffusionDPOAdapter:
        self.policy.train()
        iterator = iter(dataloader)

        for step in range(1, self.config.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            self.optimizer.zero_grad(set_to_none=True)
            terms = self.compute_loss(batch)
            terms["weighted_loss"].backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.grad_clip,
                )
            self.optimizer.step()

            if step % self.config.log_every == 0 or step == 1:
                print(
                    f"step={step} "
                    f"loss={terms['weighted_loss'].item():.5f} "
                    f"dpo={terms['dpo_loss'].mean().item():.5f} "
                    f"sft={terms['sft_loss'].mean().item():.5f} "
                    f"pref_acc={terms['preference_accuracy'].mean().item():.3f}"
                )

            if step % self.config.save_every == 0 or step == self.config.max_steps:
                self.save(step)

        return self.policy

    def save(self, step: int) -> None:
        path = Path(self.config.output_dir) / f"policy_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )
