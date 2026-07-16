"""Tiny executable example proving the full generic pipeline is connected."""

import tempfile

import torch
from torch import nn

from .interfaces import DiffusionDPOAdapter
from .pipeline import run_dpo_with_reward_suite
from .preference import RewardSuite, RewardTerm
from .trainer import DPOTrainingConfig


class ToyAdapter(DiffusionDPOAdapter):
    def __init__(self, dimensions: int = 4, diffusion_steps: int = 100):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.model = nn.Sequential(
            nn.Linear(2 * dimensions + 1, 32),
            nn.SiLU(),
            nn.Linear(32, dimensions),
        )

    @torch.no_grad()
    def generate_motion(self, condition, num_samples):
        # A real adapter would call the pretrained model's generate_motion here.
        return condition.unsqueeze(0) + 0.5 * torch.randn(num_samples, *condition.shape)

    def sample_timesteps(self, batch_size, device):
        return torch.randint(self.diffusion_steps, (batch_size,), device=device)

    def add_noise(self, clean_motion, timesteps, noise, condition):
        alpha = 1.0 - (timesteps.float() + 1.0) / (self.diffusion_steps + 1.0)
        alpha = alpha.view(-1, *([1] * (clean_motion.ndim - 1)))
        return alpha.sqrt() * clean_motion + (1.0 - alpha).sqrt() * noise

    def denoise(self, noisy_motion, timesteps, condition, shared_state):
        time = (timesteps.float() / self.diffusion_steps).unsqueeze(1)
        return self.model(torch.cat([noisy_motion, condition, time], dim=1))

    def training_target(self, clean_motion, noise, timesteps):
        return clean_motion

    def per_sample_loss(self, prediction, target, mask=None):
        return (prediction - target).square().flatten(1).mean(1)


def main():
    torch.manual_seed(0)
    conditions = [torch.randn(4) for _ in range(16)]

    # In a real project, define this function in reward_functions.py and import
    # it here. Keeping it inline is only convenient for this executable toy.
    def quality_reward(motion, condition):
        return -(motion - condition).square().mean()

    rewards = RewardSuite(
        [
            RewardTerm(
                "quality",
                evaluator=quality_reward,
                weight=1.0,
                higher_is_better=True,
            )
        ],
        min_score_gap=1e-4,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        policy = run_dpo_with_reward_suite(
            model=ToyAdapter(),
            conditions=conditions,
            rewards=rewards,
            preference_path=f"{output_dir}/preferences.pt",
            candidates_per_condition=4,
            batch_size=4,
            training=DPOTrainingConfig(
                lambda_dpo=1.0,
                lambda_sft=2.0,
                max_steps=2,
                log_every=1,
                save_every=2,
                output_dir=f"{output_dir}/checkpoints",
            ),
            device=torch.device("cpu"),
        )
        assert isinstance(policy, ToyAdapter)


if __name__ == "__main__":
    main()
