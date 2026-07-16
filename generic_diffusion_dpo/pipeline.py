from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .interfaces import Condition, DiffusionDPOAdapter
from .preference import (
    PreferenceDataset,
    PreferencePairSelector,
    RewardEvaluator,
    RewardSuite,
    build_preference_data,
    collate_preference_batch,
)
from .trainer import DPOTrainer, DPOTrainingConfig


def run_dpo_post_training(
    model: DiffusionDPOAdapter,
    conditions: Iterable[Condition],
    reward_evaluator: RewardEvaluator,
    pair_selector: PreferencePairSelector,
    preference_path: str,
    training: DPOTrainingConfig,
    candidates_per_condition: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> DiffusionDPOAdapter:
    """End-to-end offline preference generation followed by Diffusion-DPO."""
    build_preference_data(
        model=model,
        conditions=conditions,
        reward_evaluator=reward_evaluator,
        pair_selector=pair_selector,
        output_path=preference_path,
        candidates_per_condition=candidates_per_condition,
    )

    dataset = PreferenceDataset(preference_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_preference_batch,
    )
    trainer = DPOTrainer(model, training, device=device)
    return trainer.train(dataloader)


def run_dpo_with_reward_suite(
    model: DiffusionDPOAdapter,
    conditions: Iterable[Condition],
    rewards: RewardSuite,
    preference_path: str,
    training: DPOTrainingConfig,
    candidates_per_condition: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> DiffusionDPOAdapter:
    """Convenience entry point with all reward choices in one RewardSuite.

    Define concrete functions such as `control_reward` in reward_functions.py,
    and assemble/enable them in reward_config.py::build_reward_suite(). Do not
    put concrete reward implementations in this generic pipeline module.
    """
    return run_dpo_post_training(
        model=model,
        conditions=conditions,
        reward_evaluator=rewards,
        pair_selector=rewards.pair_selector(),
        preference_path=preference_path,
        training=training,
        candidates_per_condition=candidates_per_condition,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
