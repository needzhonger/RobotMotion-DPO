"""Small, model-agnostic toolkit for offline Diffusion-DPO post-training."""

from .interfaces import DiffusionDPOAdapter
from .loss import DiffusionDPOLoss
from .pipeline import run_dpo_post_training, run_dpo_with_reward_suite
from .preference import (
    MetricSpec,
    PreferenceDataset,
    PreferencePairSelector,
    RewardSuite,
    RewardTerm,
    build_preference_data,
)
from .trainer import DPOTrainer, DPOTrainingConfig

__all__ = [
    "DiffusionDPOAdapter",
    "DiffusionDPOLoss",
    "MetricSpec",
    "PreferenceDataset",
    "PreferencePairSelector",
    "RewardSuite",
    "RewardTerm",
    "build_preference_data",
    "DPOTrainer",
    "DPOTrainingConfig",
    "run_dpo_post_training",
    "run_dpo_with_reward_suite",
]
