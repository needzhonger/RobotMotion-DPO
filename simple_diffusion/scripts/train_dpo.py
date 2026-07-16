#!/usr/bin/env python3
"""Run offline preference generation and DPO on simple_diffusion."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

SIMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIMPLE_ROOT.parent
for path in (REPO_ROOT, SIMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_diffusion_dpo import DPOTrainingConfig, run_dpo_with_reward_suite
from generic_diffusion_dpo.reward_config import build_reward_suite
from simple_diffusion.datasets import MotionWindowDataset, Normalizer
from simple_diffusion.dpo_adapter import SimpleDiffusionDPOAdapter
from simple_diffusion.models import GaussianDiffusion, TemporalDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 simple_diffusion 执行 DPO 后训练")
    parser.add_argument("--checkpoint", type=Path, default=SIMPLE_ROOT / "outputs" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=SIMPLE_ROOT / "outputs_dpo")
    parser.add_argument("--max-conditions", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--lambda-dpo", type=float, default=1.0)
    parser.add_argument("--lambda-sft", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="启用 W&B DPO 指标和采样评估")
    parser.add_argument("--wandb-project", default="robot-motion-dpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-num-conditions", type=int, default=16)
    parser.add_argument("--eval-samples-per-condition", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_adapter(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    diffusion = GaussianDiffusion(
        TemporalDenoiser(**checkpoint["model_config"]),
        num_steps=checkpoint["diffusion_steps"],
    )
    diffusion.load_state_dict(checkpoint["model"])
    adapter = SimpleDiffusionDPOAdapter(
        diffusion,
        motion_dim=checkpoint["model_config"]["motion_dim"],
        sampling_steps=args.sampling_steps,
    ).to(device)
    return adapter, checkpoint


def build_conditions(args: argparse.Namespace, checkpoint: dict):
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    train_files = [args.data_dir / relative for relative in checkpoint["file_splits"]["train"]]
    dataset = MotionWindowDataset(
        train_files,
        normalizer,
        window_size=checkpoint["window_size"],
        stride=checkpoint["stride"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.max_conditions].tolist()
    return [
        {
            "condition": dataset[index]["condition"],
            "mask": dataset[index]["mask"],
        }
        for index in indices
    ]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter, checkpoint = load_adapter(args, device)
    conditions = build_conditions(args, checkpoint)
    rewards = build_reward_suite()
    training = DPOTrainingConfig(
        learning_rate=args.learning_rate,
        beta=args.beta,
        lambda_dpo=args.lambda_dpo,
        lambda_sft=args.lambda_sft,
        max_steps=args.max_steps,
        output_dir=str(args.output_dir / "checkpoints"),
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
        wandb_log_every=args.wandb_log_every,
        eval_every=args.eval_every,
        eval_num_conditions=args.eval_num_conditions,
        eval_samples_per_condition=args.eval_samples_per_condition,
    )

    print(f"device={device} conditions={len(conditions)} rewards={rewards.describe()}")
    run_dpo_with_reward_suite(
        model=adapter,
        conditions=conditions,
        rewards=rewards,
        preference_path=str(args.output_dir / "preferences.pt"),
        training=training,
        candidates_per_condition=args.candidates,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"DPO complete: {args.output_dir}")


if __name__ == "__main__":
    main()
