#!/usr/bin/env python3
"""Predict robot vectors for every NPZ file while mirroring data directories."""

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

from simple_diffusion.datasets import Normalizer, load_motion
from simple_diffusion.dpo_adapter import SimpleDiffusionDPOAdapter
from simple_diffusion.models import GaussianDiffusion, TemporalDenoiser
from simple_diffusion.scripts.test import predict_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用基础或 DPO 后训练的 simple_diffusion 批量预测机器人动作"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SIMPLE_ROOT / "outputs" / "best.pt",
        help="用于预测的基础 best.pt 或 DPO policy_step_*.pt",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=SIMPLE_ROOT / "outputs" / "best.pt",
        help="提供模型结构与归一化统计；使用 DPO checkpoint 时仍需此文件",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "输出目录；默认根据 checkpoint 类型自动选择 outputs/vector_pairs_pred "
            "或 outputs_dpo/vector_pairs_pred"
        ),
    )
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-files", type=int, default=None, help="仅用于快速测试")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    checkpoint_path: Path,
    base_checkpoint_path: Path,
    device: torch.device,
) -> tuple[GaussianDiffusion, Normalizer, dict]:
    """Load either a pretraining checkpoint or generic DPO adapter checkpoint."""
    base = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    model = GaussianDiffusion(
        TemporalDenoiser(**base["model_config"]),
        num_steps=base["diffusion_steps"],
    )
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in selected:
        model.load_state_dict(selected["model"])
        checkpoint_kind = "pretrained"
    elif "model_state_dict" in selected:
        adapter = SimpleDiffusionDPOAdapter(
            model,
            motion_dim=base["model_config"]["motion_dim"],
        )
        adapter.load_state_dict(selected["model_state_dict"])
        model = adapter.diffusion
        checkpoint_kind = "dpo"
    else:
        raise ValueError(
            f"{checkpoint_path} 既不是 simple_diffusion checkpoint，也不是 DPO checkpoint"
        )

    model.to(device).eval()
    normalizer = Normalizer.from_state_dict(base["normalizer"])
    metadata = {
        "kind": checkpoint_kind,
        "window_size": base["window_size"],
        "stride": base["stride"],
    }
    return model, normalizer, metadata


def output_path_for(source_path: Path, data_dir: Path, output_dir: Path) -> Path:
    relative = source_path.relative_to(data_dir)
    return output_dir / relative.parent / f"{relative.stem}_pred.npz"


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {args.data_dir}")
    if args.sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    model, normalizer, model_info = load_model(
        args.checkpoint,
        args.base_checkpoint,
        device,
    )
    if args.output_dir is None:
        checkpoint_output_root = "outputs_dpo" if model_info["kind"] == "dpo" else "outputs"
        args.output_dir = SIMPLE_ROOT / checkpoint_output_root / "vector_pairs_pred"

    files = sorted(args.data_dir.rglob("*.npz"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"no NPZ files found under {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"checkpoint={args.checkpoint} kind={model_info['kind']} device={device} "
        f"files={len(files)} output={args.output_dir}"
    )
    written = 0
    skipped = 0
    for index, source_path in enumerate(files, start=1):
        output_path = output_path_for(source_path, args.data_dir, args.output_dir)
        if args.skip_existing and output_path.exists():
            skipped += 1
            print(f"[{index:04d}/{len(files):04d}] skip {source_path.relative_to(args.data_dir)}")
            continue

        condition, _, source_metadata = load_motion(source_path)
        prediction = predict_sequence(
            model=model,
            condition=condition,
            normalizer=normalizer,
            window_size=model_info["window_size"],
            stride=model_info["stride"],
            sampling_steps=args.sampling_steps,
            device=device,
        ).astype(np.float32, copy=False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep this schema intentionally identical to the provided examples.
        np.savez_compressed(
            output_path,
            output_root_pos=prediction[:, :3],
            output_root_rot=prediction[:, 3:7],
            output_dof_pos=prediction[:, 7:],
            robot=source_metadata["robot"],
            fps=source_metadata["fps"],
        )
        written += 1
        print(
            f"[{index:04d}/{len(files):04d}] "
            f"{source_path.relative_to(args.data_dir)} -> "
            f"{output_path.relative_to(args.output_dir)}"
        )

    print(f"done: written={written} skipped={skipped} output={args.output_dir}")


if __name__ == "__main__":
    main()
