#!/usr/bin/env python3
"""在测试文件上采样、计算误差并保存预测动作。"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Normalizer, load_motion
from models import GaussianDiffusion, TemporalDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 x0-Diffusion 并导出机器人动作")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "predictions")
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_windows(length: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    last_start = max(length - window_size, 0)
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(start, min(window_size, length - start)) for start in starts]


@torch.no_grad()
def predict_sequence(
    model: GaussianDiffusion,
    condition: np.ndarray,
    normalizer: Normalizer,
    window_size: int,
    stride: int,
    sampling_steps: int,
    device: torch.device,
) -> np.ndarray:
    normalized_condition = normalizer.normalize_condition(condition).astype(np.float32)
    prediction_sum = np.zeros((len(condition), 36), dtype=np.float32)
    prediction_count = np.zeros((len(condition), 1), dtype=np.float32)

    for start, valid_length in make_windows(len(condition), window_size, stride):
        window = normalized_condition[start : start + window_size]
        if valid_length < window_size:
            window = np.concatenate(
                (window, np.repeat(window[-1:], window_size - valid_length, axis=0))
            )
        condition_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        predicted = model.sample(condition_tensor, motion_dim=36, sampling_steps=sampling_steps)
        predicted = predicted[0, :valid_length].cpu().numpy()
        prediction_sum[start : start + valid_length] += predicted
        prediction_count[start : start + valid_length] += 1

    normalized_prediction = prediction_sum / np.maximum(prediction_count, 1)
    prediction = normalizer.denormalize_motion(normalized_prediction)
    # 网络输出不严格满足单位模约束，导出前重新归一化根四元数。
    quaternion = prediction[:, 3:7]
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    prediction[:, 3:7] = quaternion
    return prediction


def quaternion_angle_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """q 与 -q 等价，返回平均旋转角误差（度）。"""
    dot = np.abs(np.sum(predicted * target, axis=-1))
    return float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))).mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stride"] > checkpoint["window_size"]:
        raise ValueError("checkpoint 中的 stride 大于 window_size，无法无缝拼接序列")
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    model = GaussianDiffusion(
        TemporalDenoiser(**checkpoint["model_config"]),
        num_steps=checkpoint["diffusion_steps"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    test_files = [args.data_dir / relative for relative in checkpoint["file_splits"]["test"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    position_errors, rotation_errors, dof_errors = [], [], []
    for index, path in enumerate(test_files, start=1):
        condition, target, metadata = load_motion(path)
        prediction = predict_sequence(
            model,
            condition,
            normalizer,
            checkpoint["window_size"],
            checkpoint["stride"],
            args.sampling_steps,
            device,
        )
        position_errors.append(float(np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1).mean()))
        rotation_errors.append(quaternion_angle_error(prediction[:, 3:7], target[:, 3:7]))
        dof_errors.append(float(np.abs(prediction[:, 7:] - target[:, 7:]).mean()))

        output_path = args.output_dir / path.relative_to(args.data_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            output_root_pos=prediction[:, :3],
            output_root_rot=prediction[:, 3:7],
            output_dof_pos=prediction[:, 7:],
            fps=metadata["fps"],
            num_frames=len(prediction),
            motor_names=metadata["motor_names"],
            robot=metadata["robot"],
            source_file=str(path.relative_to(args.data_dir)),
        )
        print(f"[{index:02d}/{len(test_files)}] {path.relative_to(args.data_dir)}")

    print("\n测试结果（所有测试片段的逐片段均值）")
    print(f"根位置误差:       {np.mean(position_errors):.6f} m")
    print(f"根旋转角误差:     {np.mean(rotation_errors):.3f} deg")
    print(f"关节角绝对误差:   {np.mean(dof_errors):.6f} rad")
    print(f"预测文件已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
