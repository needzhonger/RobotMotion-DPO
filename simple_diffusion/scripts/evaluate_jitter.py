#!/usr/bin/env python3
"""Compare temporal jitter in pretrained and DPO prediction directories."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


SIMPLE_ROOT = Path(__file__).resolve().parents[1]
MOTION_KEYS = ("output_root_pos", "output_root_rot", "output_dof_pos")


@dataclass(frozen=True)
class JitterResult:
    """Jitter statistics for one motion sequence."""

    mean: float
    squared_sum: float
    element_count: int
    num_frames: int


def load_motion_vector(path: Path) -> np.ndarray:
    """Load exported root position, root rotation and joint pose as [T, 36]."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in MOTION_KEYS if key not in data]
        if missing:
            raise KeyError(f"{path} 缺少字段: {', '.join(missing)}")
        arrays = [np.asarray(data[key], dtype=np.float64) for key in MOTION_KEYS]

    if any(array.ndim != 2 for array in arrays):
        raise ValueError(f"{path} 的动作字段必须全部为二维数组")
    frame_counts = {array.shape[0] for array in arrays}
    if len(frame_counts) != 1:
        raise ValueError(f"{path} 的动作字段帧数不一致")
    motion = np.concatenate(arrays, axis=-1)
    if motion.shape[1] != 36:
        raise ValueError(f"{path} 期望 36 维动作，实际为 {motion.shape[1]} 维")
    if not np.isfinite(motion).all():
        raise ValueError(f"{path} 包含 NaN 或 Inf")
    return motion


def motion_jitter(motion: np.ndarray, motion_mean: np.ndarray, motion_std: np.ndarray) -> JitterResult:
    """Compute the same normalized second-difference metric used by the reward.

    The reward is the negative of ``mean`` returned here, so lower jitter is
    better. Sequences shorter than three frames contain no measurable second
    difference and are rejected instead of silently biasing the average.
    """
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"motion 必须为 [T, 36]，实际为 {motion.shape}")
    if motion.shape[0] < 3:
        raise ValueError("至少需要 3 帧才能计算二阶帧差")
    normalized = (motion - motion_mean) / motion_std
    second_difference = normalized[2:] - 2.0 * normalized[1:-1] + normalized[:-2]
    squared_sum = float(np.square(second_difference).sum(dtype=np.float64))
    element_count = int(second_difference.size)
    return JitterResult(
        mean=squared_sum / element_count,
        squared_sum=squared_sum,
        element_count=element_count,
        num_frames=int(motion.shape[0]),
    )


def load_motion_normalization(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        normalizer = checkpoint["normalizer"]
        mean = np.asarray(normalizer["motion_mean"], dtype=np.float64)
        std = np.asarray(normalizer["motion_std"], dtype=np.float64)
    except KeyError as error:
        raise KeyError(f"{checkpoint_path} 缺少 normalizer/{error.args[0]}") from error
    if mean.shape != (36,) or std.shape != (36,):
        raise ValueError(f"checkpoint 中 motion_mean/motion_std 必须为 [36]")
    if np.any(std <= 0):
        raise ValueError("checkpoint 中 motion_std 必须全部大于 0")
    return mean, std


def prediction_files(directory: Path) -> dict[Path, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"预测目录不存在: {directory}")
    files = {path.relative_to(directory): path for path in sorted(directory.rglob("*.npz"))}
    if not files:
        raise FileNotFoundError(f"预测目录中没有 .npz 文件: {directory}")
    return files


def aggregate(results: Iterable[JitterResult]) -> tuple[float, float]:
    results = list(results)
    clip_mean = float(np.mean([result.mean for result in results]))
    weighted_mean = sum(result.squared_sum for result in results) / sum(
        result.element_count for result in results
    )
    return clip_mean, weighted_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较普通模型和 DPO 模型预测动作的平均帧间抖动（越低越好）"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=SIMPLE_ROOT / "outputs" / "vector_pairs_pred",
    )
    parser.add_argument(
        "--outputs-dpo-dir",
        type=Path,
        default=SIMPLE_ROOT / "outputs_dpo" / "vector_pairs_pred",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=SIMPLE_ROOT / "outputs" / "best.pt",
        help="提供与 DPO reward 相同的动作归一化统计",
    )
    parser.add_argument("--csv", type=Path, default=None, help="可选：保存逐文件结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_mean, motion_std = load_motion_normalization(args.base_checkpoint)
    base_files = prediction_files(args.outputs_dir)
    dpo_files = prediction_files(args.outputs_dpo_dir)
    matched = sorted(base_files.keys() & dpo_files.keys())
    if not matched:
        raise RuntimeError("两个预测目录中没有相同相对路径的 .npz 文件")

    rows = []
    base_results, dpo_results = [], []
    for relative_path in matched:
        base_motion = load_motion_vector(base_files[relative_path])
        dpo_motion = load_motion_vector(dpo_files[relative_path])
        if base_motion.shape != dpo_motion.shape:
            raise ValueError(
                f"{relative_path} 两组预测形状不一致: {base_motion.shape} vs {dpo_motion.shape}"
            )
        base = motion_jitter(base_motion, motion_mean, motion_std)
        dpo = motion_jitter(dpo_motion, motion_mean, motion_std)
        base_results.append(base)
        dpo_results.append(dpo)
        improvement = base.mean - dpo.mean
        improvement_percent = improvement / base.mean * 100.0 if base.mean != 0 else float("nan")
        rows.append(
            {
                "file": str(relative_path),
                "frames": base.num_frames,
                "outputs_jitter": base.mean,
                "outputs_dpo_jitter": dpo.mean,
                "improvement": improvement,
                "improvement_percent": improvement_percent,
            }
        )

    base_clip, base_weighted = aggregate(base_results)
    dpo_clip, dpo_weighted = aggregate(dpo_results)
    improvement = base_weighted - dpo_weighted
    improvement_percent = improvement / base_weighted * 100.0 if base_weighted != 0 else float("nan")

    print("帧间抖动评估（二阶帧差均方，归一化动作空间，越低越好）")
    print(f"匹配文件数:              {len(matched)}")
    print(f"仅 outputs 中存在:       {len(base_files.keys() - dpo_files.keys())}")
    print(f"仅 outputs_dpo 中存在:   {len(dpo_files.keys() - base_files.keys())}")
    print(f"outputs 逐片段平均:       {base_clip:.10g}")
    print(f"outputs_dpo 逐片段平均:   {dpo_clip:.10g}")
    print(f"outputs 总体加权平均:     {base_weighted:.10g}")
    print(f"outputs_dpo 总体加权平均: {dpo_weighted:.10g}")
    print(f"DPO 抖动减少量:           {improvement:.10g}")
    print(f"DPO 相对改善:             {improvement_percent:.4f}%")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"逐文件结果:               {args.csv}")


if __name__ == "__main__":
    main()
