"""NPZ 动捕数据的划分、归一化和窗口数据集。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


CONDITION_KEYS = ("input_pose_body", "input_root_orient", "input_trans")
MOTION_KEYS = ("output_root_pos", "output_root_rot", "output_dof_pos")


def _continuous_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """归一化四元数，并消除 q/-q 表示同一旋转造成的符号跳变。"""
    quaternion = quaternion.copy()
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion /= np.maximum(norm, 1e-8)
    if quaternion[0, 0] < 0:
        quaternion[0] *= -1
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1
    return quaternion


def load_motion(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """读取一个片段，返回人体条件、机器人动作和元数据。"""
    with np.load(path, allow_pickle=False) as data:
        condition = np.concatenate([data[key] for key in CONDITION_KEYS], axis=-1).astype(np.float32)
        root_rotation = _continuous_quaternion(data["output_root_rot"].astype(np.float32))
        motion = np.concatenate(
            [data["output_root_pos"], root_rotation, data["output_dof_pos"]], axis=-1
        ).astype(np.float32)
        metadata = {
            "fps": np.asarray(data["fps"]),
            "motor_names": np.asarray(data["motor_names"]),
            "robot": np.asarray(data["robot"]),
        }
    return condition, motion, metadata


def split_files(
    data_dir: Path, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
) -> Dict[str, List[Path]]:
    """按文件固定随机划分，确保一个完整片段只属于一个集合。"""
    files = sorted(data_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有找到 .npz 文件")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio/test_ratio 必须非负，且两者之和小于 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files))
    test_count = int(round(len(files) * test_ratio))
    val_count = int(round(len(files) * val_ratio))
    test_indices = indices[:test_count]
    val_indices = indices[test_count : test_count + val_count]
    train_indices = indices[test_count + val_count :]
    return {
        "train": [files[i] for i in train_indices],
        "val": [files[i] for i in val_indices],
        "test": [files[i] for i in test_indices],
    }


@dataclass
class Normalizer:
    condition_mean: np.ndarray
    condition_std: np.ndarray
    motion_mean: np.ndarray
    motion_std: np.ndarray

    @classmethod
    def fit(cls, files: Sequence[Path]) -> "Normalizer":
        """只使用训练集帧统计均值和标准差。"""
        conditions, motions = [], []
        for path in files:
            condition, motion, _ = load_motion(path)
            conditions.append(condition)
            motions.append(motion)
        condition_array = np.concatenate(conditions)
        motion_array = np.concatenate(motions)
        return cls(
            condition_array.mean(0),
            np.maximum(condition_array.std(0), 1e-6),
            motion_array.mean(0),
            np.maximum(motion_array.std(0), 1e-6),
        )

    def normalize_condition(self, value: np.ndarray) -> np.ndarray:
        return (value - self.condition_mean) / self.condition_std

    def normalize_motion(self, value: np.ndarray) -> np.ndarray:
        return (value - self.motion_mean) / self.motion_std

    def denormalize_motion(self, value: np.ndarray) -> np.ndarray:
        return value * self.motion_std + self.motion_mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "condition_mean": self.condition_mean,
            "condition_std": self.condition_std,
            "motion_mean": self.motion_mean,
            "motion_std": self.motion_std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "Normalizer":
        return cls(**{key: np.asarray(value) for key, value in state.items()})


class MotionWindowDataset(Dataset):
    """把不同长度片段切成定长窗口；短片段采用末帧补齐。"""

    def __init__(
        self, files: Sequence[Path], normalizer: Normalizer, window_size: int = 64, stride: int = 32
    ) -> None:
        self.window_size = window_size
        self.normalizer = normalizer
        self.sequences = []
        self.windows = []
        for sequence_index, path in enumerate(files):
            condition, motion, _ = load_motion(path)
            condition = normalizer.normalize_condition(condition).astype(np.float32)
            motion = normalizer.normalize_motion(motion).astype(np.float32)
            self.sequences.append((condition, motion))
            last_start = max(len(condition) - window_size, 0)
            starts = list(range(0, last_start + 1, stride))
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            self.windows.extend((sequence_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence_index, start = self.windows[index]
        condition, motion = self.sequences[sequence_index]
        valid_length = min(self.window_size, len(condition) - start)
        condition_window = condition[start : start + self.window_size]
        motion_window = motion[start : start + self.window_size]
        if valid_length < self.window_size:
            pad_length = self.window_size - valid_length
            condition_window = np.concatenate(
                (condition_window, np.repeat(condition_window[-1:], pad_length, axis=0))
            )
            motion_window = np.concatenate(
                (motion_window, np.repeat(motion_window[-1:], pad_length, axis=0))
            )
        mask = np.arange(self.window_size) < valid_length
        return {
            "condition": torch.from_numpy(condition_window.copy()),
            "motion": torch.from_numpy(motion_window.copy()),
            "mask": torch.from_numpy(mask),
        }
