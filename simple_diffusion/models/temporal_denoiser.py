"""用于动作序列的轻量级一维卷积去噪网络。"""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """将离散扩散步编码为正余弦向量。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, dilation: int) -> None:
        super().__init__()
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_feature = self.time_projection(time_embedding).unsqueeze(-1)
        return x + self.block(x + time_feature)


class TemporalDenoiser(nn.Module):
    """根据人体条件和带噪机器人动作直接预测干净动作 x_0。

    输入和输出均采用 ``(batch, time, feature)`` 排布。
    """

    def __init__(
        self,
        condition_dim: int = 69,
        motion_dim: int = 36,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim 必须能被 8 整除，以供 GroupNorm 使用")

        self.condition_dim = condition_dim
        self.motion_dim = motion_dim
        self.input_projection = nn.Conv1d(condition_dim + motion_dim, hidden_dim, 1)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, time_dim, dilation=2 ** (i % 4))
            for i in range(num_blocks)
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, motion_dim, 1),
        )

    def forward(
        self, noisy_motion: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if noisy_motion.shape[:2] != condition.shape[:2]:
            raise ValueError("noisy_motion 与 condition 的 batch/time 维度必须一致")
        feature = torch.cat((noisy_motion, condition), dim=-1).transpose(1, 2)
        feature = self.input_projection(feature)
        time_embedding = self.time_mlp(timesteps)
        for block in self.blocks:
            feature = block(feature, time_embedding)
        return self.output_projection(feature).transpose(1, 2)
