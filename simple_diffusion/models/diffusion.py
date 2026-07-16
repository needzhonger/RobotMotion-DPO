"""直接预测 x_0 的高斯 Diffusion。"""

from typing import Optional

import torch
from torch import nn


def cosine_beta_schedule(num_steps: int, offset: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine 噪声日程。"""
    x = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / num_steps + offset) / (1 + offset)) * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    """按 batch 中的时间步取系数，并扩展到目标张量维数。"""
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    """训练时以 MSE(x_0_pred, x_0) 为目标的条件 Diffusion。"""

    def __init__(self, denoiser: nn.Module, num_steps: int = 100) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps

        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1), alpha_bars[:-1]))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1 - alpha_bars).sqrt())
        posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer(
            "posterior_mean_coef_x0", betas * alpha_bars_prev.sqrt() / (1 - alpha_bars)
        )
        self.register_buffer(
            "posterior_mean_coef_xt", (1 - alpha_bars_prev) * alphas.sqrt() / (1 - alpha_bars)
        )

    def add_noise(
        self, clean_motion: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_motion) if noise is None else noise
        return (
            extract(self.sqrt_alpha_bars, timesteps, clean_motion.ndim) * clean_motion
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_motion.ndim) * noise
        )

    def training_loss(
        self, clean_motion: torch.Tensor, condition: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = clean_motion.shape[0]
        timesteps = torch.randint(self.num_steps, (batch_size,), device=clean_motion.device)
        noisy_motion = self.add_noise(clean_motion, timesteps)
        predicted_x0 = self.denoiser(noisy_motion, timesteps, condition)
        squared_error = (predicted_x0 - clean_motion).square()
        if mask is None:
            return squared_error.mean()
        weights = mask.unsqueeze(-1).to(squared_error.dtype)
        return (squared_error * weights).sum() / (weights.sum() * clean_motion.shape[-1]).clamp_min(1)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        motion_dim: int,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """使用确定性 DDIM 采样；减少 sampling_steps 可加速测试。"""
        sampling_steps = min(sampling_steps or self.num_steps, self.num_steps)
        time_indices = torch.linspace(
            self.num_steps - 1, 0, sampling_steps, device=condition.device
        ).long().unique_consecutive()
        motion = torch.randn((*condition.shape[:2], motion_dim), device=condition.device)

        for index, timestep in enumerate(time_indices):
            t = torch.full((condition.shape[0],), timestep, device=condition.device, dtype=torch.long)
            predicted_x0 = self.denoiser(motion, t, condition)
            if index == len(time_indices) - 1:
                motion = predicted_x0
                continue
            next_timestep = time_indices[index + 1]
            alpha_bar = self.alpha_bars[timestep]
            next_alpha_bar = self.alpha_bars[next_timestep]
            predicted_noise = (motion - alpha_bar.sqrt() * predicted_x0) / (1 - alpha_bar).sqrt()
            motion = next_alpha_bar.sqrt() * predicted_x0 + (1 - next_alpha_bar).sqrt() * predicted_noise
        return motion
