from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor

from .interfaces import DiffusionDPOAdapter
from .loss import DiffusionDPOLoss


@dataclass
class DPOTrainingConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    beta: float = 20.0
    lambda_dpo: float = 1.0
    lambda_sft: float = 2.0
    max_steps: int = 5_000
    grad_clip: float = 1.0
    log_every: int = 20
    save_every: int = 200
    output_dir: str = "dpo_output"
    wandb_enabled: bool = False
    wandb_project: str = "robot-motion-dpo"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_log_every: int = 10
    eval_every: int = 100
    eval_num_conditions: int = 16
    eval_samples_per_condition: int = 1


class DPOTrainer:
    def __init__(
        self,
        policy: DiffusionDPOAdapter,
        config: DPOTrainingConfig,
        device: Optional[torch.device] = None,
        reward_evaluator: Optional[Callable[[Tensor, Any], Mapping[str, float]]] = None,
        eval_conditions: Optional[List[Any]] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.reward_evaluator = reward_evaluator
        self.eval_conditions = list(eval_conditions or [])[: config.eval_num_conditions]

        # The reference is an immutable snapshot of the initial policy.
        self.reference = copy.deepcopy(policy).to(self.device)
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

        self.objective = DiffusionDPOLoss(
            beta=config.beta,
            lambda_dpo=config.lambda_dpo,
            lambda_sft=config.lambda_sft,
        )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.policy.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.wandb_run = self._init_wandb()

    def _init_wandb(self):
        if not self.config.wandb_enabled:
            return None
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "wandb_enabled=True but wandb is not installed; run `pip install wandb`"
            ) from error
        run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_name,
            mode=self.config.wandb_mode,
            config=asdict(self.config),
            dir=self.config.output_dir,
        )
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("eval/*", step_metric="train/step")
        return run

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        winner = batch["winner"].to(self.device)
        loser = batch["loser"].to(self.device)
        pair_batch_size = winner.shape[0]

        condition = self.policy.collate_conditions(batch["conditions"])
        condition = self.policy.move_condition_to_device(condition, self.device)
        doubled_condition = self.policy.repeat_condition(condition, repeats=2)

        timesteps = self.policy.sample_timesteps(pair_batch_size, self.device)
        timestep_weights = self.policy.timestep_weights(timesteps)
        doubled_timesteps = torch.cat([timesteps, timesteps], dim=0)

        # Same noise for the winner and loser of each pair.
        pair_noise = torch.randn_like(winner)
        clean_motion = torch.cat([winner, loser], dim=0)
        doubled_noise = torch.cat([pair_noise, pair_noise], dim=0)
        noisy_motion = self.policy.add_noise(
            clean_motion,
            doubled_timesteps,
            doubled_noise,
            doubled_condition,
        )
        target = self.policy.training_target(
            clean_motion,
            doubled_noise,
            doubled_timesteps,
        )
        loss_mask = self.policy.loss_mask(doubled_condition)

        # The adapter must encode CFG/dropout masks in this shared state.
        shared_state = self.policy.make_shared_forward_state(
            pair_batch_size,
            self.device,
        )

        # Match PhysMoDPO: policy/reference have the same train/eval mode and
        # consume identical internal randomness (e.g. Transformer dropout).
        self.reference.train(self.policy.training)
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        policy_prediction = self.policy.denoise(
            noisy_motion,
            doubled_timesteps,
            doubled_condition,
            shared_state,
        )
        cpu_rng_after = torch.get_rng_state()
        cuda_rng_after = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        policy_losses = self.policy.per_sample_loss(
            policy_prediction,
            target,
            mask=loss_mask,
        )
        policy_winner_loss, policy_loser_loss = policy_losses.chunk(2)

        with torch.no_grad():
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, self.device)
            reference_prediction = self.reference.denoise(
                noisy_motion,
                doubled_timesteps,
                doubled_condition,
                shared_state,
            )
            reference_losses = self.reference.per_sample_loss(
                reference_prediction,
                target,
                mask=loss_mask,
            )
            reference_winner_loss, reference_loser_loss = reference_losses.chunk(2)

        # Advance the global RNG as if only the policy forward had happened.
        torch.set_rng_state(cpu_rng_after)
        if cuda_rng_after is not None:
            torch.cuda.set_rng_state(cuda_rng_after, self.device)

        terms = self.objective(
            policy_winner_loss,
            policy_loser_loss,
            reference_winner_loss,
            reference_loser_loss,
        )
        terms["weighted_loss"] = (terms["loss"] * timestep_weights).mean()
        return terms

    def train(self, dataloader: Iterable[Dict[str, Any]]) -> DiffusionDPOAdapter:
        self.policy.train()
        iterator = iter(dataloader)

        for step in range(1, self.config.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            self.optimizer.zero_grad(set_to_none=True)
            terms = self.compute_loss(batch)
            terms["weighted_loss"].backward()

            grad_norm = self._gradient_norm()
            if self.config.grad_clip > 0:
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.grad_clip,
                )
                grad_norm = float(clipped_norm)
            self.optimizer.step()

            if self.wandb_run is not None and (
                step % self.config.wandb_log_every == 0 or step == 1
            ):
                self._log_training_metrics(step, terms, grad_norm)

            if step % self.config.log_every == 0 or step == 1:
                print(
                    f"step={step} "
                    f"loss={terms['weighted_loss'].item():.5f} "
                    f"dpo={terms['dpo_loss'].mean().item():.5f} "
                    f"sft={terms['sft_loss'].mean().item():.5f} "
                    f"pref_acc={terms['preference_accuracy'].mean().item():.3f}"
                )

            if step % self.config.save_every == 0 or step == self.config.max_steps:
                self.save(step)

            if (
                self.reward_evaluator is not None
                and self.eval_conditions
                and self.config.eval_every > 0
                and (step % self.config.eval_every == 0 or step == self.config.max_steps)
            ):
                evaluation = self.evaluate_generation(step)
                self._print_evaluation(step, evaluation)

        if self.wandb_run is not None:
            self.wandb_run.finish()
        return self.policy

    def _gradient_norm(self) -> float:
        squared_norm = 0.0
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().norm(2).item() ** 2
        return squared_norm ** 0.5

    def _log_training_metrics(
        self,
        step: int,
        terms: Dict[str, Tensor],
        grad_norm: float,
    ) -> None:
        self.wandb_run.log(
            {
                "train/step": step,
                "train/total_loss": terms["weighted_loss"].item(),
                "train/dpo_loss": terms["dpo_loss"].mean().item(),
                "train/sft_loss": terms["sft_loss"].mean().item(),
                "train/preference_accuracy": terms["preference_accuracy"].mean().item(),
                "train/policy_logratio": terms["policy_logratio"].mean().item(),
                "train/reference_logratio": terms["reference_logratio"].mean().item(),
                "train/logit_margin": terms["logits"].mean().item(),
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )

    @staticmethod
    def _motion_statistics(motion: Tensor) -> Dict[str, float]:
        motion = motion.detach().float()
        variance = motion.var(dim=0, unbiased=False).mean().item() if motion.shape[0] > 1 else 0.0
        velocity = (
            (motion[1:] - motion[:-1]).square().mean().sqrt().item()
            if motion.shape[0] > 1
            else 0.0
        )
        return {"motion_variance": variance, "motion_velocity_rms": velocity}

    @staticmethod
    def _capture_rng(device: torch.device):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng(state, device: torch.device) -> None:
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    @torch.no_grad()
    def evaluate_generation(self, step: int) -> Dict[str, Any]:
        """Compare full policy/reference samples under identical random noise."""
        policy_was_training = self.policy.training
        reference_was_training = self.reference.training
        self.policy.eval()
        self.reference.eval()

        policy_rewards: Dict[str, List[float]] = {}
        reference_rewards: Dict[str, List[float]] = {}
        policy_stats: Dict[str, List[float]] = {}
        reference_stats: Dict[str, List[float]] = {}

        for condition in self.eval_conditions:
            rng_before = self._capture_rng(self.device)
            policy_samples = self.policy.generate_motion(
                condition, self.config.eval_samples_per_condition
            )
            rng_after = self._capture_rng(self.device)
            self._restore_rng(rng_before, self.device)
            reference_samples = self.reference.generate_motion(
                condition, self.config.eval_samples_per_condition
            )
            self._restore_rng(rng_after, self.device)

            for policy_motion, reference_motion in zip(policy_samples, reference_samples):
                policy_result = self.reward_evaluator(policy_motion, condition)
                reference_result = self.reward_evaluator(reference_motion, condition)
                for name, value in policy_result.items():
                    policy_rewards.setdefault(name, []).append(float(value))
                for name, value in reference_result.items():
                    reference_rewards.setdefault(name, []).append(float(value))
                for name, value in self._motion_statistics(policy_motion).items():
                    policy_stats.setdefault(name, []).append(value)
                for name, value in self._motion_statistics(reference_motion).items():
                    reference_stats.setdefault(name, []).append(value)

        metrics: Dict[str, Any] = {"train/step": step}
        for name in policy_rewards:
            policy_values = torch.tensor(policy_rewards[name])
            reference_values = torch.tensor(reference_rewards[name])
            delta = policy_values - reference_values
            metrics.update(
                {
                    f"eval/{name}/policy_mean": policy_values.mean().item(),
                    f"eval/{name}/reference_mean": reference_values.mean().item(),
                    f"eval/{name}/improvement": delta.mean().item(),
                    f"eval/{name}/policy_win_rate": (delta > 0).float().mean().item(),
                }
            )
            if self.wandb_run is not None:
                import wandb

                metrics[f"eval/{name}/policy_distribution"] = wandb.Histogram(
                    policy_values.numpy()
                )
                metrics[f"eval/{name}/reference_distribution"] = wandb.Histogram(
                    reference_values.numpy()
                )
                metrics[f"eval/{name}/delta_distribution"] = wandb.Histogram(delta.numpy())

        for name in policy_stats:
            policy_mean = sum(policy_stats[name]) / len(policy_stats[name])
            reference_mean = sum(reference_stats[name]) / len(reference_stats[name])
            metrics[f"eval/{name}/policy_mean"] = policy_mean
            metrics[f"eval/{name}/reference_mean"] = reference_mean
            metrics[f"eval/{name}/ratio"] = policy_mean / max(reference_mean, 1e-12)

        if self.wandb_run is not None:
            self.wandb_run.log(metrics)
        self.policy.train(policy_was_training)
        self.reference.train(reference_was_training)
        return metrics

    @staticmethod
    def _print_evaluation(step: int, metrics: Dict[str, Any]) -> None:
        scalars = [
            f"{key}={value:.5g}"
            for key, value in metrics.items()
            if key.endswith(("/improvement", "/policy_win_rate", "/ratio"))
            and isinstance(value, (float, int))
        ]
        print(f"eval step={step} " + " ".join(scalars))

    def save(self, step: int) -> None:
        path = Path(self.config.output_dir) / f"policy_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )
