# Generic Diffusion-DPO

This folder implements an offline DPO post-training pipeline independent of
OmniControl. Only `DiffusionDPOAdapter` is model-specific.

## Required adapter methods

`generate_motion()` alone is insufficient for training because sampling is
normally non-differentiable. Implement the following one-step training API:

```python
class MyAdapter(DiffusionDPOAdapter):
    def __init__(self, pretrained_model, diffusion):
        super().__init__()
        self.model = pretrained_model       # registered trainable module
        self.diffusion = diffusion

    @torch.no_grad()
    def generate_motion(self, condition, num_samples):
        return generate_motion(condition, num_samples=num_samples)

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.diffusion.num_steps, (batch_size,), device=device)

    def add_noise(self, clean_motion, timesteps, noise, condition):
        return self.diffusion.q_sample(clean_motion, timesteps, noise=noise)

    def denoise(self, noisy_motion, timesteps, condition, shared_state):
        return self.model(noisy_motion, timesteps, condition, **shared_state)

    def training_target(self, clean_motion, noise, timesteps):
        return clean_motion                 # x0 prediction
        # return noise                      # epsilon prediction

    def per_sample_loss(self, prediction, target, mask=None):
        loss = (prediction - target).square()
        if mask is not None:
            loss = loss * mask
        return loss.flatten(1).mean(1)
```

If the model uses classifier-free condition dropout, override
`make_shared_forward_state()` and explicitly create a mask once, repeated in
`[winner_batch, loser_batch]` order. The same state is sent to policy and
reference, preventing preference gradients from being caused by different
dropout masks. The trainer also restores the PyTorch RNG before the reference
forward so ordinary Transformer dropout is shared, matching PhysMoDPO.

## Reward and pipeline

Put concrete reward implementations in `reward_functions.py`, inside the
`USER CUSTOMIZATION` block. Enable and weight them centrally in
`reward_config.py::build_reward_suite()`:

```python
from generic_diffusion_dpo import (
    DPOTrainingConfig,
    run_dpo_with_reward_suite,
)
from generic_diffusion_dpo.reward_config import build_reward_suite


rewards = build_reward_suite()
training = DPOTrainingConfig(
    beta=20.0,
    lambda_dpo=1.0,
    lambda_sft=2.0,
    learning_rate=1e-6,
    max_steps=5_000,
    output_dir="outputs/checkpoints",
)

finetuned_model = run_dpo_with_reward_suite(
    model=MyAdapter(pretrained_model, diffusion),
    conditions=training_conditions,
    rewards=rewards,
    preference_path="outputs/preferences.pt",
    candidates_per_condition=12,
    batch_size=64,
    training=training,
)
```

The two training coefficients and all reward coefficients are therefore changed
without touching implementation code:

```python
training.lambda_dpo = 0.5
training.lambda_sft = 1.0

rewards.set_weight("control_error", 2.0)
rewards.set_enabled("simulation_error", False)
print(rewards.describe())
```

The reward evaluator runs only during offline preference construction. During
training, gradients flow through policy winner/loser denoising losses; the
frozen reference model and reward metrics never receive gradients.

The default `reward_config.py` enables only `motion_smoothness_reward`, a
simulator-free smoke-test reward equal to the negative mean squared second-order
temporal difference. It expects normalized motion `[T, D]` and optionally reads
`condition["mask"]`. Smoothness alone can favor low-activity motion, so combine
or replace it with task/control rewards for real training.

## W&B monitoring

Enable W&B through `DPOTrainingConfig`:

```python
training = DPOTrainingConfig(
    wandb_enabled=True,
    wandb_project="robot-motion-dpo",
    wandb_name="my-dpo-run",
    wandb_mode="online",       # use "offline" without network
    wandb_log_every=10,
    eval_every=100,
    eval_num_conditions=16,
    eval_samples_per_condition=1,
)
```

The trainer logs optimization metrics (`total_loss`, DPO/SFT losses, preference
accuracy, policy/reference log-ratios, gradient norm and learning rate). At each
evaluation interval it performs full policy/reference sampling from identical
random noise and logs reward means, improvement, policy win rate and reward
histograms. It also compares temporal motion variance and RMS velocity; ratios
collapsing toward zero indicate that a smoothness reward may be producing nearly
static motion.
