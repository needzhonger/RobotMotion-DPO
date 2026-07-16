"""Project-specific reward functions used during offline preference generation.

Keep concrete simulator/model dependencies in this file. The generic DPO loss
and trainer must not import them, because rewards never participate in backward.
"""

from typing import Any, Mapping

import torch
from torch import Tensor


# ============================================================================
# USER CUSTOMIZATION START: define your concrete Reward functions in this block
# ============================================================================

def control_reward(motion: Tensor, condition: Any) -> float:
    """Return the control metric for one generated motion.

    Replace this function body with your finished control_reward implementation.

    Args:
        motion: One generated candidate motion, without a candidate-batch axis.
        condition: The original condition passed to generate_motion. It may
            contain text, target joint trajectories, masks, motion length, etc.

    Returns:
        A Python float. If this is an error/distance, configure the RewardTerm
        with `higher_is_better=False`.
    """
    # Example integration:
    # target = condition["control_target"]
    # mask = condition["control_mask"]
    # return calculate_control_error(motion, target, mask)
    raise NotImplementedError("implement control_reward in reward_functions.py")


def simulation_reward(motion: Tensor, condition: Any) -> float:
    """Optional physics/simulator reward; replace with your implementation."""
    # Example integration:
    # tracked_motion = simulator.track(motion)
    # return simulator.tracking_error(tracked_motion, motion)
    raise NotImplementedError("implement simulation_reward or disable its RewardTerm")


def motion_smoothness_reward(motion: Tensor, condition: Any) -> float:
    """Reward temporally smooth robot motion in normalized action space.

    This is a deliberately simple reward for validating the DPO pipeline without
    a physics simulator. For a motion ``x`` with shape ``[T, D]``, it computes

        reward = -mean((x[t+1] - 2*x[t] + x[t-1]) ** 2)

    so a candidate with smaller frame-to-frame acceleration receives a larger
    (less negative) reward. Computing in normalized space prevents high-scale
    dimensions from dominating low-scale dimensions.

    If ``condition`` is a dictionary containing a boolean ``mask`` of shape
    ``[T]``, padded frames are excluded. Otherwise all frames are used.

    This reward is suitable for a smoke test, but smoothness alone may prefer
    motions with too little movement. A production system should combine it
    with task accuracy or a simulator-based reward.
    """
    if motion.ndim != 2:
        raise ValueError(
            f"motion_smoothness_reward expects motion [T, D], got {tuple(motion.shape)}"
        )

    valid_motion = motion.float()
    if isinstance(condition, Mapping) and "mask" in condition:
        mask = torch.as_tensor(condition["mask"], device=motion.device).bool()
        if mask.ndim != 1 or mask.shape[0] != motion.shape[0]:
            raise ValueError(
                f"condition['mask'] must have shape [{motion.shape[0]}], "
                f"got {tuple(mask.shape)}"
            )
        valid_motion = valid_motion[mask]

    if valid_motion.shape[0] < 3:
        # No second-order temporal difference can be measured.
        return 0.0

    acceleration = valid_motion[2:] - 2.0 * valid_motion[1:-1] + valid_motion[:-2]
    return -float(acceleration.square().mean().item())


# Add further functions here, for example:
#
# def text_alignment_reward(motion: Tensor, condition: Any) -> float:
#     return tmr_model.similarity(motion, condition["text"])
#
# After defining it, import the function and add its RewardTerm in
# reward_config.py::build_reward_suite().

# ============================================================================
# USER CUSTOMIZATION END
# ============================================================================
