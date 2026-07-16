"""Project-specific reward functions used during offline preference generation.

Keep concrete simulator/model dependencies in this file. The generic DPO loss
and trainer must not import them, because rewards never participate in backward.
"""

from typing import Any

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
