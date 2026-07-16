"""Central configuration for selecting and weighting preference rewards."""

from .preference import RewardSuite, RewardTerm
from .reward_functions import control_reward, motion_smoothness_reward, simulation_reward


def build_reward_suite() -> RewardSuite:
    """Build the rewards used to select winner/loser motion pairs.

    This is the only function that normally needs editing when changing reward
    combinations or their selection hyperparameters.
    """
    rewards = RewardSuite(
        terms=[
            RewardTerm(
                name="control_error",
                evaluator=control_reward,
                weight=1.0,
                higher_is_better=False,  # Error: lower is better.
                margin=0.0,
                enabled=False,
            ),
            RewardTerm(
                name="simulation_error",
                evaluator=simulation_reward,
                weight=0.1,
                higher_is_better=False,
                margin=0.0,
                enabled=False,
            ),
            RewardTerm(
                name="motion_smoothness",
                evaluator=motion_smoothness_reward,
                weight=1.0,
                higher_is_better=True,  # Less negative is smoother and better.
                margin=0.0,
                enabled=True,
            ),

            # -----------------------------------------------------------------
            # ADD A NEW REWARD HERE after defining it in reward_functions.py:
            #
            # RewardTerm(
            #     name="text_alignment",
            #     evaluator=text_alignment_reward,
            #     weight=0.2,
            #     higher_is_better=True,
            #     margin=0.0,
            #     enabled=True,
            # ),
            # -----------------------------------------------------------------
        ],
        # Normalized smoothness rewards differ at roughly 1e-5 in the provided
        # simple_diffusion checkpoint, so use a small but non-zero smoke-test gap.
        min_score_gap=1e-6,
    )

    # Runtime switches can be placed here. This is equivalent to setting the
    # corresponding RewardTerm's `enabled` field above.
    # rewards.set_enabled("simulation_error", False)

    # Other convenient runtime adjustments:
    # rewards.set_enabled("simulation_error", True)
    # rewards.set_weight("control_error", 2.0)

    return rewards
