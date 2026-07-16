from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .interfaces import Condition, DiffusionDPOAdapter


RewardEvaluator = Callable[[Tensor, Condition], Mapping[str, float]]


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


@dataclass(frozen=True)
class MetricSpec:
    name: str
    weight: float = 1.0
    higher_is_better: bool = True
    margin: float = 0.0


@dataclass
class RewardTerm:
    """One swappable reward and its preference-selection hyperparameters."""

    name: str
    evaluator: Callable[[Tensor, Condition], float]
    weight: float = 1.0
    higher_is_better: bool = True
    margin: float = 0.0
    enabled: bool = True


class RewardSuite:
    """Composable reward configuration used by preference generation.

    Reward functions and their lambda-like weights live in one place. Terms can
    be changed between DPO rounds without touching the dataset or trainer.
    """

    def __init__(self, terms: Sequence[RewardTerm], min_score_gap: float = 0.0):
        names = [term.name for term in terms]
        if len(names) != len(set(names)):
            raise ValueError("reward term names must be unique")
        self.terms = {term.name: term for term in terms}
        self.min_score_gap = min_score_gap

    def __call__(self, motion: Tensor, condition: Condition) -> Mapping[str, float]:
        return {
            term.name: float(term.evaluator(motion, condition))
            for term in self.terms.values()
            if term.enabled
        }

    def set_weight(self, name: str, weight: float) -> "RewardSuite":
        self.terms[name].weight = weight
        return self

    def set_enabled(self, name: str, enabled: bool) -> "RewardSuite":
        self.terms[name].enabled = enabled
        return self

    def pair_selector(self) -> "PreferencePairSelector":
        metrics = [
            MetricSpec(
                name=term.name,
                weight=term.weight,
                higher_is_better=term.higher_is_better,
                margin=term.margin,
            )
            for term in self.terms.values()
            if term.enabled
        ]
        return PreferencePairSelector(metrics, min_score_gap=self.min_score_gap)

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": term.name,
                "weight": term.weight,
                "higher_is_better": term.higher_is_better,
                "margin": term.margin,
                "enabled": term.enabled,
            }
            for term in self.terms.values()
        ]


class PreferencePairSelector:
    """Select the largest-gap pair satisfying all metric margins."""

    def __init__(self, metrics: Sequence[MetricSpec], min_score_gap: float = 0.0):
        if not metrics:
            raise ValueError("at least one metric is required")
        self.metrics = list(metrics)
        self.min_score_gap = min_score_gap

    @staticmethod
    def _cost(value: float, spec: MetricSpec) -> float:
        return -float(value) if spec.higher_is_better else float(value)

    def score(self, rewards: Mapping[str, float]) -> float:
        # Lower score is better.
        return sum(
            spec.weight * self._cost(rewards[spec.name], spec)
            for spec in self.metrics
        )

    def select(
        self,
        rewards: Sequence[Mapping[str, float]],
    ) -> Optional[Tuple[int, int, float]]:
        scores = [self.score(item) for item in rewards]
        best_pair: Optional[Tuple[int, int, float]] = None

        for winner in range(len(rewards)):
            for loser in range(len(rewards)):
                if winner == loser:
                    continue
                dominates = all(
                    self._cost(rewards[winner][spec.name], spec)
                    < self._cost(rewards[loser][spec.name], spec) - spec.margin
                    for spec in self.metrics
                )
                gap = scores[loser] - scores[winner]
                if dominates and gap >= self.min_score_gap:
                    if best_pair is None or gap > best_pair[2]:
                        best_pair = (winner, loser, gap)
        return best_pair


@torch.no_grad()
def build_preference_data(
    model: DiffusionDPOAdapter,
    conditions: Iterable[Condition],
    reward_evaluator: RewardEvaluator,
    pair_selector: PreferencePairSelector,
    output_path: str,
    candidates_per_condition: int = 12,
) -> List[Dict[str, Any]]:
    """Generate candidates, evaluate rewards and persist winner/loser pairs."""
    was_training = model.training
    model.eval()
    records: List[Dict[str, Any]] = []

    for condition in conditions:
        candidates = model.generate_motion(condition, candidates_per_condition)
        if candidates.shape[0] != candidates_per_condition:
            raise ValueError("generate_motion must return [num_samples, ...]")
        candidate_rewards = [
            dict(reward_evaluator(motion, condition)) for motion in candidates
        ]
        selected = pair_selector.select(candidate_rewards)
        if selected is None:
            continue
        winner, loser, score_gap = selected
        records.append(
            {
                "winner": candidates[winner].detach().cpu(),
                "loser": candidates[loser].detach().cpu(),
                "condition": _to_cpu(condition),
                "winner_metrics": candidate_rewards[winner],
                "loser_metrics": candidate_rewards[loser],
                "score_gap": float(score_gap),
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)
    model.train(was_training)
    return records


class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        try:
            self.records = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions without weights_only
            self.records = torch.load(path, map_location="cpu")
        if not self.records:
            raise ValueError(f"no preference pairs found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def collate_preference_batch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "winner": torch.stack([item["winner"] for item in records]),
        "loser": torch.stack([item["loser"] for item in records]),
        "conditions": [item["condition"] for item in records],
        "score_gap": torch.tensor([item["score_gap"] for item in records]),
    }
