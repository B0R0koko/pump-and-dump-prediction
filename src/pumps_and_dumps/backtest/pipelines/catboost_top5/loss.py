"""An equal-event Top-5 hinge objective for finding the pumped asset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pumps_and_dumps.backtest.datasets.columns import (
    COL_CURRENCY_PAIR,
    COL_IS_PUMPED,
    COL_PUMP_HASH,
    COL_PUMP_ID,
)


@dataclass(frozen=True)
class EventGroup:
    positive: int
    negatives: np.ndarray


@dataclass(frozen=True)
class EventGroups:
    """Event membership expressed as positions in the original frame."""

    group_column: str
    row_count: int
    events: tuple[EventGroup, ...]

    @classmethod
    def from_frame(cls, data: pd.DataFrame) -> "EventGroups":
        group_column = _validate_columns(data)
        labels = data[COL_IS_PUMPED].to_numpy(dtype=bool)
        events = []
        for positions in data.groupby(group_column, sort=False, observed=True).indices.values():
            positives = positions[labels[positions]]
            if len(positives) != 1:
                raise ValueError("Top5 requires exactly one pumped asset per event")
            events.append(EventGroup(int(positives[0]), positions[~labels[positions]]))
        return cls(group_column, len(data), tuple(events))


def _validate_columns(data: pd.DataFrame) -> str:
    if data.empty:
        raise ValueError("Top5 requires a nonempty dataset")
    group_column = next((column for column in (COL_PUMP_HASH, COL_PUMP_ID) if column in data), None)
    if group_column is None:
        raise ValueError("Top5 requires pump_hash or pump_id event groups")
    if data[group_column].isna().any():
        raise ValueError("Top5 event groups must not be missing")
    if COL_IS_PUMPED not in data or not data[COL_IS_PUMPED].isin([0, 1]).all():
        raise ValueError("Top5 requires binary is_pumped labels (0 or 1)")
    if COL_CURRENCY_PAIR not in data or data[COL_CURRENCY_PAIR].isna().any():
        raise ValueError("Top5 requires a currency_pair for every asset")
    if data.duplicated([group_column, COL_CURRENCY_PAIR]).any():
        raise ValueError("Top5 requires each asset to occur once per event")
    return group_column


class Top5Loss:
    """Mean event hinge against the fifth-highest negative asset score.

    Each event contributes max(0, margin + fifth_negative - positive). Events
    with fewer than five negatives already fit into a five-asset portfolio
    and contribute zero. Higher-ranked negatives receive no direct penalty.

    Gradients share ties at the fifth-negative boundary equally, making updates
    independent of input row order. The subgradient at the hinge itself is zero.
    """

    def __init__(self, groups: EventGroups, margin: float = 1.0):
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be positive and finite")
        self.groups = groups
        self.margin = float(margin)

    @classmethod
    def from_frame(cls, data: pd.DataFrame, margin: float = 1.0) -> "Top5Loss":
        return cls(EventGroups.from_frame(data), margin=margin)

    def value(self, logits: np.ndarray) -> float:
        scores = self._scores(logits)
        total = 0.0
        for event in self.groups.events:
            if len(event.negatives) < 5:
                continue
            hinge, _ = self._event_terms(scores, event)
            total += max(hinge, 0.0)
        return total / len(self.groups.events)

    def negative_gradient(self, logits: np.ndarray) -> np.ndarray:
        """Return -dL/dlogit in the same positional order as the input frame."""
        scores = self._scores(logits)
        gradient = np.zeros_like(scores)
        for event in self.groups.events:
            if len(event.negatives) < 5:
                continue
            hinge, boundary = self._event_terms(scores, event)
            if hinge <= 0.0:
                continue
            gradient[event.positive] = 1.0
            gradient[event.negatives[boundary]] = -1.0 / boundary.sum()
        return gradient / len(self.groups.events)

    def _scores(self, logits: np.ndarray) -> np.ndarray:
        scores = np.asarray(logits, dtype=float)
        if scores.shape != (self.groups.row_count,) or not np.isfinite(scores).all():
            raise ValueError("Top5 requires one finite logit per dataset row")
        return scores

    def _event_terms(self, scores: np.ndarray, event: EventGroup) -> tuple[float, np.ndarray]:
        negatives = scores[event.negatives]
        cutoff = np.partition(negatives, len(negatives) - 5)[-5]
        hinge = self.margin + cutoff - scores[event.positive]
        return float(hinge), negatives == cutoff
