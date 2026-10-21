"""Metrics. The dense ones are CLASS-BALANCED: success and failure frames
weigh 1/2 each (failure episodes run to the cap, so they hold ~2/3 of the
eval frames).

  RMSE         sqrt(MSE(r, g)) of the student's progress reward r = V(s_t,
               track_t) - V(s_t) against the teacher's g = T_{t+16} - T_t, in
               teacher-logit units (lower is better)
  reverse acc  on the chunks the teacher says make progress (g > 0.5), the
               share where r with the real track beats r with the reversed
               track (-track, same observation): 0.5 = blind to the motion's
               direction
"""

from __future__ import annotations

import numpy as np

from .config import SUCCESS

PROGRESS = 0.5


def balanced(label: np.ndarray) -> np.ndarray:
    """Weights summing to 1: half on the success frames, half on the failure frames."""
    succ = label == SUCCESS
    return np.where(succ, 0.5 / succ.sum(), 0.5 / (~succ).sum())


def rmse(r: np.ndarray, g: np.ndarray, label: np.ndarray) -> float:
    return float(np.sqrt(np.average((r - g) ** 2, weights=balanced(label))))


def reverse_acc(forward: np.ndarray, mirrored: np.ndarray, g: np.ndarray, label: np.ndarray) -> float:
    """P(V(s, track) > V(s, -track)) on progress chunks g > PROGRESS; the
    static value cancels, so this is r(real) > r(-track)."""
    m = g > PROGRESS
    return float(np.average(forward[m] > mirrored[m], weights=balanced(label[m])))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of `scores` for labels 1 vs 0, ties averaged."""
    order = np.argsort(scores, kind="mergesort")
    _, inv, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
