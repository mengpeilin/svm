"""Held-out metrics of a dense SVM reward: does the per-chunk reward tell
eventual success from failure early in an episode, and does it track progress?

One reward r_c per decision step c. K = decision steps of the shortest
held-out success, so every episode is alive for c < K.

  era                mean over c < K of the episode-level success-vs-failure
                     AUC of the chunk reward r_c (early per-chunk AUC)
  era_margin         mean over c < K of (mean r_c+ - mean r_c-) / pooled std
  step_auc           success-vs-failure AUC over all held-out steps
  early_step_auc     the same over steps c < K
  initial_step_auc   first step only (nothing has happened yet)
  progress_spearman  rank correlation of reward with time in success episodes
  clip_fraction      share of logits at the +-beta clip
"""

from __future__ import annotations

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC of `scores` for `labels` 1 vs 0."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def reward_metrics(succ: list[np.ndarray], fail: list[np.ndarray], logits: np.ndarray, beta: float) -> dict:
    """`succ` / `fail`: per-episode reward series (failures already capped);
    `logits`: all their logits, for the clip fraction."""
    K = int(min(len(r) for r in succ))
    Rs, Rf = np.stack([r[:K] for r in succ]), np.stack([r[:K] for r in fail])
    ep_labels = np.concatenate([np.ones(len(succ)), np.zeros(len(fail))])
    chunk_auc = [roc_auc(np.concatenate([Rs[:, c], Rf[:, c]]), ep_labels) for c in range(K)]
    pooled = np.sqrt(0.5 * (Rs.var(0, ddof=1) + Rf.var(0, ddof=1)))
    step_labels = np.concatenate([np.ones(sum(map(len, succ))), np.zeros(sum(map(len, fail)))])
    return {
        "K": K,
        "n_heldout": [len(succ), len(fail)],
        "era": float(np.mean(chunk_auc)),
        "era_margin": float(np.mean((Rs.mean(0) - Rf.mean(0)) / np.maximum(pooled, 1e-8))),
        "step_auc": roc_auc(np.concatenate(succ + fail), step_labels),
        "early_step_auc": roc_auc(np.concatenate([Rs.ravel(), Rf.ravel()]),
                                  np.concatenate([np.ones(Rs.size), np.zeros(Rf.size)])),
        "initial_step_auc": roc_auc(np.array([r[0] for r in succ + fail]), ep_labels),
        "progress_spearman": float(np.mean([spearman(r, np.arange(len(r), dtype=np.float64)) for r in succ])),
        "mean_reward_success": float(np.concatenate(succ).mean()),
        "mean_reward_failure": float(np.concatenate(fail).mean()),
        "clip_fraction": float((np.abs(logits) >= beta).mean()),
    }
