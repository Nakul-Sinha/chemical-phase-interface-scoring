"""Exact challenge metric for Chemical Phase Interface Assessment + ordinal helpers.

The official metric (LOWER is better) is dominated (0.80 weight) by ordinal zone
accuracy. This module provides:
  - evaluate(): byte-for-byte copy of the official grader.
  - to_zone(): map burden -> ordinal zone {0,1,2,3}.
  - evaluate_components(): per-term breakdown for diagnostics.
  - best_constant(): the optimal single-value baseline on a label array.
  - decision_value_from_zone_probs(): Bayes-optimal output scalar given P(zone)
    and a regression point estimate, minimizing expected metric. This is the
    key exploit of a known, piecewise-constant metric.
"""
import numpy as np

SEVERITY_BINS = np.array([0.0, 12.0, 35.0, 48.0, 100.000001])
N_ZONES = 4
# Representative "safe center" of each zone (interior, away from boundaries),
# used as a default within-zone output when minimizing expected zone cost.
ZONE_CENTERS = np.array([6.0, 23.0, 41.0, 65.0])


def evaluate(y_true, y_pred):
    """Official grader. Lower is better. Returns float in [0, 100]."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, 100)
    severity_bins = np.array([0.0, 12.0, 35.0, 48.0, 100.000001])
    absolute_gap = np.abs(y_pred - y_true)
    high_burden = y_true >= 48.0
    absolute_component = absolute_gap.mean()
    high_component = absolute_gap[high_burden].mean() if high_burden.any() else absolute_component
    true_zone = np.digitize(y_true, severity_bins) - 1
    pred_zone = np.digitize(y_pred, severity_bins) - 1
    zone_distance = np.abs(true_zone - pred_zone)
    zone_penalty = np.where(zone_distance == 0, 0.0, np.where(zone_distance == 1, 60.0, 100.0))
    zone_component = zone_penalty.mean()
    extreme_miss = ((y_true <= 12.0) & (y_pred > 25.0)) | ((y_true >= 48.0) & (y_pred < 40.0))
    extreme_component = extreme_miss.mean() * 100.0
    score = (
        0.05 * absolute_component
        + 0.05 * high_component
        + 0.80 * zone_component
        + 0.10 * extreme_component
    )
    return float(np.clip(score, 0.0, 100.0))


def to_zone(y):
    """Map burden value(s) to ordinal zone index {0,1,2,3}."""
    return np.digitize(np.asarray(y, dtype=float), SEVERITY_BINS) - 1


def evaluate_components(y_true, y_pred):
    """Return the 4 weighted components plus the total, for diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    absolute_gap = np.abs(y_pred - y_true)
    high = y_true >= 48.0
    abs_c = absolute_gap.mean()
    high_c = absolute_gap[high].mean() if high.any() else abs_c
    tz, pz = to_zone(y_true), to_zone(y_pred)
    zd = np.abs(tz - pz)
    zp = np.where(zd == 0, 0.0, np.where(zd == 1, 60.0, 100.0))
    zone_c = zp.mean()
    extreme = ((y_true <= 12.0) & (y_pred > 25.0)) | ((y_true >= 48.0) & (y_pred < 40.0))
    ext_c = extreme.mean() * 100.0
    total = float(np.clip(0.05 * abs_c + 0.05 * high_c + 0.80 * zone_c + 0.10 * ext_c, 0, 100))
    return {
        "total": total,
        "w_absolute": 0.05 * abs_c,
        "w_high": 0.05 * high_c,
        "w_zone": 0.80 * zone_c,
        "w_extreme": 0.10 * ext_c,
        "raw_mae": abs_c,
        "raw_mae_high": high_c,
        "raw_zone_penalty": zone_c,
        "zone_accuracy": float((zd == 0).mean()),
        "extreme_miss_rate": float(extreme.mean()),
    }


def best_constant(y_true, grid=None):
    """Find the single constant prediction minimizing the metric on y_true."""
    if grid is None:
        grid = np.arange(0, 100.01, 0.25)
    y_true = np.asarray(y_true, dtype=float)
    scores = [evaluate(y_true, np.full_like(y_true, c)) for c in grid]
    i = int(np.argmin(scores))
    return float(grid[i]), float(scores[i])


def _zone_cost_matrix():
    """cost[pred_zone, true_zone] from the 0/60/100 zone-distance penalty."""
    z = np.arange(N_ZONES)
    d = np.abs(z[:, None] - z[None, :])
    return np.where(d == 0, 0.0, np.where(d == 1, 60.0, 100.0))


ZONE_COST = _zone_cost_matrix()  # shape (pred, true)


def decision_value_from_zone_probs(zone_probs, reg_pred=None, true_prevalence=None):
    """Bayes-optimal output scalar(s) minimizing expected metric, given per-zone
    probabilities. zone_probs: (N,4). reg_pred: optional (N,) regression estimate
    used to pick a sensible within-zone value (and to feed the small MAE terms).

    Strategy: zone term dominates (0.80), so choose the output ZONE that minimizes
    expected zone cost E_true[cost[pred_zone, true_zone]]. Then emit a within-zone
    scalar: clip reg_pred into the chosen zone if available, else the zone center.
    """
    zone_probs = np.asarray(zone_probs, dtype=float)
    zone_probs = zone_probs / np.clip(zone_probs.sum(axis=1, keepdims=True), 1e-9, None)
    # expected zone cost for each candidate output zone: (N,4) = probs @ cost^T
    exp_cost = zone_probs @ ZONE_COST.T  # (N, pred_zone)
    chosen = np.argmin(exp_cost, axis=1)  # (N,)
    out = ZONE_CENTERS[chosen].astype(float)
    if reg_pred is not None:
        reg_pred = np.asarray(reg_pred, dtype=float)
        lo = SEVERITY_BINS[chosen]
        hi = SEVERITY_BINS[chosen + 1]
        # keep a small interior margin so boundary jitter doesn't flip zones
        margin = np.minimum(1.0, (hi - lo) * 0.15)
        out = np.clip(reg_pred, lo + margin, hi - margin)
    return out, chosen


if __name__ == "__main__":
    # sanity checks
    yt = np.array([5.0, 20.0, 40.0, 60.0])
    print("perfect:", evaluate(yt, yt))
    print("components perfect:", evaluate_components(yt, yt))
    print("off by zone:", evaluate(yt, yt + 15))
    print("zones of [5,20,40,60]:", to_zone(yt))
