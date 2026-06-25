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


def expected_cost_decision(pmf, centers, high_frac=0.384, out_grid=None, return_loss=False):
    """Bayes-optimal output minimizing the EXACT metric, integrating the model's
    predictive PMF over burden. pmf: (N,K) probabilities over `centers` (K,).

    Per-sample loss (the true metric factored by 1/N; argmin is unaffected):
        l(a,y) = 0.80*zone_pen + 0.05*|a-y| + (0.05/high_frac)*|a-y|*[y>=48] + 10*extreme
    where the high term's weight (0.05*N/N_high) reproduces the metric's
    conditional `high_component` mean. We grid-search a over out_grid.
    """
    pmf = np.asarray(pmf, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if out_grid is None:
        out_grid = np.arange(0.0, max(centers.max(), 68.0) + 0.001, 0.5)
    A = np.asarray(out_grid, dtype=float)
    za = to_zone(A)[:, None]          # (nA,1)
    zc = to_zone(centers)[None, :]    # (1,K)
    dz = np.abs(za - zc)
    zone_pen = np.where(dz == 0, 0.0, np.where(dz == 1, 60.0, 100.0))
    mae = np.abs(A[:, None] - centers[None, :])
    high_w = 0.05 / max(high_frac, 1e-6)
    mae_high = mae * (centers[None, :] >= 48.0) * high_w
    extreme = (((centers[None, :] <= 12.0) & (A[:, None] > 25.0)) |
               ((centers[None, :] >= 48.0) & (A[:, None] < 40.0))).astype(float) * 10.0
    Lmat = 0.80 * zone_pen + 0.05 * mae + mae_high + extreme   # (nA, K)
    exp_loss = pmf @ Lmat.T            # (N, nA)
    best = exp_loss.argmin(1)
    out = A[best]
    if return_loss:
        return out, exp_loss[np.arange(len(out)), best]
    return out


def fit_temperature(pmf_logits, y_true, centers, grid=None):
    """Pick a single temperature T (scaling logits) that minimizes the OOF metric
    after the expected-cost decision. Returns (best_T, best_score)."""
    from scipy.special import softmax
    if grid is None:
        grid = np.concatenate([np.arange(0.3, 1.0, 0.1), np.arange(1.0, 4.01, 0.25)])
    best = (1.0, 1e9)
    for T in grid:
        pmf = softmax(np.asarray(pmf_logits) / T, axis=1)
        pred = expected_cost_decision(pmf, centers)
        s = evaluate(y_true, pred)
        if s < best[1]:
            best = (float(T), float(s))
    return best


if __name__ == "__main__":
    # sanity checks
    yt = np.array([5.0, 20.0, 40.0, 60.0])
    print("perfect:", evaluate(yt, yt))
    print("components perfect:", evaluate_components(yt, yt))
    print("off by zone:", evaluate(yt, yt + 15))
    print("zones of [5,20,40,60]:", to_zone(yt))
