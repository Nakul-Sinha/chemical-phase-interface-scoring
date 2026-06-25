"""Tune the decision/post-processing on the full OOF (working/oof.npz), HONESTLY.

The metric is ~80% ordinal-zone, so the output->value map and zone cutpoints are a
no-retrain lever. BUT threshold tuning overfits (esp. under high leave-experiment-out
variance). So we SELECT by the per-fold held-out (nested) score, not full-OOF, and
constrain cuts near the true bounds to avoid degenerate solutions. Robust default =
expected_cost (temperature-only). Saves working/decision.json for predict_test.
"""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import evaluate, evaluate_components, to_zone
from solution_core import apply_decision

OUT = Path(os.environ.get("OUT_DIR", "working"))
oof = np.load(OUT / "oof.npz")
logits, reg_oof, y = oof["logits"], oof["reg"], oof["y"]
fold, done, centers = oof["fold"], oof["done"], oof["centers"]
m = done.copy()
folds = sorted(set(fold[m].tolist()))
print(f"OOF samples: {m.sum()} | folds: {folds}")

# constrain cuts to windows around the true bounds [12,35,48] with a min gap
LO = np.array([6.0, 28.0, 42.0]); HI = np.array([20.0, 41.0, 54.0])
def proj_cuts(c):
    c = np.clip(np.asarray(c, float), LO, HI)
    c[1] = max(c[1], c[0] + 4); c[2] = max(c[2], c[1] + 4)
    return c

def score_dc(dc, mask):
    pmf = softmax(logits[mask] / dc.get("T", 1.0), 1)
    return evaluate(y[mask], apply_decision(pmf, reg_oof[mask], centers, dc))

def best_T(strategy, mask):
    grid = list(np.round(np.r_[np.arange(0.4, 1.0, 0.1), np.arange(1.0, 3.51, 0.25)], 3))
    return min(((float(T), score_dc({"strategy": strategy, "T": T}, mask)) for T in grid), key=lambda t: t[1])

def optimize_blend(mask):
    best = (None, 1e9)
    for w in [1.0, 0.7, 0.5, 0.3, 0.0]:
        for init in ([12, 35, 48], [11, 33, 46], [13, 37, 50]):
            def obj(c):
                c = proj_cuts(c)
                return score_dc({"strategy": "blend_thresh", "w": w, "cuts": c.tolist(), "T": 1.0, "margin": 0.5}, mask)
            r = minimize(obj, np.array(init, float), method="Nelder-Mead",
                         options={"xatol": 0.15, "fatol": 1e-4, "maxiter": 300})
            if r.fun < best[1]:
                best = ({"strategy": "blend_thresh", "w": w, "cuts": proj_cuts(r.x).round(3).tolist(),
                         "T": 1.0, "margin": 0.5}, float(r.fun))
    return best

print("\n--- base strategies (full OOF, context only) ---")
for s in ["reg", "pmf_exp", "expected_cost"]:
    T, sc = best_T(s, m); print(f"  {s:14s} bestT={T:<5} full-OOF={sc:.4f}")
dc_blend_full, s_blend_full = optimize_blend(m)
print(f"  blend_thresh   full-OOF={s_blend_full:.4f}  (w={dc_blend_full['w']} cuts={dc_blend_full['cuts']})")

# ---- HONEST per-fold held-out: tune on other folds, score the held-out fold ----
print("\n--- per-fold held-out (HONEST selection) ---")
ec_oos, blend_oos = [], []
for f in folds:
    tr = m & (fold != f); te = m & (fold == f)
    Tt, _ = best_T("expected_cost", tr)
    ec_oos.append(score_dc({"strategy": "expected_cost", "T": Tt}, te))
    dcf, _ = optimize_blend(tr)
    blend_oos.append(score_dc(dcf, te))
ec_h, blend_h = float(np.mean(ec_oos)), float(np.mean(blend_oos))
print(f"  expected_cost held-out: {ec_h:.4f} +/- {np.std(ec_oos):.3f}")
print(f"  blend_thresh  held-out: {blend_h:.4f} +/- {np.std(blend_oos):.3f}")

# choose the strategy with the better HONEST held-out; fit its params on full OOF
if blend_h + 0.15 < ec_h:           # require a clear margin to prefer the riskier tune
    final = dc_blend_full
    chosen_honest = blend_h
else:
    T_ec, _ = best_T("expected_cost", m)
    final = {"strategy": "expected_cost", "T": T_ec}
    chosen_honest = ec_h
json.dump(final, open(OUT / "decision.json", "w"), indent=2)
comp = evaluate_components(y[m], apply_decision(softmax(logits[m]/final.get("T",1.0),1), reg_oof[m], centers, final))
print(f"\nSAVED decision.json: {final}")
print(f"  honest held-out estimate = {chosen_honest:.4f} | full-OOF = {comp['total']:.4f} (zone_acc {comp['zone_accuracy']:.3f})")
