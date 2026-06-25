"""Tune the decision/post-processing on the full OOF (working/oof.npz).

The metric is ~80% ordinal-zone, so how we map model output -> final value (and
where we place the zone cutpoints) is a high-ROI, no-retrain lever. We compare:
  reg | pmf_exp | expected_cost | blend_thresh(w, cuts)
and pick the best by the EXACT metric, then sanity-check stability with a
per-fold "optimize on 4 folds, score the held-out fold" pass (guards overfitting
the decision to OOF). Saves working/decision.json for predict_test.
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
print(f"OOF samples: {m.sum()} | folds present: {sorted(set(fold[m].tolist()))}")

def pmf_at(T, mask):
    return softmax(logits[mask] / T, 1)

def score_dc(dc, mask):
    pmf = pmf_at(dc.get("T", 1.0), mask)
    pred = apply_decision(pmf, reg_oof[mask], centers, dc)
    return evaluate(y[mask], pred)

# ---- base strategies (search T where relevant) ----
def best_T(strategy, mask, grid=None):
    grid = grid or list(np.round(np.r_[np.arange(0.4,1.0,0.1), np.arange(1.0,3.01,0.25)],3))
    b = (1.0, 1e9)
    for T in grid:
        s = score_dc({"strategy": strategy, "T": T}, mask)
        if s < b[1]: b = (float(T), s)
    return b

print("\n--- base strategies (full OOF) ---")
for strat in ["reg", "pmf_exp", "expected_cost"]:
    T, s = best_T(strat, m)
    print(f"  {strat:14s} bestT={T:<4} score={s:.4f}")

# ---- blend_thresh: optimize w and cuts ----
def make_obj(mask, w):
    def obj(cuts):
        c = np.sort(cuts)
        return score_dc({"strategy": "blend_thresh", "w": w, "cuts": c.tolist(), "T": 1.0, "margin": 0.5}, mask)
    return obj

def optimize_blend(mask):
    best = (None, 1e9)
    for w in [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]:
        # multi-start Nelder-Mead around the true bounds
        for init in ([12,35,48], [10,33,46], [14,37,50]):
            r = minimize(make_obj(mask, w), np.array(init, float), method="Nelder-Mead",
                         options={"xatol": 0.1, "fatol": 1e-4, "maxiter": 400})
            if r.fun < best[1]:
                best = ({"strategy": "blend_thresh", "w": w, "cuts": np.sort(r.x).round(3).tolist(),
                         "T": 1.0, "margin": 0.5}, float(r.fun))
    return best

dc_best, s_best = optimize_blend(m)
print(f"\n--- blend_thresh (optimized on full OOF) ---")
print(f"  best: w={dc_best['w']} cuts={dc_best['cuts']} score={s_best:.4f}")
comp = evaluate_components(y[m], apply_decision(pmf_at(1.0, m), reg_oof[m], centers, dc_best))
print(f"  components: zone={comp['w_zone']:.3f} abs={comp['w_absolute']:.3f} high={comp['w_high']:.3f} "
      f"ext={comp['w_extreme']:.3f} zone_acc={comp['zone_accuracy']:.3f}")

# ---- per-fold stability: optimize on other folds, score held-out fold ----
print("\n--- per-fold honesty check (optimize on 4 folds, score the 5th) ---")
folds = sorted(set(fold[m].tolist()))
oos_scores = []
for f in folds:
    tr = m & (fold != f); te = m & (fold == f)
    dcf, _ = optimize_blend(tr)
    s_te = score_dc(dcf, te)
    s_naive = score_dc({"strategy": "expected_cost", "T": best_T("expected_cost", tr)[0]}, te)
    oos_scores.append(s_te)
    print(f"  fold{f}: tuned-on-rest -> held-out={s_te:.4f} (cuts={dcf['cuts']} w={dcf['w']}) | exp_cost={s_naive:.4f}")
print(f"  mean held-out (tuned decision) = {np.mean(oos_scores):.4f} +/- {np.std(oos_scores):.4f}")

# choose final decision: blend_thresh if it beats expected_cost on full OOF, else expected_cost
T_ec, s_ec = best_T("expected_cost", m)
if s_best < s_ec:
    final = dc_best
else:
    final = {"strategy": "expected_cost", "T": T_ec}
json.dump(final, open(OUT / "decision.json", "w"), indent=2)
print(f"\nSAVED decision.json: {final}\n  full-OOF score={min(s_best, s_ec):.4f} (expected_cost={s_ec:.4f})")
