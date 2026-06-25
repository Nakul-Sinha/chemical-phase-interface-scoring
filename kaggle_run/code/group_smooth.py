"""Validate WITHIN-EXPERIMENT prediction smoothing on the OOF.

Test images are video frames; within an experiment the true burden is ~constant
(intra-group std 0.68). Averaging the model's PMF across frames of the same
experiment reduces per-frame noise (esp. near zone boundaries). Because our CV
groups never split an experiment across folds, averaging within groups on the OOF
is a faithful simulation of doing it on the test set. If it improves OOF, apply at
test time (grouping test frames by visual similarity for compliance).
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution_core as S
from metric import evaluate, evaluate_components, expected_cost_decision

OUT = Path(os.environ["OUT_DIR"]); DATA = os.environ["DATA_ROOT"]
oof = np.load(OUT / "oof.npz", allow_pickle=True)
logits, reg, y, done, centers = oof["logits"], oof["reg"], oof["y"], oof["done"], oof["centers"]
dc = json.load(open(OUT / "decision.json")); T = float(dc.get("T", 1.0))
df = pd.read_csv(Path(DATA) / "train.csv")
pmf = softmax(logits / T, 1); m = done

pred_pf = expected_cost_decision(pmf[m], centers)
c_pf = evaluate_components(y[m], pred_pf)
print(f"per-frame OOF      : {c_pf['total']:.4f}  zone_acc={c_pf['zone_accuracy']:.3f}")

for tol in [0, 1, 2]:   # exact-size (purest) ... near-size@2px
    groups = S.compute_groups(df, DATA, tol_abs=tol)
    # intra-group burden std (purity)
    stds = [y[groups == g].std() for g in np.unique(groups) if (groups == g).sum() >= 2]
    pmf_g = pmf.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pmf_g[idx] = pmf[idx].mean(0, keepdims=True)
    pred_g = np.clip(expected_cost_decision(pmf_g[m], centers), 0, 100)
    c_g = evaluate_components(y[m], pred_g)
    print(f"smooth tol={tol}px: {c_g['total']:.4f}  zone_acc={c_g['zone_accuracy']:.3f}  "
          f"impr={c_pf['total']-c_g['total']:+.3f}  | n_grp={len(np.unique(groups))} "
          f"intra_std={np.mean(stds):.2f} medsz={int(np.median(np.bincount(groups)))}")
