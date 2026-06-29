import os, sys
import numpy as np, pandas as pd
from scipy.special import softmax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import evaluate, to_zone, expected_cost_decision
from collections import Counter

OOF = os.environ.get("OOF_DIR", "working/rebuild_cv")
GROUPS = os.environ.get("GROUPS_CSV", "eda/out/train_groups.csv")

oof = np.load(os.path.join(OOF, "oof.npz"), allow_pickle=True)
done = oof["done"]
y = oof["y"][done]
T = float(oof["T"])
centers = oof["centers"]
ids = oof["ids"][done]
pmf = softmax(oof["logits"][done] / T, 1)

gdf = pd.read_csv(GROUPS)
id2g = dict(zip(gdf.id, gdf.group))
g = np.array([id2g.get(i, -1 - k) for k, i in enumerate(ids)])

zc = np.array([6., 23.5, 41.5, 57.5])
pf = np.clip(expected_cost_decision(pmf, centers), 0, 100)


def grp_apply(vals_or_pmf, mode):
    out = pf.copy()
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        if len(idx) <= 1:
            continue
        if mode == "pmf_mean":
            out[idx] = np.clip(expected_cost_decision(pmf[idx].mean(0, keepdims=True).repeat(len(idx), 0), centers), 0, 100)
        elif mode == "median":
            out[idx] = np.median(pf[idx])
        elif mode == "zonemode":
            z = to_zone(pf[idx])
            mz = Counter(z.tolist()).most_common(1)[0][0]
            inz = pf[idx][z == mz]
            out[idx] = np.median(inz) if len(inz) else zc[mz]
    return out


n_multi = sum(1 for grp in g if (g == grp).sum() > 1)
print(f"OOF n={len(y)}  frames in multi-frame train groups={n_multi} ({100*n_multi/len(y):.1f}%)")
print(f"  (train is mostly singletons, so absolute gains understate the 96.4%-grouped TEST)")
res = {}
res["per_frame"] = evaluate(y, pf)
for mode in ["pmf_mean", "median", "zonemode"]:
    res[mode] = evaluate(y, grp_apply(None, mode))
for k, v in res.items():
    print(f"  {k:10s} OOF={v:.4f}  zones={np.bincount(to_zone(grp_apply(None,k) if k!='per_frame' else pf),minlength=4).tolist()}")
best = min(res, key=res.get)
print(f"BEST consensus method on held-out: {best} ({res[best]:.4f})  vs per_frame ({res['per_frame']:.4f})  delta={res['per_frame']-res[best]:+.3f}")
