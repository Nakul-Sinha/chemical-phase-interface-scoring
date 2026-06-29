"""Compute the CALIBRATED honest score from an oof.npz (the LB analog: same blend +
quantile-calibration the submission uses). Usage: python honest_cal.py <oof_dir> [train.csv]
"""
import sys, os
import numpy as np, pandas as pd
from scipy.special import softmax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import evaluate, to_zone

oof_dir = sys.argv[1]
train_csv = sys.argv[2] if len(sys.argv) > 2 else r"G:/ml/data/Chemical Phase dataset/public/train.csv"
tb = pd.read_csv(train_csv).interface_burden.values

def calib(v, ref):
    r = v.argsort().argsort(); return np.quantile(ref, (r + 0.5) / len(v))

oof = np.load(os.path.join(oof_dir, "oof.npz"))
done = oof["done"]; y = oof["y"][done]; T = float(oof["T"]); cen = oof["centers"]
pmf = softmax(oof["logits"][done] / T, 1); pexp = pmf @ cen
reg = np.clip(oof["reg"][done], 0, 100); blend = 0.5 * pexp + 0.5 * reg
print(f"n={done.sum()}  T*={T:.3f}  train_zones={np.bincount(to_zone(tb),minlength=4).tolist()}")
for nm, v in [("blend", blend), ("pmf_exp", pexp), ("reg", reg)]:
    c = calib(v, tb)
    print(f"  {nm:8s} raw={evaluate(y, np.clip(v,0,100)):.3f}  CALIBRATED={evaluate(y, c):.3f}  "
          f"zones={np.bincount(to_zone(c),minlength=4).tolist()}")
print(f"\n*** CALIBRATED honest (blend) = {evaluate(y, calib(blend, tb)):.3f} ***")
