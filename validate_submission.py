"""Strict validator for submission.csv (run before any official submission)."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"G:/ml/data/Chemical Phase dataset/public"))
sub_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("working/submission.csv")

sub = pd.read_csv(sub_path)
sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")

assert list(sub.columns) == ["id", "interface_burden"], f"cols must be [id,interface_burden], got {list(sub.columns)}"
assert len(sub) == len(test) == 1261, f"row count {len(sub)} != {len(test)} (expect 1261)"
assert sub.id.is_unique, "duplicate ids"
assert set(sub.id) == set(test.id), "id set mismatch vs test.csv"
v = sub.interface_burden.to_numpy()
assert np.isfinite(v).all(), "non-finite values"
assert (v >= 0).all() and (v <= 100).all(), f"values out of [0,100]: min={v.min()} max={v.max()}"

from sys import path as _p; _p.insert(0, str(Path(__file__).resolve().parent))
from metric import to_zone
print("VALID:", sub.shape)
print("pred min/mean/max:", round(float(v.min()), 2), round(float(v.mean()), 2), round(float(v.max()), 2))
print("pred zone dist:", np.bincount(to_zone(v), minlength=4).tolist())
