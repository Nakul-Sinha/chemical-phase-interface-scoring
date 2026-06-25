"""Average OOF predictions across runs that share the SAME fold split (FOLD_SEED),
producing an ensemble oof.npz (logits = log of the mean softmax-PMF, so the existing
logit-based decision code applies unchanged via softmax(logits/T) = pmf^(1/T)).

Usage: python ensemble_oof.py <run_dir1> <run_dir2> ... <ENS_OUT_DIR>
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import softmax

run_dirs = [Path(d) for d in sys.argv[1:-1]]
ens = Path(sys.argv[-1]); ens.mkdir(parents=True, exist_ok=True)
oofs = [np.load(d / "oof.npz") for d in run_dirs]
# sanity: same fold split + labels
for o in oofs[1:]:
    assert np.array_equal(o["fold"], oofs[0]["fold"]), "fold split mismatch — runs must share FOLD_SEED"
pmf = np.mean([softmax(o["logits"], 1) for o in oofs], 0)
reg = np.mean([o["reg"] for o in oofs], 0)
o0 = oofs[0]
logits = np.log(pmf + 1e-12).astype(np.float32)
np.savez(ens / "oof.npz", logits=logits, reg=reg.astype(np.float32),
         fold=o0["fold"], y=o0["y"], done=o0["done"], centers=o0["centers"], T=np.float64(1.0), ids=o0["ids"])
print(f"ensembled {len(run_dirs)} OOFs -> {ens/'oof.npz'}  (n={len(reg)})")
