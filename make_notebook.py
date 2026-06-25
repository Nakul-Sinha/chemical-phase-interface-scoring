"""Generate a self-contained solution.ipynb from metric.py + solution_core.py.

The notebook embeds the metric and the full pipeline as cells (import lines to the
sibling .py files are stripped, since everything lives in one namespace), then a
runner cell that trains the multi-seed ensemble and writes ./working/submission.csv
from ./dataset/public. A FAST flag gives a quick single-fold smoke for validation.
"""
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent

def strip(src, drop_main=True):
    lines = src.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("sys.path.insert"): continue
        if re.match(r"\s*from metric import", ln): continue
        if drop_main and ln.startswith('if __name__ == "__main__":'):
            break
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"

metric_src = strip((REPO / "metric.py").read_text())
core_src = strip((REPO / "solution_core.py").read_text())
ens_oof = strip((REPO / "ensemble_oof.py").read_text(), drop_main=False)
ens_pred = strip((REPO / "ensemble_predict.py").read_text(), drop_main=False)

intro = """# Chemical Phase Interface Assessment — Solution

Seed-free **leave-experiment-out** image-regression for an ordinal `interface_burden` (0–100),
scored ~80% on the ordinal burden ZONE. Pipeline (see `Approach.md`):
- **Honest CV** by inferred experiment groups (exact image size ≈ experiment id; used ONLY for folds).
- **ConvNeXt-V2 (nano + femto) + GeM**, dual **SORD ordinal-PMF + BCE-regression** heads.
- **Bayes-optimal expected-cost decision** over the exact metric (+ temperature calibration).
- **Multi-seed ensemble** (per-fold variance is large) + hflip TTA. Aggressive color/style aug HURTS
  here (color/intensity is the genuine turbidity signal), so augmentation is deliberately light.

Reads `./dataset/public/`, writes `./working/submission.csv`. Set `FAST=1` for a quick smoke.
"""

runner = '''import os
os.environ.setdefault("DATA_ROOT", "./dataset/public")
os.environ.setdefault("OUT_DIR", "./working")
FAST = os.environ.get("FAST", "0") == "1"
DATA = os.environ["DATA_ROOT"]; OUT = os.environ["OUT_DIR"]
Path(OUT).mkdir(parents=True, exist_ok=True)

# config: light, well-tuned base configs (bigger/aggressive-aug overfit the experiments)
BASE = dict(N_FOLDS=5, IMG_H=320, IMG_W=192, EPOCHS=18, BATCH=32, NUM_WORKERS=8, FOLD_SEED=42)
CONFIGS = [("convnextv2_nano.fcmae_ft_in22k_in1k", 18), ("convnextv2_femto.fcmae_ft_in1k", 20)]
SEEDS = [42, 43, 44]
if FAST:
    CONFIGS = [("convnextv2_nano.fcmae_ft_in22k_in1k", 2)]; SEEDS = [42]; BASE["N_FOLDS"] = 3

run_dirs = []
for bb, ep in CONFIGS:
    for sd in SEEDS:
        for k, v in BASE.items(): os.environ[k] = str(v)
        os.environ["BACKBONE"] = bb; os.environ["EPOCHS"] = str(ep); os.environ["SEED"] = str(sd)
        d = f"{OUT}/{bb.split('.')[0]}_s{sd}"; os.environ["OUT_DIR"] = d
        cfg = Config()
        print(f"=== {bb} seed {sd} ===")
        run_cv(cfg)
        run_dirs.append(d)
os.environ["OUT_DIR"] = OUT

# ensemble OOF -> honest decision -> ensemble test prediction
import numpy as np, pandas as pd, glob, torch
from scipy.special import softmax
oofs = [np.load(Path(d)/"oof.npz") for d in run_dirs]
pmf = np.mean([softmax(o["logits"],1) for o in oofs],0); reg = np.mean([o["reg"] for o in oofs],0)
o0 = oofs[0]; ens = Path(OUT)
np.savez(ens/"oof.npz", logits=np.log(pmf+1e-12).astype(np.float32), reg=reg.astype(np.float32),
         fold=o0["fold"], y=o0["y"], done=o0["done"], centers=o0["centers"], T=np.float64(1.0), ids=o0["ids"])
# honest decision (reuse decision logic)
%run_decision_opt%

# ensemble test predict (mixed backbones via each ckpt's cfg)
import json as _json
dc = _json.load(open(ens/"decision.json")); T = float(dc.get("T",1.0)); centers = o0["centers"]
test = pd.read_csv(Path(DATA)/"test.csv"); _STORE = None
mps = [m for d in run_dirs for m in sorted(glob.glob(f"{d}/model_f*.pt"))]
dev = "cuda" if torch.cuda.is_available() else "cpu"
ps = np.zeros((len(test),len(centers))); rs = np.zeros(len(test)); nv = 0
for mp in mps:
    ck = torch.load(mp, map_location=dev); c = ck["cfg"]
    cf = Config(); cf.backbone=c["backbone"]; cf.img_h=c["img_h"]; cf.img_w=c["img_w"]; cf.n_bins=c["n_bins"]; cf.data_root=DATA
    md = Net(cf.backbone, cf.n_bins, False).to(dev); md.load_state_dict(ck["sd"]); md.eval()
    dl = DataLoader(ChemDataset(test, cf, False, None, None), batch_size=64, shuffle=False, num_workers=8)
    with torch.no_grad():
        ptr=0
        for b in dl:
            x=b["x"].to(dev)
            for xx in (x, torch.flip(x,[3])):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev=="cuda"):
                    cl,rg = md(xx)
                ps[ptr:ptr+len(x)] += softmax(cl.float().cpu().numpy()/T,1)
                rs[ptr:ptr+len(x)] += torch.sigmoid(rg).float().cpu().numpy()*100
            ptr+=len(x)
    nv+=2
pmf_t = ps/ps.sum(1,keepdims=True); reg_t = rs/nv
pred = np.clip(apply_decision(pmf_t, reg_t, centers, dc), 0, 100)
pd.DataFrame({"id":test.id,"interface_burden":pred}).to_csv(Path(OUT)/"submission.csv", index=False)
print("wrote", Path(OUT)/"submission.csv", "| pred zones", np.bincount(to_zone(pred),minlength=4).tolist())
'''

# inline the decision-opt (honest) so the notebook is self-contained
dec_src = strip((REPO / "decision_opt.py").read_text(), drop_main=False)
dec_src = dec_src.replace('OUT = Path(os.environ.get("OUT_DIR", "working"))', 'OUT = Path(os.environ["OUT_DIR"])')
runner = runner.replace("%run_decision_opt%", "exec(_DECISION_OPT_SRC)")

def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}

nb = {"cells": [
    md(intro),
    code(metric_src),
    code(core_src),
    code("_DECISION_OPT_SRC = " + repr(dec_src)),
    code(runner),
    code('# strict validation\n'
         'import pandas as pd, numpy as np\n'
         'sub = pd.read_csv(Path(OUT)/"submission.csv"); samp = pd.read_csv(Path(DATA)/"sample_submission.csv")\n'
         'assert list(sub.columns)==["id","interface_burden"]; assert len(sub)==len(samp)\n'
         'assert sub.id.is_unique and set(sub.id)==set(samp.id)\n'
         'v=sub.interface_burden.values; assert np.isfinite(v).all() and (v>=0).all() and (v<=100).all()\n'
         'print("submission valid:", sub.shape)'),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

(REPO / "solution.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote solution.ipynb (cells:", len(nb["cells"]), ")")
