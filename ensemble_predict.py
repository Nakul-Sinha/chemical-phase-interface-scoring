"""Ensemble test prediction across many models (possibly different backbones/sizes).
Each checkpoint carries its own cfg, so we build the right Net + input size per model.
Averages softmax-PMF (+reg) over all models x hflip-TTA, then applies the tuned decision.

Usage: python ensemble_predict.py <run_dir1> ... <ENS_DIR>   (ENS_DIR holds decision.json + oof.npz)
"""
import os, sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import softmax
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution_core as S
from metric import to_zone

run_dirs = sys.argv[1:-1]; ens = Path(sys.argv[-1])
DATA = os.environ.get("DATA_ROOT", "/mnt/chem/data")
device = "cuda" if torch.cuda.is_available() else "cpu"
test = pd.read_csv(Path(DATA) / "test.csv")
dc = json.load(open(ens / "decision.json"))
T = float(dc.get("T", 1.0))
centers = np.load(ens / "oof.npz")["centers"]
model_paths = []
for d in run_dirs:
    model_paths += sorted(glob.glob(f"{d}/model_f*.pt"))
print(f"ensembling {len(model_paths)} models over {len(test)} test images, T={T}")

S._STORE = None
pmf_sum = np.zeros((len(test), len(centers)), np.float64); reg_sum = np.zeros(len(test), np.float64); nv = 0
for mp in model_paths:
    ckpt = torch.load(mp, map_location=device); c = ckpt["cfg"]
    cfg = S.Config(); cfg.backbone = c["backbone"]; cfg.img_h = c["img_h"]; cfg.img_w = c["img_w"]
    cfg.n_bins = c["n_bins"]; cfg.data_root = DATA
    model = S.Net(cfg.backbone, cfg.n_bins, False).to(device); model.load_state_dict(ckpt["sd"]); model.eval()
    ds = S.ChemDataset(test, cfg, False, None, None)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    with torch.no_grad():
        ptr = 0
        for b in dl:
            x = b["x"].to(device)
            for xx in (x, torch.flip(x, [3])):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    cl, rg = model(xx)
                pmf_sum[ptr:ptr + len(x)] += softmax(cl.float().cpu().numpy() / T, 1)
                reg_sum[ptr:ptr + len(x)] += torch.sigmoid(rg).float().cpu().numpy() * 100
            ptr += len(x)
    nv += 2
pmf = pmf_sum / pmf_sum.sum(1, keepdims=True); reg = reg_sum / nv
pred = np.clip(S.apply_decision(pmf, reg, centers, dc), 0, 100)
sub = pd.DataFrame({"id": test.id, "interface_burden": pred})
sub.to_csv(ens / "submission.csv", index=False)
print(f"wrote {ens/'submission.csv'} rows={len(sub)} pred[min/mean/max]={pred.min():.1f}/{pred.mean():.1f}/{pred.max():.1f}")
print("pred zone dist:", np.bincount(to_zone(pred), minlength=4).tolist())
