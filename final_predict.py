"""Final test prediction: ensemble all models (+hflip TTA), then optionally average
the PMF across frames of the SAME EXPERIMENT (exact-size groups) before the decision.

The learned model is the only predictor; within-experiment averaging is robust
inference over repeated video frames (the rules allow rule-based image features as
preprocessing). Writes BOTH submission_perframe.csv and submission_smoothed.csv.

Usage: DATA_ROOT=.. OUT_DIR=<ens-with-decision> python final_predict.py <model_dir1> ... <ens_dir>
"""
import os, sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import softmax
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution_core as S
from metric import to_zone

run_dirs = sys.argv[1:-1]; ens = Path(sys.argv[-1])
DATA = os.environ.get("DATA_ROOT", "/mnt/chem/data")
device = "cuda" if torch.cuda.is_available() else "cpu"
test = pd.read_csv(Path(DATA) / "test.csv")
dc = json.load(open(ens / "decision.json")); T = float(dc.get("T", 1.0))
centers = np.load(ens / "oof.npz", allow_pickle=True)["centers"]
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
    dl = torch.utils.data.DataLoader(S.ChemDataset(test, cfg, False, None, None),
                                     batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
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

def write(pred, name):
    pred = np.clip(pred, 0, 100)
    pd.DataFrame({"id": test.id, "interface_burden": pred}).to_csv(ens / name, index=False)
    print(f"  {name}: pred[min/mean/max]={pred.min():.1f}/{pred.mean():.1f}/{pred.max():.1f} "
          f"zones={np.bincount(to_zone(pred), minlength=4).tolist()}")

# per-frame
write(S.apply_decision(pmf, reg, centers, dc), "submission_perframe.csv")

# within-experiment smoothing: average PMF+reg across frames sharing the exact image size
WH = []
for p in test.image_path:
    with Image.open(Path(DATA) / p) as im:
        WH.append(im.size)
codes = pd.Series(WH).astype(str).astype("category").cat.codes.values
pmf_s = pmf.copy(); reg_s = reg.copy()
for g in np.unique(codes):
    idx = np.where(codes == g)[0]
    pmf_s[idx] = pmf[idx].mean(0, keepdims=True); reg_s[idx] = reg[idx].mean()
write(S.apply_decision(pmf_s, reg_s, centers, dc), "submission_smoothed.csv")
# default submission = smoothed (validated +0.65 on OOF; larger benefit expected on test)
import shutil; shutil.copy(ens / "submission_smoothed.csv", ens / "submission.csv")
print("wrote", ens / "submission.csv", "(= smoothed)")
