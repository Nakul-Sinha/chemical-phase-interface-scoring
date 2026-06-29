"""Ensemble predict over full-data + fold models, expected-cost decision, and an
OPTIONAL within-experiment smoothing pass. Writes:
  submission_perframe.csv  -- pure per-frame learned-model output (always-compliant)
  submission_smoothed.csv  -- + within-experiment consensus (group frames, avg PMF)

Grouping for smoothing: GROUP_MODE=size (compute_groups) or =emb (visual similarity,
fully pixels-based -> compliant). Decision T from CV decision.json if present.
Env: DATA_ROOT, FULL_DIR, CV_DIR (optional), T, GROUP_MODE, EMB_THR.
"""
import os, sys, glob, json
import numpy as np, pandas as pd, torch
from scipy.special import softmax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution_core as S
from metric import expected_cost_decision, to_zone

DATA = os.environ.get("DATA_ROOT", r"G:/ml/data/Chemical Phase dataset/public")
FULL_DIR = os.environ.get("FULL_DIR", "working/full320")
CV_DIR = os.environ.get("CV_DIR", "")
GROUP_MODE = os.environ.get("GROUP_MODE", "size")
EMB_THR = float(os.environ.get("EMB_THR", "0.93"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = S.Config(); cfg.data_root = DATA
cfg.backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"; cfg.img_h = 320; cfg.img_w = 192
centers = cfg.centers
T = float(os.environ.get("T", "2.75"))
if CV_DIR and os.path.exists(os.path.join(CV_DIR, "decision.json")):
    dc = json.load(open(os.path.join(CV_DIR, "decision.json")))
    T = float(dc.get("T", T)); print("using CV decision T=", T)

test = pd.read_csv(os.path.join(DATA, "test.csv"))
models = sorted(glob.glob(FULL_DIR + "/model_full_s*.pt"))
if CV_DIR:
    models += sorted(glob.glob(CV_DIR + "/model_f*.pt"))
assert models, "no models"
print(f"ensemble of {len(models)} models: {[os.path.basename(m) for m in models]}")

S._STORE = None
cache = {}
pmf_sum = np.zeros((len(test), cfg.n_bins), np.float64)
for mp in models:
    ck = torch.load(mp, map_location=dev); c = ck["cfg"]
    cc = S.Config(); cc.backbone = c["backbone"]; cc.img_h = c["img_h"]; cc.img_w = c["img_w"]
    cc.n_bins = c["n_bins"]; cc.data_root = DATA
    cen = np.linspace(c["bin_lo"], c["bin_hi"], c["n_bins"])
    m = S.Net(cc.backbone, cc.n_bins, False).to(dev); m.load_state_dict(ck["sd"]); m.eval()
    dl = torch.utils.data.DataLoader(S.ChemDataset(test, cc, False, None, cache),
                                     batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        ptr = 0
        for b in dl:
            x = b["x"].to(dev)
            for xx in (x, torch.flip(x, [3])):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    cl, _ = m(xx)
                pmf_sum[ptr:ptr + len(x)] += softmax(cl.float().cpu().numpy() / T, 1)
            ptr += len(x)
pmf = pmf_sum / pmf_sum.sum(1, keepdims=True)

pred_pf = np.clip(expected_cost_decision(pmf, centers), 0, 100)
pd.DataFrame({"id": test.id, "interface_burden": pred_pf}).to_csv("submission_perframe.csv", index=False)
print(f"per-frame: mean={pred_pf.mean():.1f} zones={np.bincount(to_zone(pred_pf), minlength=4).tolist()}")

if GROUP_MODE == "emb":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda"))
    import timm
    from PIL import Image
    em = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval().to(dev)
    dcfg = timm.data.resolve_data_config({}, model=em); tf = timm.data.create_transform(**dcfg)
    feats = []
    with torch.no_grad():
        for p in test.image_path:
            x = tf(Image.open(os.path.join(DATA, p)).convert("RGB")).unsqueeze(0).to(dev)
            feats.append(em(x).float().cpu().numpy())
    E = np.concatenate(feats, 0); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    parent = list(range(len(E)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    sims = E @ E.T
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            if sims[i, j] >= EMB_THR:
                parent[find(i)] = find(j)
    groups = np.array([find(i) for i in range(len(E))])
    _, groups = np.unique(groups, return_inverse=True)
else:
    groups = S.compute_groups(test, DATA, tol_abs=2)

pmf_g = pmf.copy()
for g in np.unique(groups):
    idx = np.where(groups == g)[0]
    if len(idx) > 1:
        pmf_g[idx] = pmf[idx].mean(0, keepdims=True)
pred_sm = np.clip(expected_cost_decision(pmf_g, centers), 0, 100)
pd.DataFrame({"id": test.id, "interface_burden": pred_sm}).to_csv("submission_smoothed.csv", index=False)
ng = len(np.unique(groups)); multi = sum(1 for g in np.unique(groups) if (groups == g).sum() > 1)
print(f"smoothed[{GROUP_MODE}]: mean={pred_sm.mean():.1f} zones={np.bincount(to_zone(pred_sm), minlength=4).tolist()} "
      f"| {ng} groups, {multi} multi-frame, {(groups!=np.arange(len(groups))).sum()} frames grouped")
