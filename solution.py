import os, sys, glob, json
import numpy as np, pandas as pd, torch
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution_core as S
from metric import expected_cost_decision, to_zone

DATA = os.environ.get("DATA_ROOT", "dataset/public")
OUT = os.environ.get("OUT_DIR", "working")
FULL_SEEDS = int(os.environ.get("FULL_SEEDS_N", "10"))
EP = int(os.environ.get("EP", "18"))
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def mkcfg(out_dir):
    cfg = S.Config()
    cfg.data_root = DATA
    cfg.out_dir = out_dir
    cfg.backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"
    cfg.img_h = 320
    cfg.img_w = 192
    cfg.batch_size = 32
    cfg.epochs = EP
    cfg.drop_path = 0.0
    cfg.num_workers = 0
    cfg.cache = True
    cfg.fold_seed = 42
    cfg.seed = 42
    return cfg


cvdir = os.path.join(OUT, "cv")
fulldir = os.path.join(OUT, "full")
os.makedirs(cvdir, exist_ok=True)
os.makedirs(fulldir, exist_ok=True)

cfg_cv = mkcfg(cvdir)
cfg_cv.n_folds = 5
cfg_cv.cv_mode = "group"
S.run_cv(cfg_cv)
oof = np.load(os.path.join(cvdir, "oof.npz"))
T = float(oof["T"])

cfg_full = mkcfg(fulldir)
os.environ["FULL_SEEDS"] = ",".join(str(42 + i) for i in range(FULL_SEEDS))
S.train_full(cfg_full)

test = pd.read_csv(os.path.join(DATA, "test.csv"))
centers = cfg_full.centers
S._STORE = None
cache = {}
models = sorted(glob.glob(fulldir + "/model_full_s*.pt")) + sorted(glob.glob(cvdir + "/model_f*.pt"))
assert models, "no models"
pmf_sum = np.zeros((len(test), cfg_full.n_bins), np.float64)
for mp in models:
    ck = torch.load(mp, map_location=dev)
    c = ck["cfg"]
    cc = S.Config()
    cc.backbone = c["backbone"]
    cc.img_h = c["img_h"]
    cc.img_w = c["img_w"]
    cc.n_bins = c["n_bins"]
    cc.data_root = DATA
    cen = np.linspace(c["bin_lo"], c["bin_hi"], c["n_bins"])
    m = S.Net(cc.backbone, cc.n_bins, False).to(dev)
    m.load_state_dict(ck["sd"])
    m.eval()
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

if SMOOTH:
    groups = S.compute_groups(test, DATA, tol_abs=2)
    pmf_g = pmf.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            pmf_g[idx] = pmf[idx].mean(0, keepdims=True)
    pred = np.clip(expected_cost_decision(pmf_g, centers), 0, 100)
else:
    pred = pred_pf
pd.DataFrame({"id": test.id, "interface_burden": pred}).to_csv("submission.csv", index=False)
print(f"wrote submission.csv models={len(models)} mean={pred.mean():.1f} zones={np.bincount(to_zone(pred), minlength=4).tolist()}", flush=True)
