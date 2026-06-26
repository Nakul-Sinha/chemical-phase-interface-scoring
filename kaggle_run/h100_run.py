"""H100 driver: leave-experiment-out 5-fold honest CONFIRM (calibrated, the LB analog)
for a given resolution, then full-data train + calibrated per-frame submission.
Run: DATA=/mnt/work/mychem/data OUT=/mnt/work/mychem/working RES=512 python h100_run.py
"""
import os, sys, time, glob
import numpy as np, pandas as pd
from scipy.special import softmax

DATA = os.environ.get("DATA", "/mnt/work/mychem/data")
OUT = os.environ.get("OUT", "/mnt/work/mychem/working")
RES = os.environ.get("RES", "512")
EP = int(os.environ.get("EP", "16"))
BS = int(os.environ.get("BS", "32"))
NS = int(os.environ.get("NS", "4"))
DO_FULL = os.environ.get("DO_FULL", "1") == "1"
DO_CV = os.environ.get("DO_CV", "1") == "1"
H = {"384": (384, 224), "512": (512, 320), "576": (576, 352), "640": (640, 384), "768": (768, 448)}[RES]
os.environ["DATA_ROOT"] = DATA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution_core as S
from metric import evaluate, to_zone

tb = pd.read_csv(os.path.join(DATA, "train.csv")).interface_burden.values
def calib(v, ref):
    r = v.argsort().argsort(); return np.quantile(ref, (r + 0.5) / len(v))

tag = f"nano{RES}"
cvdir = f"{OUT}/cv_{tag}"; fulldir = f"{OUT}/full_{tag}"
os.makedirs(cvdir, exist_ok=True); os.makedirs(fulldir, exist_ok=True)

def base(cfg):
    cfg.data_root = DATA; cfg.backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"
    cfg.img_h, cfg.img_w = H; cfg.epochs = EP; cfg.batch_size = BS; cfg.drop_path = 0.0
    cfg.fold_seed = 42; cfg.seed = 42; cfg.num_workers = 8

if DO_CV:
    cfg = S.Config(); base(cfg); cfg.out_dir = cvdir; cfg.n_folds = 5; cfg.cv_mode = "group"
    t = time.time(); print(f"=== {tag} 5-fold GROUP CV (ep{EP} bs{BS}) ===", flush=True)
    ec = S.run_cv(cfg)
    print(f"[CV {(time.time()-t)/60:.1f}min] expected-cost OOF={ec:.3f}", flush=True)
    oof = np.load(cvdir + "/oof.npz"); done = oof["done"]; y = oof["y"][done]
    T = float(oof["T"]); cen = oof["centers"]
    pmf = softmax(oof["logits"][done] / T, 1); pexp = pmf @ cen
    reg = np.clip(oof["reg"][done], 0, 100); blend = 0.5 * pexp + 0.5 * reg
    for nm, v in [("blend", blend), ("pmf_exp", pexp), ("reg", reg)]:
        print(f"  honest[{nm:7s}] raw={evaluate(y,np.clip(v,0,100)):.3f} CAL={evaluate(y,calib(v,tb)):.3f}", flush=True)
    print(f"\n*** CONFIRM {tag} 5-fold CALIBRATED = {evaluate(y, calib(blend, tb)):.3f} ***\n", flush=True)

if DO_FULL:
    import torch
    cfgf = S.Config(); base(cfgf); cfgf.out_dir = fulldir
    os.environ["FULL_SEEDS"] = ",".join(str(42 + i) for i in range(NS))
    print(f"=== {tag} full-data ({NS} seeds) ===", flush=True); S.train_full(cfgf)
    test = pd.read_csv(os.path.join(DATA, "test.csv")); dev = "cuda"; S._STORE = None
    mps = sorted(glob.glob(fulldir + "/model_*.pt")); print(f"predict {len(mps)} models", flush=True)
    val = np.zeros(len(test), np.float64); nv = 0
    for mp in mps:
        ck = torch.load(mp, map_location=dev); c = ck["cfg"]
        cc = S.Config(); cc.backbone = c["backbone"]; cc.img_h = c["img_h"]; cc.img_w = c["img_w"]
        cc.n_bins = c["n_bins"]; cc.data_root = DATA
        cen = np.linspace(c["bin_lo"], c["bin_hi"], c["n_bins"])
        m = S.Net(cc.backbone, cc.n_bins, False).to(dev); m.load_state_dict(ck["sd"]); m.eval()
        dl = torch.utils.data.DataLoader(S.ChemDataset(test, cc, False, None, None),
                                         batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        with torch.no_grad():
            ptr = 0
            for b in dl:
                x = b["x"].to(dev)
                for xx in (x, torch.flip(x, [3])):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        cl, rg = m(xx)
                    val[ptr:ptr+len(x)] += 0.5*(softmax(cl.float().cpu().numpy(),1)@cen) + 0.5*(torch.sigmoid(rg).float().cpu().numpy()*100)
                ptr += len(x)
        nv += 2
    raw = np.clip(val/nv, 0, 100); cal = calib(raw, tb)
    pd.DataFrame({"id": test.id, "interface_burden": raw}).to_csv(f"{OUT}/sub_{tag}_raw.csv", index=False)
    pd.DataFrame({"id": test.id, "interface_burden": cal}).to_csv(f"{OUT}/sub_{tag}_cal.csv", index=False)
    print(f"wrote sub_{tag}_cal.csv zones={np.bincount(to_zone(cal),minlength=4).tolist()} mean={cal.mean():.1f}", flush=True)
print("ALLDONE", flush=True)
