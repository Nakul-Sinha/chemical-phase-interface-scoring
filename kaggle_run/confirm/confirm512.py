"""Kaggle (P100-robust) confirmation kernel.
Goal: get the TRUE leave-experiment-out 5-fold honest number for nano512 (the run
that died with the H100), using the SAME blend+quantile-calibration the submission
uses -- this is the honest analog of the LB. Then train full-data + write a fresh
calibrated per-frame submission.

Prints `CONFIRM nano512 5-fold CALIBRATED = X` EARLY and flushes, so the number
survives even if the kernel is killed during the later full-data stage.
"""
import os, sys, subprocess, glob, time

RES = os.environ.get("RES", "512")            # 512 | 640 | 768
EP  = int(os.environ.get("EP", "16"))
BS  = int(os.environ.get("BS", "32"))
DO_FULL = os.environ.get("DO_FULL", "1") == "1"
NS  = int(os.environ.get("NS", "3"))          # full-data seeds
H = {"512": (512, 320), "576": (576, 352), "640": (640, 384), "768": (768, 448)}[RES]

# 1) P100-compatible torch BEFORE importing torch (Kaggle's default dropped sm_60)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.6.0", "torchvision==0.21.0",
                "--index-url", "https://download.pytorch.org/whl/cu124"], check=False)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
import torch.nn.functional as _F
_F.conv2d(torch.randn(1, 3, 8, 8, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
print("GPU conv kernel OK", flush=True)

# 2) resolve data + code
DATA = CODE = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "image_path" in open(os.path.join(root, "train.csv")).readline():
        DATA = root
    if "solution_core.py" in files:
        CODE = root
assert DATA and CODE, f"DATA={DATA} CODE={CODE}"
if not os.path.isdir(os.path.join(DATA, "images")):
    import zipfile, shutil
    dst = "/kaggle/tmp/data"; os.makedirs(dst, exist_ok=True)
    for c in ["train.csv", "test.csv", "sample_submission.csv"]:
        shutil.copy(os.path.join(DATA, c), dst)
    for z in glob.glob(os.path.join(DATA, "*.zip")):
        zipfile.ZipFile(z).extractall(dst)
    DATA = dst
os.environ["DATA_ROOT"] = DATA; os.environ["OUT_DIR"] = "/kaggle/working"
print("DATA", DATA, "| imgs:", len(glob.glob(DATA + "/images/*.jpg")), "| RES", RES, H, flush=True)
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from scipy.special import softmax
import solution_core as S
from metric import evaluate, to_zone

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
tb = train_df.interface_burden.values

def calibrate(v, ref):
    r = v.argsort().argsort()
    return np.quantile(ref, (r + 0.5) / len(v))

# 3) 5-fold leave-experiment-out CV (matches public)
cvdir = "/kaggle/working/cv"; os.makedirs(cvdir, exist_ok=True)
cfg = S.Config(); cfg.data_root = DATA; cfg.out_dir = cvdir
cfg.backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"
cfg.img_h, cfg.img_w = H
cfg.n_folds = 5; cfg.epochs = EP; cfg.batch_size = BS; cfg.drop_path = 0.0
cfg.cv_mode = "group"; cfg.fold_seed = 42; cfg.seed = 42; cfg.num_workers = 2
t0 = time.time()
print(f"=== nano{RES} 5-fold GROUP CV (ep={EP} bs={BS}) ===", flush=True)
ec_score = S.run_cv(cfg)
print(f"[CV done in {(time.time()-t0)/60:.1f} min] expected-cost OOF = {ec_score:.3f}", flush=True)

# 4) calibrated honest score (the LB analog)
oof = np.load(cvdir + "/oof.npz")
done = oof["done"]; y = oof["y"][done]; T = float(oof["T"]); centers = oof["centers"]
pmf = softmax(oof["logits"][done] / T, 1)
pmf_exp = pmf @ centers
reg = np.clip(oof["reg"][done], 0, 100)
for name, v in [("blend", 0.5 * pmf_exp + 0.5 * reg), ("pmf_exp", pmf_exp), ("reg", reg)]:
    cal = calibrate(v, tb)
    print(f"  honest[{name:7s}] raw={evaluate(y, np.clip(v,0,100)):.3f}  CALIBRATED={evaluate(y, cal):.3f}"
          f"  zones={np.bincount(to_zone(cal),minlength=4).tolist()}", flush=True)
hb = calibrate(0.5 * pmf_exp + 0.5 * reg, tb)
print(f"\n*** CONFIRM nano{RES} 5-fold CALIBRATED = {evaluate(y, hb):.3f} "
      f"(train zones {np.bincount(to_zone(tb),minlength=4).tolist()}) ***\n", flush=True)

# 5) full-data train + fresh calibrated submission
if DO_FULL:
    fulldir = "/kaggle/working/full"; os.makedirs(fulldir, exist_ok=True)
    cfgf = S.Config(); cfgf.data_root = DATA; cfgf.out_dir = fulldir
    cfgf.backbone = cfg.backbone; cfgf.img_h, cfgf.img_w = H
    cfgf.epochs = EP; cfgf.batch_size = BS; cfgf.drop_path = 0.0
    cfgf.fold_seed = 42; cfgf.num_workers = 2
    os.environ["FULL_SEEDS"] = ",".join(str(42 + i) for i in range(NS))
    print(f"=== full-data train ({NS} seeds) ===", flush=True)
    S.train_full(cfgf)
    # predict test (blend + hflip TTA) -> calibrate -> submission
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    dev = "cuda"
    S._STORE = None
    mps = sorted(glob.glob(fulldir + "/model_*.pt"))
    print(f"predicting test with {len(mps)} full-data models", flush=True)
    val = np.zeros(len(test), np.float64); nv = 0
    for mp in mps:
        ck = torch.load(mp, map_location=dev); c = ck["cfg"]
        cc = S.Config(); cc.backbone = c["backbone"]; cc.img_h = c["img_h"]; cc.img_w = c["img_w"]
        cc.n_bins = c["n_bins"]; cc.data_root = DATA
        cen = np.linspace(c["bin_lo"], c["bin_hi"], c["n_bins"])
        m = S.Net(cc.backbone, cc.n_bins, False).to(dev); m.load_state_dict(ck["sd"]); m.eval()
        dl = torch.utils.data.DataLoader(S.ChemDataset(test, cc, False, None, None),
                                         batch_size=48, shuffle=False, num_workers=2, pin_memory=True)
        with torch.no_grad():
            ptr = 0
            for b in dl:
                x = b["x"].to(dev)
                for xx in (x, torch.flip(x, [3])):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        cl, rg = m(xx)
                    exp = softmax(cl.float().cpu().numpy(), 1) @ cen
                    rgv = torch.sigmoid(rg).float().cpu().numpy() * 100
                    val[ptr:ptr + len(x)] += 0.5 * exp + 0.5 * rgv
                ptr += len(x)
        nv += 2
    raw = np.clip(val / nv, 0, 100)
    cal = calibrate(raw, tb)
    pd.DataFrame({"id": test.id, "interface_burden": raw}).to_csv("/kaggle/working/submission_raw.csv", index=False)
    pd.DataFrame({"id": test.id, "interface_burden": cal}).to_csv("/kaggle/working/submission.csv", index=False)
    print(f"wrote submission.csv (calibrated) zones={np.bincount(to_zone(cal),minlength=4).tolist()} "
          f"mean={cal.mean():.1f}", flush=True)
print("DONE", flush=True)
