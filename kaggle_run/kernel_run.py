"""Kaggle kernel: P100-robust env + screen model configs by CV, then train the
winner full-data + within-experiment smoothing -> submission.

Kaggle's preinstalled torch dropped Pascal sm_60 (P100), so we reinstall a
compatible torch FIRST (works on P100 and T4). Reads chem-phase-interface +
chem-code datasets; writes /kaggle/working/submission.csv.
"""
import os, sys, subprocess, glob, json, time

# -------- 1) P100-compatible torch: install BEFORE importing torch (re-import
# won't pick up a new build in-process). Kaggle's default torch lacks sm_60 (P100).
MODE = os.environ.get("RUNMODE", "screen")   # smoke | screen | final
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.6.0", "torchvision==0.21.0",
                "--index-url", "https://download.pytorch.org/whl/cu124"], check=False)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# trigger a real conv kernel to confirm the GPU arch is supported
import torch.nn.functional as _F
_F.conv2d(torch.randn(1, 3, 8, 8, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
print("GPU conv kernel OK")

# -------- 2) data + code resolution --------
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
print("DATA", DATA, "| imgs:", len(glob.glob(DATA + "/images/*.jpg")))
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import solution_core as S
from scipy.special import softmax

def cv(name, **cfgkw):
    out = f"/kaggle/working/{name}"; os.makedirs(out, exist_ok=True)
    cfg = S.Config(); cfg.data_root = DATA; cfg.out_dir = out; cfg.fold_seed = 42; cfg.num_workers = 2
    for k, v in cfgkw.items(): setattr(cfg, k, v)
    t = time.time(); score = S.run_cv(cfg); print(f"[{name}] CV={score:.3f} ({time.time()-t:.0f}s)")
    return score

if MODE == "smoke":
    cv("smoke", backbone="convnextv2_nano.fcmae_ft_in22k_in1k", img_h=320, img_w=192,
       n_folds=5, folds_to_run="0", epochs=3, batch_size=32)
    print("SMOKE OK")
elif MODE == "screen":
    # 1-seed, 3-fold quick screen of genuinely different directions vs nano320 (~25)
    res = {}
    res["nano320"] = cv("nano320", backbone="convnextv2_nano.fcmae_ft_in22k_in1k", img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=32, drop_path=0.0)
    res["nano512"] = cv("nano512", backbone="convnextv2_nano.fcmae_ft_in22k_in1k", img_h=512, img_w=288, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=16, drop_path=0.0)
    res["base320"] = cv("base320", backbone="convnextv2_base.fcmae_ft_in22k_in1k", img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=16, drop_path=0.2)
    res["tiny384"] = cv("tiny384", backbone="convnextv2_tiny.fcmae_ft_in22k_in1k", img_h=384, img_w=224, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=24, drop_path=0.1)
    print("SCREEN RESULTS:", {k: round(v, 3) for k, v in res.items()})
    json.dump(res, open("/kaggle/working/screen.json", "w"))
print("DONE")
