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
elif MODE == "screen2":
    # genuinely-different paradigms: foundation features + different CNN family
    res = {}
    res["effv2s"] = cv("effv2s", backbone="tf_efficientnetv2_s.in21k_ft_in1k", img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=24, drop_path=0.1)
    res["dinov2b"] = cv("dinov2b", backbone="vit_base_patch14_reg4_dinov2.lvd142m", img_h=322, img_w=210, n_folds=5, folds_to_run="0,1,2", epochs=12, batch_size=12, drop_path=0.1, lr=1e-4, head_lr_mult=10.0)
    print("SCREEN2 RESULTS:", {k: round(v, 3) for k, v in res.items()})
    json.dump(res, open("/kaggle/working/screen2.json", "w"))
elif MODE == "screen3":
    # colour-invariance: reagent colour is the OOD nuisance; turbidity is brightness/texture
    nb = "convnextv2_nano.fcmae_ft_in22k_in1k"
    res = {}
    res["nano320"] = cv("s3_nano320", backbone=nb, img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=32)
    res["nano_hue"] = cv("s3_huejit", backbone=nb, img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=32, color_jitter=0.2, hue_jitter=0.5)
    res["nano_gray"] = cv("s3_gray", backbone=nb, img_h=320, img_w=192, n_folds=5, folds_to_run="0,1,2", epochs=16, batch_size=32, gray_input=True)
    print("SCREEN3 RESULTS:", {k: round(v, 3) for k, v in res.items()})
    json.dump(res, open("/kaggle/working/screen3.json", "w"))
elif MODE == "final":
    import runpy
    BB = os.environ.get("FBB", "convnextv2_nano.fcmae_ft_in22k_in1k")
    H, W = int(os.environ.get("FH", "320")), int(os.environ.get("FW", "192"))
    BS, DP, EP = int(os.environ.get("FBS", "32")), float(os.environ.get("FDP", "0.0")), int(os.environ.get("FEP", "18"))
    NS = int(os.environ.get("FSEEDS", "6"))
    common = dict(backbone=BB, img_h=H, img_w=W, batch_size=BS, drop_path=DP, epochs=EP, n_folds=5)
    # 1) 5-fold CV for OOF + honest decision
    cvdir = "/kaggle/working/cv"; cfg = S.Config(); cfg.data_root = DATA; cfg.out_dir = cvdir
    cfg.fold_seed = 42; cfg.seed = 42; cfg.num_workers = 2
    for k, v in common.items(): setattr(cfg, k, v)
    print("=== CV (for OOF+decision) ==="); S.run_cv(cfg)
    os.environ["OUT_DIR"] = cvdir; runpy.run_path(CODE + "/decision_opt.py", run_name="__main__")
    # 2) full-data models (stronger than 80% folds)
    fulldir = "/kaggle/working/full"; cfgf = S.Config(); cfgf.data_root = DATA; cfgf.out_dir = fulldir
    cfgf.fold_seed = 42; cfgf.num_workers = 2
    for k, v in common.items(): setattr(cfgf, k, v)
    os.environ["FULL_SEEDS"] = ",".join(str(42 + i) for i in range(NS))
    print("=== full-data train ==="); S.train_full(cfgf)
    # 3) ensemble predict (full + fold models) + within-experiment smoothing -> submission
    os.environ["DATA_ROOT"] = DATA
    sys.argv = ["final_predict.py", fulldir, cvdir, cvdir]   # models from full+cv; decision/oof in cvdir
    runpy.run_path(CODE + "/final_predict.py", run_name="__main__")
    import shutil
    src = cvdir + "/submission_smoothed.csv"
    if os.path.exists(src): shutil.copy(src, "/kaggle/working/submission.csv")
    print("FINAL submission written")
print("DONE")
