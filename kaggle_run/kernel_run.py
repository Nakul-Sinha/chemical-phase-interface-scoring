"""Kaggle kernel entry: train 5-fold CV + tune decision + predict test.

Reads dataset 'chem-phase-interface' (+ code dataset 'chem-code'). Writes
/kaggle/working/{oof.npz, model_f*.pt, decision.json, submission.csv}.
Edit CONFIG below per experiment. Offline weights: add a 'chem-weights' dataset
(a HF hub cache) and this sets HF_HOME + HF_HUB_OFFLINE automatically.
"""
import os, sys, glob, zipfile, shutil, runpy

# -------- CONFIG (edit per experiment) --------
CONFIG = {
    "BACKBONE": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "IMG_H": "384", "IMG_W": "224",
    "EPOCHS": "18", "BATCH": "24", "LR": "2.5e-4",
    "N_FOLDS": "5", "NUM_WORKERS": "2", "SEED": "42",
    "CMIX_P": "0.5", "CJ": "0.3", "GRAY_P": "0.1", "REG_W": "0.3",
}
for k, v in CONFIG.items():
    os.environ[k] = v

# -------- data resolution (robust: walk /kaggle/input to find train.csv anywhere) --------
DATA = None; CODE = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "image_path" in open(os.path.join(root, "train.csv")).readline():
        DATA = root
    if "solution_core.py" in files:
        CODE = root
print("DATA dir:", DATA, "| CODE dir:", CODE)
assert DATA, "could not locate train.csv under /kaggle/input"
if not os.path.isdir(os.path.join(DATA, "images")):
    dst = "/kaggle/temp/data"; os.makedirs(dst, exist_ok=True)
    for c in ["train.csv", "test.csv", "sample_submission.csv"]:
        shutil.copy(os.path.join(DATA, c), dst)
    zips = glob.glob(os.path.join(DATA, "*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dst)
    DATA = dst
os.environ["DATA_ROOT"] = DATA
os.environ["OUT_DIR"] = "/kaggle/working"
print("DATA_ROOT:", DATA, "| has images/:", os.path.isdir(os.path.join(DATA, "images")))

# -------- offline weights (optional: a dataset that contains a 'hub' folder) --------
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.basename(root) == "hub" and any(d.startswith("models--timm") for d in dirs):
        os.environ["HF_HOME"] = os.path.dirname(root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("using offline HF cache:", os.path.dirname(root)); break

# -------- code --------
assert CODE, "could not locate solution_core.py under /kaggle/input"
sys.path.insert(0, CODE)
import solution_core as S
cfg = S.Config()
print("CFG:", {k: getattr(cfg, k) for k in ["backbone", "img_h", "img_w", "epochs", "batch_size", "n_folds"]})

S.run_cv(cfg)
runpy.run_path(os.path.join(CODE, "decision_opt.py"), run_name="__main__")
S.predict_test(cfg)
print("DONE — submission at /kaggle/working/submission.csv")
