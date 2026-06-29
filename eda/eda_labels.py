"""EDA: label/zone distribution, metric baselines, image properties, montages.

Run: python eda/eda_labels.py
Writes outputs to eda/out/.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from metric import evaluate, evaluate_components, to_zone, best_constant, SEVERITY_BINS, ZONE_CENTERS

DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"G:/ml/data/Chemical Phase dataset/public"))
OUT = REPO / "eda" / "out"
OUT.mkdir(parents=True, exist_ok=True)

def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

train = pd.read_csv(DATA_ROOT / "train.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")
sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); log_lines.append(s)

section("SHAPES / COLUMNS")
log("train:", train.shape, list(train.columns))
log("test :", test.shape, list(test.columns))
log("sample:", sample.shape, list(sample.columns))
log("train ids unique:", train.id.is_unique, "| test ids unique:", test.id.is_unique)
log("train/test id overlap:", len(set(train.id) & set(test.id)))

y = train.interface_burden.values.astype(float)
section("LABEL DISTRIBUTION (interface_burden)")
log("count:", len(y))
log("min/max:", y.min(), y.max())
log("mean/median/std:", round(y.mean(),3), round(np.median(y),3), round(y.std(),3))
for q in [1,5,10,25,50,75,90,95,99]:
    log(f"  p{q:02d}:", round(np.percentile(y, q),3))

section("ZONE DISTRIBUTION (bins [0,12,35,48,100])")
tz = to_zone(y)
zone_names = ["Z0 [0,12)", "Z1 [12,35)", "Z2 [35,48)", "Z3 [48,100]"]
for z in range(4):
    n = int((tz==z).sum())
    log(f"  {zone_names[z]}: {n}  ({100*n/len(y):.2f}%)")
log("class imbalance ratio (max/min):", round((np.bincount(tz,minlength=4).max())/(np.bincount(tz,minlength=4).min()+1e-9),2))

section("SAMPLE_SUBMISSION baseline values (the 'random baseline')")
sv = sample.interface_burden.values.astype(float)
log("sample sub burden min/mean/max:", round(sv.min(),2), round(sv.mean(),2), round(sv.max(),2))
log("sample sub zone dist:", np.bincount(to_zone(sv), minlength=4).tolist())

section("METRIC BASELINES (evaluated on TRAIN labels, no CV) -- lower is better")
# These tell us the trivial floor/ceiling of the metric landscape.
def show(name, pred):
    c = evaluate_components(y, pred)
    log(f"  {name:34s} score={c['total']:.4f}  (zone={c['w_zone']:.3f} abs={c['w_absolute']:.3f} high={c['w_high']:.3f} ext={c['w_extreme']:.3f}, zone_acc={c['zone_accuracy']:.3f})")
show("predict mean", np.full_like(y, y.mean()))
show("predict median", np.full_like(y, np.median(y)))
for z,c in enumerate(ZONE_CENTERS):
    show(f"predict zone{z} center ({c})", np.full_like(y, c))
bc_val, bc_score = best_constant(y)
show(f"BEST CONSTANT ({bc_val:.2f})", np.full_like(y, bc_val))
# Oracle: perfect zone (predict the center of the TRUE zone) -- upper bound from zone term alone
oracle_zone = ZONE_CENTERS[tz]
show("ORACLE perfect-zone centers", oracle_zone)
show("ORACLE perfect (y_true)", y)
# "What if we predict true zone center but are 1 zone too low/high everywhere" (sanity)
log("\n  NOTE: zone term = 0.80*mean(0/60/100). Perfect zone => ~0.0 + tiny abs terms.")

section("IMAGE PROPERTIES (sample up to 400 train+test images)")
allrows = pd.concat([train[["id","image_path"]], test[["id","image_path"]]], ignore_index=True)
rng = np.random.default_rng(0)
idx = rng.choice(len(allrows), size=min(400, len(allrows)), replace=False)
props = []
for i in idx:
    p = DATA_ROOT / allrows.image_path.iloc[i]
    try:
        with Image.open(p) as im:
            props.append((im.size[0], im.size[1], im.mode, os.path.getsize(p)))
    except Exception as e:
        props.append((None, None, f"ERR:{e}", 0))
pdf = pd.DataFrame(props, columns=["w","h","mode","bytes"]).dropna()
log("modes:", pdf['mode'].value_counts().to_dict())
log("width  min/median/max:", pdf.w.min(), int(pdf.w.median()), pdf.w.max())
log("height min/median/max:", pdf.h.min(), int(pdf.h.median()), pdf.h.max())
log("unique (w,h) top:", pdf.groupby(['w','h']).size().sort_values(ascending=False).head(8).to_dict())
ar = (pdf.w/pdf.h)
log("aspect ratio min/median/max:", round(ar.min(),3), round(ar.median(),3), round(ar.max(),3))
log("filesize KB median:", int(pdf.bytes.median()/1024))

# grayscale check: is channel 0==1==2?
section("COLOR vs GRAYSCALE check (20 imgs)")
gray_like = 0
for i in idx[:20]:
    p = DATA_ROOT / allrows.image_path.iloc[i]
    a = np.asarray(Image.open(p).convert("RGB")).astype(int)
    if np.abs(a[...,0]-a[...,1]).mean() < 2 and np.abs(a[...,1]-a[...,2]).mean() < 2:
        gray_like += 1
log(f"  near-grayscale: {gray_like}/20")

# ---- plots ----
fig, ax = plt.subplots(1,2, figsize=(13,4))
ax[0].hist(y, bins=60, color="steelblue", edgecolor="k", lw=0.3)
for b in SEVERITY_BINS[1:-1]:
    ax[0].axvline(b, color="red", ls="--", lw=1)
ax[0].set_title("interface_burden histogram (red=zone bounds 12/35/48)")
ax[0].set_xlabel("interface_burden")
zc = np.bincount(tz, minlength=4)
ax[1].bar(range(4), zc, color=["#4daf4a","#ff7f00","#e41a1c","#984ea3"])
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(zone_names, rotation=20)
ax[1].set_title("zone counts")
for i,v in enumerate(zc): ax[1].text(i, v, str(v), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(OUT/"label_hist.png", dpi=110); plt.close()
log("\nsaved", OUT/"label_hist.png")

# ---- montage by zone: 4 rows (zones) x 8 cols, sorted to span each zone ----
section("MONTAGE by zone")
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
for z in range(4):
    sub = train[to_zone(train.interface_burden.values)==z]
    if len(sub)==0: continue
    sub = sub.sort_values("interface_burden")
    picks = sub.iloc[np.linspace(0, len(sub)-1, 8).astype(int)]
    for j,(_,r) in enumerate(picks.iterrows()):
        im = Image.open(DATA_ROOT / r.image_path).convert("RGB")
        axes[z,j].imshow(im); axes[z,j].axis("off")
        axes[z,j].set_title(f"{r.interface_burden:.1f}", fontsize=9)
    axes[z,0].set_ylabel(zone_names[z], fontsize=11)
plt.suptitle("Rows = burden zones 0..3 ; columns span the burden range within each zone", y=0.99)
plt.tight_layout(); plt.savefig(OUT/"montage_by_zone.png", dpi=90); plt.close()
log("saved", OUT/"montage_by_zone.png")

# save label+zone table for reuse
train_out = train.copy()
train_out["zone"] = to_zone(train_out.interface_burden.values)
train_out.to_csv(OUT/"train_with_zone.csv", index=False)

summary = {
    "n_train": int(len(train)), "n_test": int(len(test)),
    "label_min": float(y.min()), "label_max": float(y.max()),
    "label_mean": float(y.mean()), "label_median": float(np.median(y)), "label_std": float(y.std()),
    "zone_counts": np.bincount(tz, minlength=4).tolist(),
    "zone_fracs": (np.bincount(tz, minlength=4)/len(y)).round(4).tolist(),
    "best_constant_value": bc_val, "best_constant_score": bc_score,
    "image_modes": pdf['mode'].value_counts().to_dict(),
    "img_w_median": int(pdf.w.median()), "img_h_median": int(pdf.h.median()),
    "img_size_top": {str(k): int(v) for k,v in pdf.groupby(['w','h']).size().sort_values(ascending=False).head(8).items()},
}
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
(OUT/"eda_log.txt").write_text("\n".join(log_lines))
log("\nsaved", OUT/"summary.json", "and eda_log.txt")
print("\nDONE")
