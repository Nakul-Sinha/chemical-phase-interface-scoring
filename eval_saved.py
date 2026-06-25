"""Quick corrected OOF re-eval of saved fold models (validates the alignment fix)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import softmax
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution_core as S
from metric import evaluate_components, evaluate, to_zone, expected_cost_decision, fit_temperature

cfg = S.Config()
cfg.img_h, cfg.img_w = int(os.environ.get("IMG_H", 320)), int(os.environ.get("IMG_W", 192))
cfg.n_folds = int(os.environ.get("N_FOLDS", 5))
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(Path(cfg.data_root) / "train.csv")
groups = S.compute_groups(df, cfg.data_root)
fold_arr = S.make_folds(df, groups, cfg.n_folds, cfg.seed)
centers = cfg.centers
y = df.interface_burden.values

oof_logits = np.full((len(df), cfg.n_bins), np.nan, np.float32)
oof_reg = np.full(len(df), np.nan, np.float32)
for mp in sorted(Path(cfg.out_dir).glob("model_f*.pt")):
    f = int(mp.stem.split("f")[-1])
    va = df[fold_arr == f]
    ds = S.ChemDataset(va, cfg, False, None, None)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model = S.Net(cfg.backbone, cfg.n_bins, False).to(device)
    model.load_state_dict(torch.load(mp, map_location=device)["sd"]); model.eval()
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                cl, rg = model(x)
            idx = b["idx"].numpy()
            oof_logits[idx] = cl.float().cpu().numpy()
            oof_reg[idx] = (torch.sigmoid(rg).float().cpu().numpy() * 100)
    print(f"fold {f}: evaluated {len(va)} val samples")

done = ~np.isnan(oof_reg)
T, _ = fit_temperature(oof_logits[done], y[done], centers)
pmf = softmax(oof_logits[done] / T, 1)
pred = expected_cost_decision(pmf, centers)
comp = evaluate_components(y[done], pred)
pmf_exp = pmf @ centers
print(f"\n=== CORRECTED OOF (n={done.sum()}) ===")
print(f"  T*={T}  expected-cost score={comp['total']:.4f}  zone_acc={comp['zone_accuracy']:.3f}")
print(f"  reg-only={evaluate(y[done], np.clip(oof_reg[done],0,100)):.4f}  pmf-exp={evaluate(y[done], np.clip(pmf_exp,0,100)):.4f}")
print(f"  spearman(reg,y)={spearmanr(oof_reg[done],y[done]).correlation:.3f}  spearman(pmf_exp,y)={spearmanr(pmf_exp,y[done]).correlation:.3f}")
tz, pz = to_zone(y[done]), to_zone(pred)
cm = np.zeros((4,4), int)
for a,b in zip(tz,pz): cm[a,b]+=1
print("  zone confusion (rows=true,cols=pred):")
for r in range(4): print("   ", cm[r].tolist())
print("  pred zone dist:", np.bincount(pz, minlength=4).tolist(), "| true:", np.bincount(tz, minlength=4).tolist())
