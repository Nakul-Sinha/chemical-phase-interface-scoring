"""Simple compliant per-frame prediction: average (SORD-expectation + regression)
over all given models + hflip TTA, clip to [0,100]. No size grouping, no thresholds,
no hard-coding — pure learned-model output. Usage: python simple_predict.py <model_dir> <out.csv>
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import softmax
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution_core as S
from metric import to_zone

DATA = os.environ.get("DATA_ROOT", "/mnt/chem/data")
model_dir = sys.argv[1]; out = sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
test = pd.read_csv(Path(DATA) / "test.csv")
mps = sorted(glob.glob(model_dir + "/model_*.pt"))
assert mps, f"no models in {model_dir}"
print(f"{len(mps)} models over {len(test)} test images")
S._STORE = None
val = np.zeros(len(test), np.float64); nv = 0
for mp in mps:
    ck = torch.load(mp, map_location=dev); c = ck["cfg"]
    cfg = S.Config(); cfg.backbone = c["backbone"]; cfg.img_h = c["img_h"]; cfg.img_w = c["img_w"]
    cfg.n_bins = c["n_bins"]; cfg.data_root = DATA
    centers = np.linspace(c["bin_lo"], c["bin_hi"], c["n_bins"])
    m = S.Net(cfg.backbone, cfg.n_bins, False).to(dev); m.load_state_dict(ck["sd"]); m.eval()
    dl = torch.utils.data.DataLoader(S.ChemDataset(test, cfg, False, None, None),
                                     batch_size=48, shuffle=False, num_workers=8, pin_memory=True)
    with torch.no_grad():
        ptr = 0
        for b in dl:
            x = b["x"].to(dev)
            for xx in (x, torch.flip(x, [3])):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    cl, rg = m(xx)
                exp = softmax(cl.float().cpu().numpy(), 1) @ centers
                reg = torch.sigmoid(rg).float().cpu().numpy() * 100
                val[ptr:ptr + len(x)] += 0.5 * exp + 0.5 * reg
            ptr += len(x)
    nv += 2
pred = np.clip(val / nv, 0, 100)
pd.DataFrame({"id": test.id, "interface_burden": pred}).to_csv(out, index=False)
print(f"wrote {out} | pred[min/mean/max]={pred.min():.1f}/{pred.mean():.1f}/{pred.max():.1f} zones={np.bincount(to_zone(pred), minlength=4).tolist()}")
