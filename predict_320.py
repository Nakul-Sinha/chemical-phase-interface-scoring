import os, sys, glob
import numpy as np, pandas as pd, torch
from scipy.special import softmax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution_core as S
from metric import expected_cost_decision, to_zone

DATA = os.environ.get("DATA_ROOT", r"G:/Datacurve/eris/Chemical Phase dataset/public")
OUT = os.environ.get("OUT_DIR", "working/full320")
T = float(os.environ.get("T", "2.75"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = S.Config()
cfg.data_root = DATA
cfg.backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"
cfg.img_h = 320
cfg.img_w = 192
centers = cfg.centers
test = pd.read_csv(os.path.join(DATA, "test.csv"))
S._STORE = None
cache = {}
models = sorted(glob.glob(OUT + "/model_*.pt"))
assert models, f"no models in {OUT}"
print(f"predict with {len(models)} models: {[os.path.basename(m) for m in models]}", flush=True)
pmf_sum = np.zeros((len(test), cfg.n_bins), np.float64)
nv = 0
for mp in models:
    ck = torch.load(mp, map_location=dev)
    m = S.Net(cfg.backbone, cfg.n_bins, False).to(dev)
    m.load_state_dict(ck["sd"])
    m.eval()
    dl = torch.utils.data.DataLoader(S.ChemDataset(test, cfg, False, None, cache),
                                     batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        ptr = 0
        for b in dl:
            x = b["x"].to(dev)
            for xx in (x, torch.flip(x, [3])):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    cl, rg = m(xx)
                pmf_sum[ptr:ptr + len(x)] += softmax(cl.float().cpu().numpy() / T, 1)
            ptr += len(x)
    nv += 2
pmf = pmf_sum / pmf_sum.sum(1, keepdims=True)
pred = np.clip(expected_cost_decision(pmf, centers), 0, 100)
pd.DataFrame({"id": test.id, "interface_burden": pred}).to_csv("submission.csv", index=False)
print(f"wrote submission.csv mean={pred.mean():.1f} zones={np.bincount(to_zone(pred), minlength=4).tolist()}", flush=True)
