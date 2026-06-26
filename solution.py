import os, sys, glob
import numpy as np, pandas as pd, torch
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution_core as S
from metric import to_zone

DATA = os.environ.get("DATA_ROOT", "dataset/public")
OUT = os.environ.get("OUT_DIR", "working")
SEEDS = int(os.environ.get("SEEDS", "3"))
SINGLE = os.environ.get("SINGLE_MODEL", "0") == "1"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = [
    dict(tag="nano512", backbone="convnextv2_nano.fcmae_ft_in22k_in1k", img_h=512, img_w=320, batch=32),
    dict(tag="nano768", backbone="convnextv2_nano.fcmae_ft_in22k_in1k", img_h=768, img_w=448, batch=16),
    dict(tag="tiny512", backbone="convnextv2_tiny.fcmae_ft_in22k_in1k", img_h=512, img_w=320, batch=24),
]
if SINGLE:
    CONFIGS = CONFIGS[:1]

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
tb = train.interface_burden.values


def predict_dir(models_dir):
    S._STORE = None
    val = np.zeros(len(test), np.float64)
    nv = 0
    for mp in sorted(glob.glob(models_dir + "/model_*.pt")):
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
        dl = torch.utils.data.DataLoader(S.ChemDataset(test, cc, False, None, None),
                                         batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            ptr = 0
            for b in dl:
                x = b["x"].to(dev)
                for xx in (x, torch.flip(x, [3])):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                        cl, rg = m(xx)
                    exp = softmax(cl.float().cpu().numpy(), 1) @ cen
                    reg = torch.sigmoid(rg).float().cpu().numpy() * 100
                    val[ptr:ptr + len(x)] += 0.5 * exp + 0.5 * reg
                ptr += len(x)
        nv += 2
    assert nv > 0, f"no models trained in {models_dir}"
    return np.clip(val / nv, 0, 100)


def rank_norm(x):
    return x.argsort().argsort() / (len(x) - 1)


raw_preds = []
for c in CONFIGS:
    mdir = f"{OUT}/full_{c['tag']}"
    os.makedirs(mdir, exist_ok=True)
    cfg = S.Config()
    cfg.data_root = DATA
    cfg.out_dir = mdir
    cfg.backbone = c["backbone"]
    cfg.img_h = c["img_h"]
    cfg.img_w = c["img_w"]
    cfg.batch_size = c["batch"]
    cfg.epochs = 16
    cfg.drop_path = 0.0
    cfg.fold_seed = 42
    cfg.num_workers = 4
    os.environ["FULL_SEEDS"] = ",".join(str(42 + i) for i in range(SEEDS))
    print(f"train {c['tag']} ({SEEDS} seeds)", flush=True)
    S.train_full(cfg)
    raw_preds.append(predict_dir(mdir))
    print(f"{c['tag']} predicted", flush=True)

ens = np.mean([rank_norm(p) for p in raw_preds], axis=0)
ranks = ens.argsort().argsort()
cal = np.quantile(tb, (ranks + 0.5) / len(ens))

pd.DataFrame({"id": test.id, "interface_burden": np.clip(cal, 0, 100)}).to_csv("submission.csv", index=False)
print(f"wrote submission.csv models={len(CONFIGS)} zones={np.bincount(to_zone(cal), minlength=4).tolist()} mean={cal.mean():.1f}", flush=True)
