"""Chemical Phase Interface Assessment — core training/inference.

Design (grounded in EDA + 3-agent research, see notes.md / research_findings.md):
  * Honest CV: leave-experiment-out via size-based groups (exact size ~ experiment
    id; near-size@2px merge). Size is used ONLY for folds, never as a model input.
  * Model emits a DISTRIBUTION: ConvNeXt backbone + GeM pool -> SORD soft-ordinal
    head (K bins -> PMF) + auxiliary BCE-on-[0,1] regression head.
  * Decision = Bayes-optimal expected-cost over the EXACT metric (metric.py), on a
    temperature-calibrated OOF-averaged PMF. This exploits the known piecewise metric.
  * OOD augmentation: horizontal flip ONLY (vertical order is physical), color/hue
    jitter + light grayscale to kill palette shortcuts, blur/noise for glare, C-Mixup.

Run modes (env MODE): smoke | cv | predict. Paths via env DATA_ROOT / OUT_DIR.
"""
import os, sys, math, json, time, random
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_STORE = None   # optional {id: uint8 (Hs,Ws,3)} RAM image store (set by run_cv/predict_test)

def load_store(split):
    """Load the pre-decoded image store from STORE_DIR into a {id: array} dict (views)."""
    sd = os.environ.get("STORE_DIR")
    if not sd or not (Path(sd) / f"{split}_imgs.npy").exists():
        return None
    imgs = np.load(Path(sd) / f"{split}_imgs.npy")
    ids = json.load(open(Path(sd) / f"{split}_ids.json"))
    return {i: imgs[k] for k, i in enumerate(ids)}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import evaluate, evaluate_components, to_zone, expected_cost_decision, fit_temperature

# ----------------------------- config -----------------------------
def _env(k, d, cast=str):
    v = os.environ.get(k)
    return cast(v) if v is not None else d

@dataclass
class Config:
    data_root: str = _env("DATA_ROOT", r"G:/Datacurve/eris/Chemical Phase dataset/public")
    out_dir: str = _env("OUT_DIR", "working")
    backbone: str = _env("BACKBONE", "convnextv2_nano.fcmae_ft_in22k_in1k")
    pretrained: bool = _env("PRETRAINED", 1, int) == 1
    img_h: int = _env("IMG_H", 384, int)
    img_w: int = _env("IMG_W", 224, int)
    n_folds: int = _env("N_FOLDS", 5, int)
    folds_to_run: str = _env("FOLDS", "", str)   # e.g. "0,1"; empty=all
    epochs: int = _env("EPOCHS", 18, int)
    batch_size: int = _env("BATCH", 16, int)
    lr: float = _env("LR", 2.5e-4, float)
    head_lr_mult: float = _env("HEAD_LR_MULT", 5.0, float)
    weight_decay: float = _env("WD", 0.05, float)
    warmup_frac: float = _env("WARMUP", 0.1, float)
    reg_weight: float = _env("REG_W", 0.3, float)
    n_bins: int = _env("NBINS", 69, int)
    bin_lo: float = _env("BIN_LO", 0.0, float)
    bin_hi: float = _env("BIN_HI", 68.0, float)
    sord_sigma: float = _env("SORD_SIGMA", 2.0, float)   # in burden units
    ema_decay: float = _env("EMA", 0.999, float)
    cmix_p: float = _env("CMIX_P", 0.5, float)
    cmix_alpha: float = _env("CMIX_ALPHA", 0.3, float)
    color_jitter: float = _env("CJ", 0.3, float)
    gray_p: float = _env("GRAY_P", 0.1, float)
    mixstyle: bool = _env("MIXSTYLE", 0, int) == 1
    mixstyle_p: float = _env("MIXSTYLE_P", 0.5, float)
    drop_path: float = _env("DROP_PATH", 0.0, float)
    num_workers: int = _env("NUM_WORKERS", 0, int)
    seed: int = _env("SEED", 42, int)
    smoke: bool = _env("SMOKE", 0, int) == 1
    cache: bool = _env("CACHE", 0, int) == 1

    @property
    def centers(self):
        return np.linspace(self.bin_lo, self.bin_hi, self.n_bins)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------------------- groups / folds -----------------------------
def image_sizes(df, data_root):
    ws, hs = [], []
    for p in df.image_path:
        with Image.open(Path(data_root) / p) as im:
            ws.append(im.size[0]); hs.append(im.size[1])
    return np.array(ws), np.array(hs)

def compute_groups(df, data_root, tol_abs=2):
    """Leave-experiment-out groups from image size only (reproducible, no model).
    Exact-size union + merge sizes within tol_abs px in BOTH dims. Matches the
    near-size@2px grouping validated in eda/group_infer.py (272 groups)."""
    W, H = image_sizes(df, data_root)
    n = len(df)
    parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    by = {}
    for i in range(n): by.setdefault((W[i], H[i]), []).append(i)
    for ids in by.values():
        for i in ids[1:]: union(ids[0], i)
    sizes = list(by.keys()); reps = {s: by[s][0] for s in sizes}
    S = np.array(sizes, dtype=float)
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            if abs(S[a, 0]-S[b, 0]) <= tol_abs and abs(S[a, 1]-S[b, 1]) <= tol_abs:
                union(reps[sizes[a]], reps[sizes[b]])
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab

def make_folds(df, groups, n_folds, seed):
    """StratifiedGroupKFold on (zone x group): balance zones, never split a group."""
    from sklearn.model_selection import StratifiedGroupKFold
    zone = to_zone(df.interface_burden.values)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for f, (_, va) in enumerate(skf.split(df, zone, groups)):
        fold[va] = f
    assert (fold >= 0).all()
    return fold

# ----------------------------- targets -----------------------------
def soft_ordinal_targets(y, centers, sigma):
    """SORD: soft label = softmax(-(center-y)^2 / (2 sigma^2)) over bins."""
    d2 = (centers[None, :] - np.asarray(y)[:, None]) ** 2
    logits = -d2 / (2 * sigma ** 2)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    return p.astype(np.float32)

# ----------------------------- model -----------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.ones(1) * p); self.eps = eps
    def forward(self, x):                      # (B,C,H,W)
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)

class MixStyle(nn.Module):
    """Mix instance-level feature statistics across the batch (Zhou et al. ICLR'21).
    Applied at EARLY stages only (late stages carry label info). Train-time only."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__(); self.p = p; self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)
    def forward(self, x):
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x
        mu = x.mean([2, 3], keepdim=True); var = x.var([2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        xn = (x - mu) / sig
        lam = self.beta.sample((x.size(0), 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(x.size(0), device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return xn * sig_mix + mu_mix

class Net(nn.Module):
    def __init__(self, backbone, n_bins, pretrained=True, mixstyle=False, mixstyle_p=0.5, drop_path=0.0):
        super().__init__()
        import timm
        self.bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="",
                                    drop_path_rate=drop_path)
        C = self.bb.num_features
        self.pool = GeM()
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Linear(C, n_bins)
        self.reg = nn.Linear(C, 1)
        self.ms = MixStyle(p=mixstyle_p) if mixstyle else None
        if self.ms is not None and hasattr(self.bb, "stages"):
            for i in [0, 1, 2]:   # early stages only
                self.bb.stages[i].register_forward_hook(lambda mod, inp, out: self.ms(out))
    def forward(self, x):
        f = self.bb.forward_features(x)        # (B,C,h,w)
        if f.ndim == 4 and f.shape[1] != self.bb.num_features and f.shape[-1] == self.bb.num_features:
            f = f.permute(0, 3, 1, 2).contiguous()   # NHWC -> NCHW safety
        z = self.drop(self.pool(f))
        return self.cls(z), self.reg(z).squeeze(-1)

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point: s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else: s.copy_(v)
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

# ----------------------------- data -----------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class ChemDataset(Dataset):
    def __init__(self, df, cfg, train, soft=None, cache_store=None):
        self.orig_idx = df.index.to_numpy()           # GLOBAL positions for OOF alignment
        self.df = df.reset_index(drop=True); self.cfg = cfg; self.train = train
        self.soft = soft
        self.y = self.df.interface_burden.values.astype(np.float32) if "interface_burden" in df else None
        self.load_h = int(cfg.img_h * 1.12)           # same scale for train & val (center-crop)
        self.load_w = int(cfg.img_w * 1.12)
        self.cache = cache_store    # dict idx->uint8 array, optional
        self.store = _STORE         # RAM store {id: (Hs,Ws,3)} if preprocessed
        self.ids = self.df.id.values
    def __len__(self): return len(self.df)
    def _load(self, i):
        if self.store is not None:                    # fast path: from RAM, no disk
            a = self.store[self.ids[i]]
            if a.shape[0] != self.load_h or a.shape[1] != self.load_w:
                a = cv2.resize(a, (self.load_w, self.load_h), interpolation=cv2.INTER_AREA)
            return a
        key = int(self.orig_idx[i])     # GLOBAL key: cache shared safely across folds
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        p = Path(self.cfg.data_root) / self.df.image_path.iloc[i]
        im = Image.open(p).convert("RGB").resize((self.load_w, self.load_h), Image.BILINEAR)
        a = np.asarray(im, dtype=np.uint8)
        if self.cache is not None: self.cache[key] = a
        return a
    def __getitem__(self, i):
        import torchvision.transforms.v2.functional as TF
        a = self._load(i)
        x = torch.from_numpy(a).permute(2, 0, 1)          # (3,H,W) uint8
        H, W = self.cfg.img_h, self.cfg.img_w
        if self.train:
            # mild random crop (translation), horizontal flip ONLY
            top = random.randint(0, x.shape[1] - H); left = random.randint(0, x.shape[2] - W)
            x = x[:, top:top + H, left:left + W]
            if random.random() < 0.5: x = torch.flip(x, [2])
            x = x.float() / 255.0
            cj = self.cfg.color_jitter
            if cj > 0:
                x = TF.adjust_brightness(x, 1 + random.uniform(-cj, cj))
                x = TF.adjust_contrast(x, 1 + random.uniform(-cj, cj))
                x = TF.adjust_saturation(x, 1 + random.uniform(-cj, cj))
                x = TF.adjust_hue(x, random.uniform(-cj * 0.15, cj * 0.15))
            if random.random() < self.cfg.gray_p:
                g = x.mean(0, keepdim=True); x = g.repeat(3, 1, 1)
            if random.random() < 0.2:
                x = TF.gaussian_blur(x, kernel_size=3)
            if random.random() < 0.2:
                x = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
        else:
            th = (x.shape[1] - H) // 2; lw = (x.shape[2] - W) // 2
            x = x[:, th:th + H, lw:lw + W].float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out = {"x": x, "idx": int(self.orig_idx[i])}
        if self.y is not None: out["y"] = self.y[i]
        if self.soft is not None: out["soft"] = torch.from_numpy(self.soft[i])
        return out

def cmixup(batch_x, soft, yreg, alpha):
    """C-Mixup: mix each sample with its nearest-in-burden neighbor in the batch."""
    order = torch.argsort(yreg)
    partner = torch.empty_like(order); partner[order] = torch.roll(order, 1)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    x = lam * batch_x + (1 - lam) * batch_x[partner]
    s = lam * soft + (1 - lam) * soft[partner]
    yr = lam * yreg + (1 - lam) * yreg[partner]
    return x, s, yr

# ----------------------------- train one fold -----------------------------
def soft_ce(logits, target):                    # both (B,K)
    return -(target * F.log_softmax(logits, 1)).sum(1).mean()

def train_fold(cfg, df, fold_arr, fold, centers, device, cache=None, log=print):
    tr = df[fold_arr != fold]; va = df[fold_arr == fold]
    soft_tr = soft_ordinal_targets(tr.interface_burden.values, centers, cfg.sord_sigma)
    ds_tr = ChemDataset(tr, cfg, True, soft_tr, cache)
    ds_va = ChemDataset(va, cfg, False, None, cache)
    pw = cfg.num_workers > 0
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                       num_workers=cfg.num_workers, pin_memory=True, persistent_workers=pw,
                       prefetch_factor=4 if pw else None)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True, persistent_workers=pw)
    model = Net(cfg.backbone, cfg.n_bins, cfg.pretrained, cfg.mixstyle, cfg.mixstyle_p, cfg.drop_path).to(device)
    head_ids = {id(p) for n, p in model.named_parameters() if n.startswith(("cls", "reg", "pool"))}
    params = [
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg.lr},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": cfg.lr * cfg.head_lr_mult},
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    steps = len(dl_tr) * cfg.epochs; warm = int(steps * cfg.warmup_frac)
    def lr_at(s):
        if s < warm: return s / max(1, warm)
        t = (s - warm) / max(1, steps - warm); return 0.5 * (1 + math.cos(math.pi * t))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    ema = EMA(model, cfg.ema_decay)
    gstep = 0
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for b in dl_tr:
            x = b["x"].to(device, non_blocking=True); soft = b["soft"].to(device); yreg = b["y"].to(device)
            if cfg.cmix_p > 0 and random.random() < cfg.cmix_p:
                x, soft, yreg = cmixup(x, soft, yreg, cfg.cmix_alpha)
            for g in opt.param_groups: g["lr"] = (cfg.lr if g is opt.param_groups[0] else cfg.lr * cfg.head_lr_mult) * lr_at(gstep)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                cl, rg = model(x)
                loss = soft_ce(cl, soft) + cfg.reg_weight * F.binary_cross_entropy_with_logits(rg, yreg / 100.0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ema.update(model); gstep += 1; running += loss.item()
        log(f"  fold{fold} ep{ep+1}/{cfg.epochs} loss={running/len(dl_tr):.4f} {time.time()-t0:.0f}s")
    # OOF predict with EMA weights
    eval_model = Net(cfg.backbone, cfg.n_bins, False).to(device); eval_model.load_state_dict(ema.shadow); eval_model.eval()
    logits_all, reg_all, idx_all = [], [], []
    with torch.no_grad():
        for b in dl_va:
            x = b["x"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                cl, rg = eval_model(x)
            logits_all.append(cl.float().cpu().numpy()); reg_all.append(torch.sigmoid(rg).float().cpu().numpy() * 100)
            idx_all.append(b["idx"].numpy())
    return (np.concatenate(idx_all), np.concatenate(logits_all), np.concatenate(reg_all),
            {k: v.cpu() for k, v in ema.shadow.items()})

# ----------------------------- CV orchestration -----------------------------
def run_cv(cfg, log=print):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(cfg.data_root) / "train.csv")
    if cfg.smoke:
        df = df.groupby(to_zone(df.interface_burden.values)).head(150).reset_index(drop=True)
        log(f"SMOKE subset: {len(df)} rows")
    global _STORE
    _STORE = load_store("train")
    if _STORE is not None: log(f"RAM store loaded: {len(_STORE)} train images")
    groups = compute_groups(df, cfg.data_root)
    fold_arr = make_folds(df, groups, cfg.n_folds, cfg.seed)
    log(f"groups={groups.max()+1} folds={cfg.n_folds} | fold sizes={np.bincount(fold_arr).tolist()}")
    centers = cfg.centers
    cache = {} if cfg.cache else None
    oof_logits = np.zeros((len(df), cfg.n_bins), np.float32); oof_reg = np.zeros(len(df), np.float32)
    run_folds = [int(x) for x in cfg.folds_to_run.split(",") if x != ""] or list(range(cfg.n_folds))
    for f in run_folds:
        idx, logits, reg, sd = train_fold(cfg, df, fold_arr, f, centers, device, cache, log)
        oof_logits[idx] = logits; oof_reg[idx] = reg
        torch.save({"sd": sd, "cfg": asdict(cfg)}, out / f"model_f{f}.pt")
    y = df.interface_burden.values
    done = np.isin(fold_arr, run_folds)
    # decision: temperature-calibrate on OOF then expected-cost
    T, _ = fit_temperature(oof_logits[done], y[done], centers)
    from scipy.special import softmax
    pmf = softmax(oof_logits[done] / T, 1)
    pred = expected_cost_decision(pmf, centers)
    comp = evaluate_components(y[done], pred)
    # baselines for context
    reg_only = evaluate(y[done], np.clip(oof_reg[done], 0, 100))
    pmf_exp = pmf @ centers  # PMF expectation as a point estimate
    exp_only = evaluate(y[done], np.clip(pmf_exp, 0, 100))
    from scipy.stats import spearmanr
    sp_reg = spearmanr(oof_reg[done], y[done]).correlation
    sp_exp = spearmanr(pmf_exp, y[done]).correlation
    log(f"\n=== OOF (folds {run_folds}, n={done.sum()}) ===")
    log(f"  T*={T}  expected-cost score={comp['total']:.4f}  (zone_acc={comp['zone_accuracy']:.3f}, "
        f"zone={comp['w_zone']:.3f} abs={comp['w_absolute']:.3f} high={comp['w_high']:.3f} ext={comp['w_extreme']:.3f})")
    log(f"  reg-head-only score={reg_only:.4f} | pmf-expectation score={exp_only:.4f}  (context)")
    log(f"  spearman(reg,y)={sp_reg:.3f}  spearman(pmf_exp,y)={sp_exp:.3f}  (did it learn?)")
    tz, pz = to_zone(y[done]), to_zone(pred)
    cm = np.zeros((4, 4), int)
    for a, b in zip(tz, pz): cm[a, b] += 1
    log("  zone confusion (rows=true 0..3, cols=pred 0..3):")
    for r in range(4): log("    ", cm[r].tolist())
    np.savez(out / "oof.npz", logits=oof_logits, reg=oof_reg, fold=fold_arr, y=y,
             ids=df.id.values, done=done, centers=centers, T=T)
    json.dump({"score": comp["total"], "T": T, "zone_acc": comp["zone_accuracy"],
               "components": {k: float(v) for k, v in comp.items()}, "folds": run_folds},
              open(out / "cv_result.json", "w"), indent=2)
    log(f"saved {out/'oof.npz'} and cv_result.json")
    return comp["total"]

def apply_decision(pmf, reg, centers, dc):
    """Map per-sample PMF (+reg) to a final burden value using a decision config dc.
    Strategies:
      expected_cost : Bayes-optimal grid search over the exact metric (metric.py)
      pmf_exp       : PMF expectation, clipped
      reg           : regression head, clipped
      blend_thresh  : s = w*pmf_exp + (1-w)*reg; re-zone by tuned cuts; clip s into
                      the predicted zone bounds (decouples the zone decision from
                      calibration -> tunes the hard Z2/Z3 boundary directly)."""
    from metric import SEVERITY_BINS
    s_exp = pmf @ centers
    strat = dc.get("strategy", "blend_thresh")
    if strat == "expected_cost":
        return np.clip(expected_cost_decision(pmf, centers), 0, 100)
    if strat == "pmf_exp":
        return np.clip(s_exp, 0, 100)
    if strat == "reg":
        return np.clip(reg, 0, 100)
    # blend_thresh
    w = dc.get("w", 1.0); cuts = np.asarray(dc.get("cuts", [12.0, 35.0, 48.0]))
    m = dc.get("margin", 0.5)
    s = w * s_exp + (1 - w) * reg
    z = np.digitize(s, cuts)                       # 0..3
    lo = SEVERITY_BINS[z]; hi = SEVERITY_BINS[z + 1]
    return np.clip(np.clip(s, lo + m, hi - m), 0, 100)


def predict_test(cfg, log=print):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    test = pd.read_csv(Path(cfg.data_root) / "test.csv")
    global _STORE
    _STORE = load_store("test")
    centers = cfg.centers
    models = sorted(out.glob("model_f*.pt"))
    assert models, "no fold models found"
    oof = np.load(out / "oof.npz")
    dc = json.load(open(out / "decision.json")) if (out / "decision.json").exists() else {"strategy": "expected_cost"}
    T = float(dc.get("T", oof["T"]))
    ds = ChemDataset(test, cfg, False, None, None)
    dl = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    from scipy.special import softmax
    pmf_sum = np.zeros((len(test), cfg.n_bins), np.float64)
    reg_sum = np.zeros(len(test), np.float64); nviews = 0
    for mp in models:
        ckpt = torch.load(mp, map_location=device)
        model = Net(cfg.backbone, cfg.n_bins, False).to(device); model.load_state_dict(ckpt["sd"]); model.eval()
        with torch.no_grad():
            ptr = 0
            for b in dl:
                x = b["x"].to(device)
                for xx in (x, torch.flip(x, [3])):    # hflip TTA
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                        cl, rg = model(xx)
                    pmf_sum[ptr:ptr + len(x)] += softmax(cl.float().cpu().numpy() / T, 1)
                    reg_sum[ptr:ptr + len(x)] += torch.sigmoid(rg).float().cpu().numpy() * 100
                ptr += len(x)
        nviews += 2
    pmf = pmf_sum / pmf_sum.sum(1, keepdims=True)
    reg = reg_sum / nviews
    pred = np.clip(apply_decision(pmf, reg, centers, dc), 0, 100)
    log(f"decision: {dc}")
    sub = pd.DataFrame({"id": test.id, "interface_burden": pred})
    sub.to_csv(out / "submission.csv", index=False)
    log(f"wrote {out/'submission.csv'} rows={len(sub)} pred[min/mean/max]={pred.min():.1f}/{pred.mean():.1f}/{pred.max():.1f}")
    log(f"pred zone dist={np.bincount(to_zone(pred), minlength=4).tolist()}")
    return sub

if __name__ == "__main__":
    cfg = Config()
    mode = os.environ.get("MODE", "smoke")
    print("MODE:", mode, "| device:", "cuda" if torch.cuda.is_available() else "cpu", "| backbone:", cfg.backbone)
    if mode == "smoke":
        cfg.smoke = True; cfg.epochs = min(cfg.epochs, 2); cfg.n_folds = 3; cfg.folds_to_run = "0"
        cfg.img_h, cfg.img_w = 256, 160; cfg.backbone = _env("BACKBONE", "convnextv2_femto.fcmae_ft_in1k")
        run_cv(cfg)
    elif mode == "cv":
        run_cv(cfg)
    elif mode == "predict":
        predict_test(cfg)
    elif mode == "cv_predict":
        run_cv(cfg); predict_test(cfg)
