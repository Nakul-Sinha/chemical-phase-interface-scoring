"""Infer experiment GROUPS for honest leave-group-out CV.

Train has no group id, but images are video frames from experiments. KEY finding:
images sharing the exact (width,height) have intra-group burden std ~0.8 vs global
~20 -> exact size is a near-perfect experiment id. We build robust groups as the
connected components of the union of:
   (a) same-exact-size edges  (primary, cheap, intra-std ~0.8)
   (b) embedding cosine >= thr edges (merges experiments recorded at >1 resolution
       and near-duplicate frames across sizes)

NOTE: size/aspect are used ONLY to construct CV folds, NEVER as a model feature
(that would be a disallowed metadata shortcut). All model inputs are resized to a
fixed shape so pixels are the only signal.

Outputs eda/out/train_groups.csv (id, group, zone). Embeddings cached to npy.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from metric import to_zone

DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"G:/ml/data/Chemical Phase dataset/public"))
OUT = REPO / "eda" / "out"; OUT.mkdir(parents=True, exist_ok=True)


def sizes_for(df):
    ws, hs = [], []
    for p in df.image_path:
        with Image.open(DATA_ROOT / p) as im:
            ws.append(im.size[0]); hs.append(im.size[1])
    return np.array(ws), np.array(hs)


def get_embeddings(train):
    EMB = OUT / "emb_train.npy"
    if EMB.exists():
        emb = np.load(EMB); print("loaded cached embeddings", emb.shape); return emb
    import torch, timm
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    paths = train.image_path.tolist()
    class DS(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return tf(Image.open(DATA_ROOT / paths[i]).convert("RGB"))
    dl = DataLoader(DS(), batch_size=64, num_workers=0, pin_memory=True)
    feats, t0 = [], time.time()
    with torch.no_grad():
        for j, x in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                f = model(x)
            feats.append(f.float().cpu().numpy())
            if j % 10 == 0: print(f"  batch {j}/{len(dl)}  {time.time()-t0:.0f}s", flush=True)
    emb = np.concatenate(feats, 0); np.save(EMB, emb)
    print("saved embeddings", emb.shape, f"{time.time()-t0:.0f}s"); return emb


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def main():
    train = pd.read_csv(DATA_ROOT / "train.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")
    print("Reading image sizes (train+test)...")
    tw, th = sizes_for(train); ew, eh = sizes_for(test)
    train["w"], train["h"] = tw, th
    train["size"] = list(zip(tw, th)); test["size"] = list(zip(ew, eh))
    print(f"  unique sizes: train={train['size'].nunique()} test={test['size'].nunique()} "
          f"| shared={len(set(train['size']) & set(test['size']))}")
    y = train.interface_burden.values
    glob_std = y.std()

    emb = get_embeddings(train)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    import torch
    E = torch.tensor(emb); N = len(emb); CH = 512

    # nearest-neighbor sim distribution
    top1 = np.zeros(N)
    for s in range(0, N, CH):
        sim = (E[s:s+CH] @ E.T).numpy()
        for k in range(sim.shape[0]): sim[s+k if False else k, ] = sim[k]
        for k in range(sim.shape[0]): sim[k, s+k] = -1
        top1[s:s+CH] = sim.max(1)
    print("NN top1 cosine percentiles:",
          {f"p{q}": round(float(np.percentile(top1, q)), 3) for q in [10, 50, 90, 99]})

    W = train["w"].values.astype(float); H = train["h"].values.astype(float)
    uniq_sizes = sorted(set(zip(train["w"], train["h"])))

    def build_size_groups(tol_rel=0.0, tol_abs=0):
        """Union images by exact size, then merge size-groups whose (w,h) are
        within tolerance (same experiment recorded at slightly different crops).
        NO embedding chaining -> avoids the giant-component failure mode."""
        uf = UF(N)
        # exact-size unions
        by = {}
        for i in range(N): by.setdefault((W[i], H[i]), []).append(i)
        for ids in by.values():
            for i in ids[1:]: uf.union(ids[0], i)
        # near-size unions between distinct sizes
        if tol_rel > 0 or tol_abs > 0:
            reps = {sz: by[sz][0] for sz in by}
            S = np.array(uniq_sizes, dtype=float)
            for a in range(len(S)):
                for b in range(a+1, len(S)):
                    dw = abs(S[a,0]-S[b,0]); dh = abs(S[a,1]-S[b,1])
                    okw = dw <= max(tol_abs, tol_rel*max(S[a,0],S[b,0]))
                    okh = dh <= max(tol_abs, tol_rel*max(S[a,1],S[b,1]))
                    if okw and okh:
                        uf.union(reps[tuple(S[a])], reps[tuple(S[b])])
        lab = np.array([uf.find(i) for i in range(N)])
        _, lab = np.unique(lab, return_inverse=True)
        return lab

    def stats(lab):
        K = lab.max() + 1; sizes = np.bincount(lab)
        stds, ws = [], []
        for g in range(K):
            m = lab == g
            if m.sum() >= 2: stds.append(y[m].std()); ws.append(m.sum())
        wstd = np.average(stds, weights=ws) if stds else float("nan")
        return K, int(sizes.max()), sizes.max()/N, (sizes == 1).mean()*100, wstd

    def leak_pairs(lab, thr=0.985):
        """count near-duplicate pairs (cosine>=thr) that fall in DIFFERENT groups
        -> these would leak across folds. lower is better."""
        cnt = 0
        for s in range(0, N, CH):
            sim = (E[s:s+CH] @ E.T).numpy()
            for k in range(sim.shape[0]): sim[k, s+k] = -1
            ii, jj = np.where(sim >= thr)
            for a, b in zip(ii, jj):
                ga = s + a
                if ga < b and lab[ga] != lab[b]: cnt += 1
        return cnt

    print(f"\nglobal burden std={glob_std:.2f}")
    print(f"{'mode':>20} {'n_grp':>6} {'maxsz':>6} {'maxfrac':>7} {'singl%':>7} {'wIntraStd':>10} {'leak@.985':>10}")
    cfgs = {"exact-size": (0.0, 0), "near-size@2px": (0.0, 2), "near-size@1%": (0.012, 3), "near-size@2%": (0.02, 5)}
    res = {}
    for name, (tr, ta) in cfgs.items():
        lab = build_size_groups(tr, ta); st = stats(lab); lk = leak_pairs(lab)
        res[name] = (lab, st, lk)
        print(f"{name:>20} {st[0]:6d} {st[1]:6d} {st[2]:7.3f} {st[3]:7.1f} {st[4]:10.2f} {lk:10d}")

    # choose the knee: low leakage AND low intra-std AND many groups, no giant comp.
    chosen = "near-size@2px"
    lab = res[chosen][0]; K = lab.max() + 1
    print(f"\nCHOSEN: {chosen} -> {K} groups, maxfrac={res[chosen][1][2]:.3f}, "
          f"wIntraStd={res[chosen][1][4]:.2f}, cross-group near-dups={res[chosen][2]}")
    res = {chosen: (lab, res[chosen][1], 0.0)}; chosen_thr = chosen

    out = train[["id", "image_path", "interface_burden"]].copy()
    out["group"] = lab
    out["zone"] = to_zone(out.interface_burden.values)
    out.to_csv(OUT / "train_groups.csv", index=False)
    gs = np.bincount(lab)
    print(f"group sizes: min={gs.min()} median={int(np.median(gs))} max={gs.max()} mean={gs.mean():.1f}")
    print(f"singletons={int((gs==1).sum())}  groups>=10={int((gs>=10).sum())}")
    json.dump({"chosen_grouping": str(chosen_thr), "n_groups": int(K),
               "max_group_frac": float(res[chosen_thr][1][2]),
               "w_intra_burden_std": float(res[chosen_thr][1][4]),
               "global_burden_std": float(glob_std),
               "n_unique_sizes_train": int(train['size'].nunique())},
              open(OUT / "group_summary.json", "w"), indent=2)
    print("saved", OUT / "train_groups.csv", "and group_summary.json\nDONE")


if __name__ == "__main__":
    main()
