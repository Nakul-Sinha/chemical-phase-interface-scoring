"""Pre-decode all images ONCE into a single uint8 array on a fast drive (C:),
eliminating per-image small-file disk reads during training (the G: drive's cold
random-read speed was starving the GPU). Training then loads this into RAM.

Stored at STORE_H x STORE_W (>= any training load size); the dataset downscales.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DATA = Path(os.environ.get("DATA_ROOT", r"G:/ml/data/Chemical Phase dataset/public"))
OUTDIR = Path(os.environ.get("STORE_DIR", r"C:/Users/nakul/chem_store"))
OUTDIR.mkdir(parents=True, exist_ok=True)
HS, WS = int(os.environ.get("STORE_H", 448)), int(os.environ.get("STORE_W", 256))

def build(split):
    df = pd.read_csv(DATA / f"{split}.csv")
    ids, paths = df.id.tolist(), df.image_path.tolist()
    arr = np.zeros((len(df), HS, WS, 3), np.uint8)
    def work(k):
        im = Image.open(DATA / paths[k]).convert("RGB").resize((WS, HS), Image.BILINEAR)
        arr[k] = np.asarray(im, np.uint8)
    t = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, range(len(df))))
    np.save(OUTDIR / f"{split}_imgs.npy", arr)
    json.dump(ids, open(OUTDIR / f"{split}_ids.json", "w"))
    print(f"{split}: {len(df)} imgs in {time.time()-t:.0f}s -> {arr.nbytes/1e9:.2f}GB  shape={arr.shape}")

if __name__ == "__main__":
    build("train"); build("test")
    print("STORE built at", OUTDIR, f"({HS}x{WS})")
