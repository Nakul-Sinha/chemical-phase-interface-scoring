# Approach — Chemical Phase Interface Assessment

A research-backed solution for predicting an ordinal `interface_burden` index (0–100)
from a single photo of a chemical vessel. Every decision below is grounded in **(a)**
direct measurement of the public data (see `eda/`) and **(b)** a 3-agent deep-research
review (papers + repos, cited in `research_findings.md`).

---

## 1. What the task really is (and how the metric drives everything)

We output one number per image, but the grader is **~80% an ordinal 4-zone classifier**:

```
zones from bins [0,12,35,48,100]  ->  Z0[0,12) Z1[12,35) Z2[35,48) Z3[48,100]
score = 0.80·zone_penalty + 0.05·MAE + 0.05·MAE(y≥48) + 0.10·extreme_miss·100   (LOWER is better)
zone_penalty = 0 (same zone) / 60 (off-by-one) / 100 (off-by-≥2)
extreme_miss = (y≤12 & pred>25) or (y≥48 & pred<40)
```

**Consequences that shape the whole design:**
- The dominant objective is **getting the zone right**. Measured on train: predicting the
  best constant scores **45.2**; an oracle that knows the true zone and outputs its center
  scores **0.82**. So the entire winnable range is zone accuracy. Improving zone accuracy by
  10 pp improves the score by ~4.8.
- The metric is **known, deterministic, piecewise-constant, and asymmetric** → the single
  highest-leverage move is a **Bayes-optimal decision layer** (§5), not the architecture.
- The narrow middle band **Z2 [35,48)** and the **48 boundary** are where errors concentrate,
  so calibration and threshold placement there matter most.

## 2. Data facts (measured, see `eda/`)
| Property | Value | Design consequence |
|---|---|---|
| Train / test | 4,770 / 1,261 | Medium data → pretrained backbone, augmentation, CV, ensembling. |
| Label range | **0–67** (not 0–100), multimodal (spikes ~4/47/51/62) | Bins only need to span [0,68]; Z3 in practice is [48,67]. |
| Zone balance | Z0 20% / Z1 19% / Z2 22% / Z3 38% | Mild imbalance; Z3 majority. |
| Images | **all RGB**, **tall portrait** vessels, aspect ≈0.47, median 336×739 | Color matters; **vertical structure carries the phase-layer signal** → tall fixed input, **no vertical flip**. |
| Visual cue | turbidity/milkiness, suspended solids, layering, dark/heterogeneous regions; heavy **glare** | Texture+layering task; augment for glare/lighting. |
| **Exact image size ≈ experiment id** | within-size burden std **0.68** vs global 20 | Each experiment shot at fixed resolution → use **size for CV grouping only**, never as a feature. |

## 3. Honest cross-validation (the foundation)
Test holds out **entire experiment groups**, and the images are **video frames** (near-duplicates).
Random K-fold would leak near-duplicate frames and lie. We reconstruct experiment groups from
**image size** (exact-size union + merge sizes within 2 px; no embedding chaining, which created a
giant component): **272 groups**, intra-burden-std 3.1, max group 6.2%, only 12 cross-group
near-duplicate pairs. CV = **StratifiedGroupKFold on (zone × group)** → leave-experiment-out, the
real train→test shift. *(Compliance: size used only to build folds; all images are resized to a
fixed shape so the model sees pixels only.)*

## 4. Model
- **Backbone:** ConvNeXt-V2 (`convnextv2_nano/tiny.fcmae_ft_in22k_in1k`) — CNNs beat transformers at
  this data scale; FCMAE pretraining specifically helps fine-tuning.
- **Pooling:** **GeM** (learnable, > global-average for texture/fine-structure).
- **Two heads from the pooled feature:**
  - **SORD soft-ordinal head** over K=69 bins spanning [0,68]: target = `softmax(−(center−y)²/2σ²)`,
    trained with soft cross-entropy. Produces a **per-image PMF over burden** — the object the
    decision layer needs. Soft labels also regularize and absorb ±1-bin label noise.
  - **Regression head** (sigmoid·100, BCE-on-[0,1]): a robust point estimate and ensemble member.
- **Input:** resize to a tall fixed shape (320×192 for iteration → 384×224 final), ImageNet norm.

## 5. The decision layer — the metric exploit (biggest single lever)
Because the metric is fully known, the optimal output minimizes **posterior expected cost**:
`a* = argmin_a Σ_y p(y)·L(a,y)`, with `L` the exact published metric and `p(y)` the model's PMF.
We:
1. **Ensemble** PMFs across folds/seeds (+ hflip TTA) and **temperature-calibrate** on OOF.
2. Choose the output per `metric.expected_cost_decision`, **or** a tuned `blend_thresh` decision
   (`decision_opt.py`) that places the zone cutpoints (esp. the Z2/Z3 boundary) on a blended
   `w·pmf_exp + (1−w)·reg` score — chosen by whichever wins on OOF, with a per-fold stability check
   to avoid overfitting the decision.

This decouples "get the zone right" (0.80 of the score) from raw calibration and directly targets
the hard boundary. Always-safe: clip to the observed train range.

## 6. Training recipe (OOD-robust — the challenge's explicit warning)
- **Augment LIGHTLY, preserve physics:** horizontal flip **only** (vertical order is physical:
  solids sink, headspace rises); **light** colour/contrast/hue jitter (≈0.3) + a touch of grayscale;
  mild blur/noise for glare/compression; **C-Mixup** (mix only close-burden pairs — vanilla mixup is
  invalid for a continuous target). **Empirically, going harder hurt**: MixStyle and strong colour
  augmentation *worsened* OOD here, because — contrary to the usual "destroy the colour-palette
  shortcut" advice — **colour/intensity is the genuine turbidity signal**, not a spurious shortcut.
- **Optimizer:** AdamW, cosine schedule + warmup, differential LR (head > backbone), AMP, **EMA**
  weights (flat minima; ConvNeXt is LayerNorm so no BN-recompute needed).
- **Loss:** soft-CE (SORD) + 0.3·BCE(regression).

## 7. Inference
Average fold (and seed/backbone) PMFs + regression with hflip TTA → temperature → tuned decision →
clip → RLE-free CSV (`id,interface_burden`). Strict `validate_submission.py` before any submit.

## 8. Results (honest leave-experiment-out 5-fold OOF on H100 — lower is better)
| Run | Config | OOF score | zone_acc | Notes |
|---|---|---|---|---|
| baseline-constant | predict 50.5 | 45.21 | 0.38 | trivial floor |
| oracle | true-zone centers | 0.82 | 1.00 | ceiling from the zone term |
| nano320 | convnextv2_nano 320×192 18ep b32 | **24.96** | 0.601 | best single config |
| tiny384 | convnextv2_tiny 384×224 18ep | 31.41 | 0.530 | bigger+hi-res OVERFITS experiments → worse OOD |
| +MixStyle | nano320 + MixStyle | 27.43 | 0.578 | hurts (perturbs the turbidity stats = signal) |
| +strong-aug+reg | nano320 + MixStyle+colour+drop_path+WD | 28.66 | 0.581 | worst — aggressive aug destroys the colour/intensity cue |
| batch16 | nano320 b16 | 27.05 | 0.592 | b32 better |
| femto | convnextv2_femto 320×192 | 25.59 | **0.621** | close 2nd, adds diversity |
| **FINAL ensemble** | 3×nano + 3×femto, OOF-avg + decision + TTA | **_<fill>_** | _<fill>_ | multi-seed variance reduction |

*Caveat: per-fold variance is large (±~12) — single-fold numbers (e.g. an early fold scored 14) are
not representative; only the full 5-fold OOF is trustworthy.*

## 9. What worked / what didn't
- **Worked:** size-based leave-experiment-out CV (honest, reproducible from pixels-only); SORD PMF +
  dual head; EMA; the **expected-cost decision** (robustly beats reg/expectation); **multi-seed
  ensembling** (the main reliable lever given high CV variance); the model genuinely learns the
  turbidity/layering signal (Spearman ~0.74 on held-out experiments).
- **Didn't work (important negative results):** **bigger models / higher resolution OVERFIT the
  training experiments → worse OOD**; **MixStyle and aggressive colour/style augmentation HURT** —
  unlike typical "kill the colour-palette shortcut" advice, here colour/intensity *is* the genuine
  turbidity signal, so destroying it removes signal. Smaller batch (16) and stochastic-depth did not
  help. ⇒ the winning recipe is a **small backbone + light augmentation + ensembling**, not capacity
  or heavy regularization.
- **Bug caught:** OOF index misalignment (local vs global) silently produced worse-than-constant
  scores (56.1); fixed. And the decision tuner initially overfit OOF (degenerate cuts) — fixed to
  select on the per-fold **held-out** score. Lesson: always sanity-check OOF with Spearman + a zone
  confusion matrix, and select post-processing on held-out, never full-OOF.

## 10. Compliance
Learned deep model on the provided public data only. **No** id/row-order/size/path-pattern shortcuts
(size is used solely to build CV folds; the model input is a fixed-shape image). No external data,
no web/reverse-image lookup, no internet at inference. The submitted CSV is reproduced end-to-end by
the official pipeline (`solution_core.py` / notebook): read `dataset/public` → predict → write
`working/submission.csv`.

## 11. Reproduce
```
# CV (train all folds, write OOF + fold models)
MODE=cv N_FOLDS=5 EPOCHS=16 IMG_H=320 IMG_W=192 BACKBONE=convnextv2_nano.fcmae_ft_in22k_in1k python solution_core.py
python decision_opt.py            # tune decision on OOF -> working/decision.json
MODE=predict python solution_core.py    # write working/submission.csv
python validate_submission.py working/submission.csv
```

*Time spent: ~ (to fill).*
