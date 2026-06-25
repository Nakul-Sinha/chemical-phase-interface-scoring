# Notes — Chemical Phase Interface Assessment

Working log of challenge facts, EDA, validation, decisions, and runs.

## Challenge facts
- **Task:** image regression — predict `interface_burden` ∈ [0,100] from ONE RGB photo of a chemical vessel. Higher = more turbid/heterogeneous liquid, more solids/residue, more vertical phase transitions, more suspended material.
- **Data:** 4,770 train (id, image_path, interface_burden) / 1,261 test (id, image_path). 6,031 JPEGs in `images/`.
- **Metric (LOWER is better), dominated by ORDINAL ZONE accuracy:**
  - zones from bins `[0,12,35,48,100]` → 4 ordinal zones {0,1,2,3}.
  - `score = 0.80·zone_penalty + 0.05·MAE + 0.05·MAE(y_true≥48) + 0.10·extreme_miss·100`
  - zone_penalty = 0 (same zone) / 60 (off-by-one) / 100 (off-by-≥2), averaged.
  - extreme_miss = (y≤12 & pred>25) OR (y≥48 & pred<40).
  - ⇒ **≈80% of the score is 4-class ordinal zone accuracy.** Numeric closeness barely matters.
- **Generalization:** test holds out ENTIRE experiment groups → must learn visual phase cues, not memorize style/color/repeated frames.
- **Compliance:** public data only; no web/reverse-image lookups; no hard-coding from id/row-order/**hashed path patterns**; rule-based features OK as preprocessing but the core predictor must be a learned model on the provided training data.

## EDA findings (see eda/out/)
- **Labels cap at ~67** (not 100); multimodal with spikes (~4, ~47, ~51, ~62). mean 38.0, median 47.0, std 20.0.
- **Zone distribution** (fairly balanced, imbalance 1.99): Z0 951 (19.9%) / Z1 919 (19.3%) / Z2 1067 (22.4%) / Z3 1833 (38.4%). Z2 [35,48) is the narrowest band.
- **Images:** ALL RGB (color matters). **Tall portrait vessels**, aspect ≈ 0.47, median 336×739, sizes vary widely (123–507 W, 250–1040 H). Vertical structure (phase layers) carries the signal.
- **Visual nature:** glass vials/test-tubes; burden rises with turbidity/milkiness, suspended solids, layering (clear↔turbid bands), dark/heterogeneous regions. Heavy glare/specular reflection on glass = key nuisance. Look like video frames (near-duplicates exist).

## Metric landscape (computed on train labels, eda/eda_labels.py)
- predict mean → 51.2 ; predict median → 46.8 ; **best constant (50.5) → 45.21** (this is the trivial floor to beat).
- **ORACLE perfect-zone-centers → 0.82** ; perfect (exact y) → 0.0.
- ⇒ Whole game is zone accuracy. Improving zone accuracy by 10pp improves score by ~4.8 (0.80·0.60·0.10·100... i.e. 0.80·60·Δ). Target: very high zone accuracy → score in low single digits.

## KEY FINDING — exact image size ≈ experiment id (used for CV only)
- Images sharing the exact (W,H) have **intra-group burden std ≈ 0.68** vs global ~20 ⇒ each experiment was captured at a fixed resolution and burden is ~constant within an experiment.
- **COMPLIANCE:** size/aspect must NOT be a model feature (disallowed metadata shortcut). All images are resized to a FIXED shape so pixels are the only signal. Size is used ONLY to build honest CV groups.
- **CV grouping** (eda/group_infer.py): exact-size unions + merge sizes within 2px (catches one experiment recorded at slightly different crops), NO embedding chaining (that over-merged into a 657-img giant component). Chosen = **near-size@2px → 272 groups**, intra-burden-std 3.14, max group 6.2%, only 12 cross-group near-duplicate pairs (minimal leakage). 232/272 groups are single-zone.
- Saved: `eda/out/train_groups.csv` (id, image_path, interface_burden, group, zone).

## Strategy (from deep research — see research_findings.md)
1. **Honest CV first:** StratifiedGroupKFold on (zone × group) with the 272 groups. Random K-fold would be catastrophically leaky.
2. **Model emits a DISTRIBUTION, not a point:** backbone (ConvNeXt-V2-Tiny/Nano, GeM pool) → (a) regression head BCE-on-[0,1] + (b) SORD soft-ordinal classification head over fine bins → per-image PMF over burden.
3. **THE EXPLOIT — Bayes-optimal decision:** the metric is known & piecewise-constant. Choose the output scalar `a* = argmin_a Σ_y p(y)·L(a,y)` via grid search over the EXACT metric. This is provably optimal and is the single biggest separator. Calibrate the PMF (temperature scaling on OOF) first.
4. **OOD augmentation** (the challenge's explicit warning): destroy color/style shortcuts (strong color jitter/hue, RandomGrayscale); **horizontal flip only — NO vertical flip/rotation** (vertical phase order is physical); C-Mixup (mix only close-burden pairs); mild glare/brightness aug; EMA for flat minima.
5. **Ensemble** folds + seeds (+ 2nd backbone) by averaging PMFs; light hflip TTA (validate, ConvNeXt is LayerNorm so TTA is safer than BN nets).

## Compute
- Local RTX 4050 (6 GB): EDA done; prototyping + small runs.
- Kaggle T4 (16 GB, generous hours): main 5-fold training.
- Lightning A100 (~3 h HARD cap; do not idle): final 384-res sweeps/ensemble.

## CV / Run log
- (to fill) — every run logs: config, OOF metric + component breakdown, zone accuracy, per-fold spread, change vs previous.

## Submissions
- (to fill) — public score vs OOF, exact change per submission.
