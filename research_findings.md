# Research Findings — distilled, ranked, cited

Synthesis of a 3-agent deep literature review (ordinal-metric optimization; OOD/group generalization; domain CV + fine-tune recipes). Ranked by leverage for THIS metric/data. Confidence noted.

## A. The metric IS the loss — decision theory is the biggest lever [HIGH]
The metric is known, piecewise-constant, asymmetric. The Bayes-optimal action minimizes posterior expected cost — **not** the argmax zone and **not** the regression mean.
- Per-image: `a* = argmin_a Σ_y p(y)·L(a,y)`, with `L` the exact published metric. Grid `a∈[0,100]`, `p(y)` = model's PMF over a fine burden grid. Linearity of expectation makes every term (zone/MAE/MAE-high/extreme) an exact expectation → exact grid search. *(Elkan, Foundations of Cost-Sensitive Learning, IJCAI 2001 https://cseweb.ucsd.edu/~elkan/rescale.pdf; Duda-Hart Bayes decision rule.)*
- Sanity limits: pure-L1 term → posterior median; asymmetric per-unit cost → a τ-quantile τ=c_under/(c_under+c_over); pure zone cost → Elkan thresholds (not 0.5).
- **Precondition = calibration.** Temperature scaling on OOF logits (1 param, safe at this data size) before the decision. *(Guo et al., On Calibration, ICML 2017 https://arxiv.org/abs/1706.04599)*
- Competition precedent for "regression → tune ordinal thresholds on OOF": Diabetic Retinopathy 2015 o_O (regression, threshold 0.5/1.5/2.5/3.5 → 0.845); APTOS 2019 (thresholds helped only ~+0.003, noisy → don't overfit them); OptimizedRounder (Nelder-Mead on cutpoints) is the point-estimate special case of the expected-cost search. Always tune decisions on grouped OOF, never on public LB.

## B. Head / formulation — emit a usable distribution [HIGH]
- **SORD (Soft Ordinal labels)** — top pick: relabel target as softmax(−φ(distance)) over bins, train with CE → standard softmax PMF for the decision step; soft labels regularize on small data; φ can mirror the 60/100 asymmetry. *(Díaz & Marathe, CVPR 2019.)* Caveat: SORD softmax can be under-confident → calibrate.
- **CORN/CORAL** — monotone rank-consistent CDF → clean PMF; CORN ≥ CORAL (dataset-dependent). `coral-pytorch`. Not calibrated by construction.
- **Classification-then-expectation (DEX/DLDL)** — softmax over bins, predict Σ p·center; DLDL = Gaussian soft target, reduces small-data overfit.
- **Continuous head:** normalize target ÷100, **sigmoid + BCE** (PetFinder recipe, beats raw-MSE for bounded [0,1] target); Huber/smooth-L1 if noisy. Keep head small (GeM→dropout 0.1→linear).
- **Decision:** prefer a fine-bin PMF (handles zone + within-zone MAE jointly) decoded by the expected-cost search; cross-check with an OptimizedRounder fallback.

## C. Honest CV + OOD generalization [HIGH]
- **Group inference** (done): here exact image size is a near-perfect experiment id (intra-burden-std 0.68). General playbook (pHash ∪ DINOv2-cosine → connected components) over-merges into a giant component — guard against transitive chaining. We used size+near-size only.
- **StratifiedGroupKFold on (zone × group)**; also eyeball LeaveOneGroupOut spread (mean ± cross-group std = the OOD-gap estimate). *(sklearn.)*
- **Adversarial validation** train-vs-test (AUC≈0.5 ⇒ no shift) to quantify distribution shift and trust local CV.
- **Augmentation to kill shortcuts (the challenge's exact warning):**
  - Strong color jitter + hue + RandomGrayscale(p≈0.1–0.2): contrastive/texture-bias work shows nets cheat via color/texture; destroying it forces structure. *(Geirhos texture-bias ICLR 2019; SimCLR.)* [HIGH, OOD-proven]
  - **MixStyle** (mix feature mean/std across samples) at EARLY layers only (last-block collapses it): PACS LOO +3.3–4.2. *(Zhou et al., ICLR 2021.)* [HIGH] — apply cautiously to ConvNeXt (LayerNorm); A/B it.
  - **AugMix** + JS-consistency for corruption robustness. [HIGH]
  - **C-Mixup** for the continuous target — sample mixing partners by label closeness (Gaussian kernel on burden distance); vanilla mixup is unsafe for regression. +~5.8% OOD. *(Yao et al., NeurIPS 2022.)* [HIGH]
  - **DOMAIN RULE: horizontal flip OK; vertical flip / large rotation NOT** — they scramble the physical vertical phase order (solids bottom, headspace top). [HIGH, domain]
  - CutMix / large Random-Erasing can destroy thin interfaces → use sparingly/small. EMA/SWA for flat minima (ConvNeXt LayerNorm ⇒ no BN-recompute headache). [MED-HIGH]
- **Soft/noisy ordinal labels:** DLDL Gaussian soft labels or CORN absorb ±1-bin jitter; Huber/Barron robust loss on the decoded value; (heavy noise) ConFrag is regression-native clean-selection — classification noise methods (co-teaching/GCE/ELR) do NOT transfer to regression. [MED-HIGH]

## D. Backbone / resolution / pooling [HIGH]
- **CNNs > transformers at ~5k images.** Top pick **ConvNeXt-V2-Tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k`); drop to `convnextv2_nano` if Tiny overfits. EfficientNetV2-S / SwinV2-T are 2nd tier here. *(“Which Backbone to Use” arXiv:2406.05612; Battle of the Backbones.)*
- **Resolution:** iterate at 224, finalize at 384 warm-started from 224 (~+1–1.5pt); fine particulate/thin bands are small-scale → higher res helps. Given tall aspect, consider a tall input (e.g., 384×224 or 448×256) to preserve vertical detail with less distortion.
- **GeM pooling** unconditionally (cheap upgrade over GAP, p≈2–3). Optional texture branch (iSQRT-COV / bilinear) for turbidity-scattering; multi-scale/row-band features for vertical layering. [MED]
- **Schedule:** AdamW + cosine/one-cycle, layer-wise LR decay ~0.65–0.75, short warmup, EMA, AMP, batch 16–32. [HIGH]

## E. TTA + ensembling [HIGH for ensembling, MED for TTA]
- **Ensembling is the biggest reliable lever:** 5-fold OOF bag + heterogeneous 2nd backbone (CNN+transformer decorrelate errors). Average PMFs/values; greedy (Caruana) blend on OOF metric.
- **TTA minor & can backfire** (esp. BN nets; ConvNeXt LayerNorm safer): hflip + maybe mild multi-scale only, validate on OOF before trusting; aggregate by mean/median; always include the clean view.
- **Always-safe post-processing:** clip predictions to observed train-label range.

## Build order
1. Pipeline + grouped CV + the exact-metric expected-cost evaluator wired in first (score everything the way the LB does).
2. ConvNeXt-V2-Nano/Tiny, GeM, BCE-regression + SORD head, OOD aug stack.
3. Temperature-calibrate OOF PMF → expected-cost decision (+ OptimizedRounder cross-check).
4. Ensemble folds/seeds (+2nd backbone); validated hflip TTA; clip to range.
5. Resolution 224→384; final A100 sweep.

## Flagged / not-load-bearing
- Texture-bias exact % are figure-derived; duplicate-leak inflation has no single number; backbone/pooling magnitudes are classification results (transfer in direction, not magnitude); CORAL/CORN "calibrated" is FALSE (rank-consistent only).
