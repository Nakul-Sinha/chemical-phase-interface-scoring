# Chemical Phase Interface Assessment

## Overview
Transparent-vessel chemistry images often show more than a single visible state. A mixture may contain clear liquid, turbid liquid, settled solids, residue on the vessel wall, and empty headspace at the same time. For process monitoring, the useful decision is not just whether one state is present, but how severe the visible phase-interface condition is for downstream control.

This is an image-based chemistry assessment task. Given one still image of a chemical vessel, assign an `interface_burden` index from 0 to 100. Higher values indicate a more difficult visual state: more heterogeneous liquid, more solid or residue burden, more vertical phase transitions, and more suspended material away from the vessel bottom. Evaluation emphasizes the correct ordinal burden zone, not just small numeric adjustments. The test set holds out entire experiment groups, so robust solutions should learn visual phase and interface cues rather than memorizing image style, color palette, or repeated video frames.

## Dataset

### File descriptions
- `train.csv` -- 4,770 labeled chemistry images with image paths and reference `interface_burden` index values.
- `test.csv` -- 1,261 held-out chemistry images with image paths only.
- `sample_submission.csv` -- A template showing the required submission format with `id` and `interface_burden` columns, filled with random baseline values.
- `images/` -- Directory containing the JPEG vessel images referenced by train.csv and test.csv.

### Column descriptions
- `id` (string) -- Unique 12-character hexadecimal identifier for each image.
- `image_path` (string) -- Relative path from `dataset/public/` to the JPEG image.
- `interface_burden` (float) -- Reference visual burden index from 0 to 100, present only in train.csv. Higher values indicate greater visible phase-interface burden.

## Evaluation
Submissions are evaluated using **ordinal interface-zone calibration loss. Lower is better.** The metric rewards placing each vessel image in the correct burden zone, with small calibration terms for numeric closeness and high-burden cases. It also penalizes severe inversions, such as marking a nearly clear vessel as heavy-burden or missing a visibly high-burden vessel.

```python
import numpy as np

def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 100)
    severity_bins = np.array([0.0, 12.0, 35.0, 48.0, 100.000001])
    absolute_gap = np.abs(y_pred - y_true)
    high_burden = y_true >= 48.0
    absolute_component = absolute_gap.mean()
    high_component = absolute_gap[high_burden].mean() if high_burden.any() else absolute_component
    true_zone = np.digitize(y_true, severity_bins) - 1
    pred_zone = np.digitize(y_pred, severity_bins) - 1
    zone_distance = np.abs(true_zone - pred_zone)
    zone_penalty = np.where(zone_distance == 0, 0.0, np.where(zone_distance == 1, 60.0, 100.0))
    zone_component = zone_penalty.mean()
    extreme_miss = ((y_true <= 12.0) & (y_pred > 25.0)) | ((y_true >= 48.0) & (y_pred < 40.0))
    extreme_component = extreme_miss.mean() * 100.0
    score = (
        0.05 * absolute_component
        + 0.05 * high_component
        + 0.80 * zone_component
        + 0.10 * extreme_component
    )
    return float(np.clip(score, 0.0, 100.0))
```

## Submission
Submit a CSV file with an `interface_burden` index for every row in test.csv.
- `id` (string) -- The 12-character hexadecimal identifier from test.csv.
- `interface_burden` (float) -- Your assessed interface-burden index for the image, between 0 and 100.

Example:
```
id,interface_burden
0000c416ea24,17.2
00624db32e6c,21.5
00acf0be9394,46.8
```

## Requirements
- The file must contain exactly 1,261 rows plus the header, one for each observation in test.csv.
- Every id from test.csv must be present exactly once.
- All submitted values must be finite numeric values. Values below 0 or above 100 are clipped to [0, 100] before scoring.
- File format: .csv only, with exact column names `id,interface_burden`.

## What Not To Use
- Do not hard-code submitted values for public test IDs, row order, or hashed path patterns.
- Use only the files provided in `dataset/public/` during solution execution. Do not use web search, reverse-image search, external repositories, or internet lookups to identify images or recover held-out values.
- Do not submit a purely deterministic color-threshold or row-order rule that ignores the labeled training rows. Rule-based image features are fine as preprocessing, but the core model should be learned from the provided training data.
