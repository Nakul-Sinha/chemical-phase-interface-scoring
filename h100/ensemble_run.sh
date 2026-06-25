#!/bin/bash
# Multi-seed ensemble: train the SAME config with 3 seeds (fixed fold split),
# average OOF, tune the decision honestly, then predict test from all 15 models.
# Usage: NAME=final bash ensemble_run.sh BACKBONE=... IMG_H=320 IMG_W=192 EPOCHS=18 BATCH=32 N_FOLDS=5 NUM_WORKERS=16
source /mnt/chem/env.sh; cd /mnt/chem/code
export DATA_ROOT=/mnt/chem/data PYTHONUNBUFFERED=1
NAME=${NAME:-ens}; ENS=/mnt/chem/working/$NAME; mkdir -p $ENS
CFG=("$@"); DIRS=""
for S in 42 43 44; do
  D=/mnt/chem/working/${NAME}_s$S; mkdir -p $D
  echo "===== seed $S $(date +%H:%M:%S) ====="
  env "${CFG[@]}" OUT_DIR=$D SEED=$S FOLD_SEED=42 MODE=cv python solution_core.py > $D/run.log 2>&1
  grep -E "=== OOF|score=|zone_acc" $D/run.log | tail -1
  DIRS="$DIRS $D"
done
echo "===== ensemble OOF + honest decision ====="
python ensemble_oof.py $DIRS $ENS > $ENS/ens.log 2>&1
OUT_DIR=$ENS python decision_opt.py > $ENS/decision.log 2>&1
grep -E "expected_cost held-out|blend_thresh  held-out|SAVED|honest held-out" $ENS/decision.log | tail -4
i=0; for D in $DIRS; do for f in $D/model_f*.pt; do cp "$f" $ENS/model_f${i}.pt; i=$((i+1)); done; done
echo "===== predict test ($i models + hflip TTA) ====="
env "${CFG[@]}" OUT_DIR=$ENS MODE=predict python solution_core.py > $ENS/predict.log 2>&1
grep -E "wrote|pred zone|decision:" $ENS/predict.log | tail -3
echo "ALL DONE $(date +%H:%M:%S)"
