#!/bin/bash
# Final: multi-seed ensemble of the 2 best base configs (nano320 + femto320),
# fixed fold split, average OOF -> honest decision -> ensemble test prediction.
source /mnt/chem/env.sh; cd /mnt/chem/code
export DATA_ROOT=/mnt/chem/data PYTHONUNBUFFERED=1
DIRS=""
seedrun(){ local NM=$1; shift
  for S in 42 43 44; do
    D=/mnt/chem/working/$NM-s$S; mkdir -p $D
    echo "===== $NM seed $S $(date +%H:%M:%S) ====="
    env "$@" OUT_DIR=$D SEED=$S FOLD_SEED=42 MODE=cv python solution_core.py > $D/run.log 2>&1
    grep -E "expected-cost score=" $D/run.log | tail -1
    DIRS="$DIRS $D"
  done
}
seedrun nano  BACKBONE=convnextv2_nano.fcmae_ft_in22k_in1k IMG_H=320 IMG_W=192 EPOCHS=18 BATCH=32 N_FOLDS=5 NUM_WORKERS=14
seedrun femto BACKBONE=convnextv2_femto.fcmae_ft_in1k       IMG_H=320 IMG_W=192 EPOCHS=20 BATCH=32 N_FOLDS=5 NUM_WORKERS=14
FINAL=/mnt/chem/working/final; mkdir -p $FINAL
echo "===== ensemble OOF + decision ====="
python ensemble_oof.py $DIRS $FINAL > $FINAL/ens.log 2>&1; cat $FINAL/ens.log
OUT_DIR=$FINAL python decision_opt.py > $FINAL/decision.log 2>&1
grep -E "expected_cost held-out|blend_thresh  held-out|SAVED|honest held-out|full-OOF" $FINAL/decision.log | tail -5
echo "===== ensemble predict ====="
python ensemble_predict.py $DIRS $FINAL > $FINAL/predict.log 2>&1
grep -E "wrote|pred zone|ensembling" $FINAL/predict.log
echo "ALL DONE $(date +%H:%M:%S)"
