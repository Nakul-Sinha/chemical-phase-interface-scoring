#!/bin/bash
source /mnt/chem/env.sh
cd /mnt/chem/code
export DATA_ROOT=/mnt/chem/data PYTHONUNBUFFERED=1
RES=/mnt/chem/working/results.txt
echo "config,cv_score,zone_acc,decision" > $RES
run() {
  NAME=$1; shift
  export OUT_DIR=/mnt/chem/working/$NAME; mkdir -p $OUT_DIR
  echo "===== $NAME start $(date +%H:%M:%S) ====="
  env "$@" MODE=cv python solution_core.py > $OUT_DIR/run.log 2>&1
  OUT_DIR=$OUT_DIR python decision_opt.py > $OUT_DIR/decision.log 2>&1
  grep -E "=== OOF|score=|spearman|zone_acc" $OUT_DIR/run.log | tail -4
  echo "-- tuned decision:"; grep -E "SAVED|mean held-out|blend_thresh.*score" $OUT_DIR/decision.log | tail -3
  S=$(python -c "import json;d=json.load(open(\"$OUT_DIR/cv_result.json\"));print(round(d[\"score\"],3),round(d[\"zone_acc\"],3))")
  DEC=$(python -c "import json;print(json.load(open(\"$OUT_DIR/decision.json\")))" 2>/dev/null)
  echo "$NAME,$S,$DEC" >> $RES
  echo "===== $NAME done $(date +%H:%M:%S) ====="
}
run nano320 N_FOLDS=5 EPOCHS=18 BATCH=32 IMG_H=320 IMG_W=192 NUM_WORKERS=16 BACKBONE=convnextv2_nano.fcmae_ft_in22k_in1k
run tiny384  N_FOLDS=5 EPOCHS=18 BATCH=32 IMG_H=384 IMG_W=224 NUM_WORKERS=16 BACKBONE=convnextv2_tiny.fcmae_ft_in22k_in1k
run base384  N_FOLDS=5 EPOCHS=16 BATCH=24 IMG_H=384 IMG_W=224 NUM_WORKERS=16 BACKBONE=convnextv2_base.fcmae_ft_in22k_in1k
echo "ALL DONE $(date)"; cat $RES
