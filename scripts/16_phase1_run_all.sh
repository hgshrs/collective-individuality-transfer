#!/usr/bin/env bash
# Phase 1 pipeline (run after runs/brms/tracking_all_ranef.csv exists). Logs to logs/phase1_*.log
set -e; cd "$(dirname "$0")/.."; PY=.venv/bin/python
TEACHER=${TEACHER:-runs/brms/tracking_all_ranef.csv}; SUF=${SUF:-}
cp $TEACHER data/processed/individual_weights${SUF}.csv
$PY scripts/11_phase1_ridge_baseline.py --teacher $TEACHER --tag ridge$SUF                          > logs/phase1_ridge.log 2>&1
$PY scripts/10_phase1_train.py --decoder a --z_dim 4 --teacher $TEACHER --exp phase1_a_z4$SUF              > logs/phase1_train_a_z4.log 2>&1
$PY scripts/12_phase1_eval.py  --exp phase1_a_z4$SUF --tag ridge$SUF                  > logs/phase1_eval_a_z4.log 2>&1
$PY scripts/10_phase1_train.py --decoder b --z_dim 4 --teacher $TEACHER --exp phase1_b_z4$SUF              > logs/phase1_train_b_z4.log 2>&1
$PY scripts/17_phase1_eval_b.py --exp phase1_b_z4$SUF                 > logs/phase1_eval_b_z4.log 2>&1
$PY scripts/10_phase1_train.py --decoder a --z_dim 2 --teacher $TEACHER --exp phase1_a_z2$SUF              > logs/phase1_train_a_z2.log 2>&1
$PY scripts/12_phase1_eval.py  --exp phase1_a_z2$SUF --tag ridge$SUF                  > logs/phase1_eval_a_z2.log 2>&1
$PY scripts/10_phase1_train.py --decoder a --z_dim 8 --teacher $TEACHER --exp phase1_a_z8$SUF              > logs/phase1_train_a_z8.log 2>&1
$PY scripts/12_phase1_eval.py  --exp phase1_a_z8$SUF --tag ridge$SUF                  > logs/phase1_eval_a_z8.log 2>&1
$PY scripts/10_phase1_train.py --decoder a --z_dim 4 --arch gru --teacher $TEACHER --exp phase1_a_z4_gru$SUF    > logs/phase1_train_a_z4_gru.log 2>&1
$PY scripts/12_phase1_eval.py  --exp phase1_a_z4_gru$SUF --tag ridge$SUF              > logs/phase1_eval_a_z4_gru.log 2>&1
$PY scripts/14_phase1_sim_recovery.py --exp phase1_a_z4$SUF           > logs/phase1_sim_recovery.log 2>&1
$PY scripts/15_phase1_icc.py --exp phase1_a_z4$SUF                    > logs/phase1_icc.log 2>&1
echo PHASE1_PIPELINE_DONE
