#!/usr/bin/env bash
# Learning-rate sensitivity (decoder a, z=4, no early stopping, 500 epochs, seed 0, 10 folds). 3 runs in parallel on the GPU.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; T=runs/eb/tracking_all_ranef.csv
run() { lr=$1; $PY scripts/10_phase1_train.py --decoder a --z_dim 4 --teacher $T --seeds 0 --exp phase1_a_z4_eb_lr$lr --set lr=$lr no_early_stop=true > logs/phase1_lr_$lr.log 2>&1 && $PY scripts/12_phase1_eval.py --exp phase1_a_z4_eb_lr$lr --tag ridge_eb > logs/phase1_lr_eval_$lr.log 2>&1; }
run 1e-3 & run 3e-4 & run 1e-4 & wait
run 3e-5 & run 1e-5 & run 1e-6 & wait
echo LR_SWEEP_DONE
