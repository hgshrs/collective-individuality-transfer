#!/usr/bin/env bash
# Resume the 50k-epoch sweep (folds already finished are skipped), then evaluate. Long --max_hours.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; T=runs/eb/tracking_all_ranef.csv
run() { lr=$1; $PY scripts/10_phase1_train.py --decoder a --z_dim 4 --teacher $T --seeds 0 --exp phase1_a_z4_eb_ep50k_lr$lr --set lr=$lr no_early_stop=true max_epochs=50000 --max_hours 100 > logs/phase1_lr50k_resume_$lr.log 2>&1 && $PY scripts/12_phase1_eval.py --exp phase1_a_z4_eb_ep50k_lr$lr --tag ridge_eb > logs/phase1_lr50k_eval_$lr.log 2>&1; }
for lr in 1e-3 3e-4 1e-4 3e-5 1e-5 1e-6; do run $lr & done; wait
echo LR50K_RESUME_DONE
