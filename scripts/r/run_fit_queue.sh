#!/usr/bin/env bash
# Sequential queue of brms fits (after tracking_all finishes): search_all, then split-half fits for the noise ceiling.
cd "$(dirname "$0")/../.."
R="$HOME/micromamba/bin/micromamba run -r $HOME/micromamba -n rbrms Rscript scripts/r/fit_rotation_models.R"
while pgrep -f "tag tracking_all" > /dev/null; do sleep 60; done
$R --model search   --half all    --chains 4 --iter 2500 --warmup 1000 --threads 5 --tag search_all      > logs/brms_search_all.log 2>&1
$R --model tracking --half first  --chains 4 --iter 2000 --warmup 1000 --threads 5 --tag tracking_first  > logs/brms_tracking_first.log 2>&1
$R --model tracking --half second --chains 4 --iter 2000 --warmup 1000 --threads 5 --tag tracking_second > logs/brms_tracking_second.log 2>&1
$R --model search   --half first  --chains 4 --iter 2000 --warmup 1000 --threads 5 --tag search_first    > logs/brms_search_first.log 2>&1
$R --model search   --half second --chains 4 --iter 2000 --warmup 1000 --threads 5 --tag search_second   > logs/brms_search_second.log 2>&1
echo QUEUE_DONE > logs/brms_queue_done.flag
