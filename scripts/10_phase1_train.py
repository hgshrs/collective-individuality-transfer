"""Phase 1 main experiment: 10 folds x 3 seeds, decoder a (teacher weights) or b (rotation likelihood).
Usage: python scripts/10_phase1_train.py --decoder a [--z_dim 4] [--exp phase1_a] [--device cuda] [--folds 0-9] [--seeds 0,1,2]"""
import argparse, json, yaml, numpy as np, pandas as pd, torch
from collective_id.paths import ROOT, RUNS, CONFIGS
from collective_id.phase1 import Phase1Data, train_fold
ap = argparse.ArgumentParser()
ap.add_argument("--decoder", default="a"); ap.add_argument("--z_dim", type=int, default=None); ap.add_argument("--exp", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--folds", default="0-9")
ap.add_argument("--teacher", default=None); ap.add_argument("--max_hours", type=float, default=10); ap.add_argument("--max_epochs", type=int, default=None); ap.add_argument("--arch", default="cnn"); ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (yaml-parsed)")
a = ap.parse_args()
cfg = yaml.safe_load(open(CONFIGS / "phase1.yaml"))
if a.z_dim: cfg["z_dim"] = a.z_dim
if a.teacher: cfg["teacher"] = a.teacher
if a.max_epochs: cfg["max_epochs"] = a.max_epochs
cfg["arch"] = a.arch
for kv in a.set:
    k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
exp = a.exp or f"phase1_{a.decoder}_z{cfg['z_dim']}" + ("" if a.arch == "cnn" else f"_{a.arch}")
seeds = [int(s) for s in a.seeds.split(",")]; f0, f1 = (int(x) for x in a.folds.split("-")) if "-" in a.folds else (int(a.folds), int(a.folds))
D = Phase1Data(cfg); print("participants:", len(D.participants), "device:", a.device)
import time; t0 = time.time()
for seed in seeds:
    for fold in range(f0, f1 + 1):
        od = RUNS / exp / f"seed{seed}" / f"fold{fold}"
        if (od / "metrics.json").exists(): print("skip", od); continue
        res, _ = train_fold(D, cfg, fold, seed, a.decoder, a.device, od, verbose=True)
        if (time.time() - t0) / 3600 > a.max_hours: print("max_hours reached"); raise SystemExit
json.dump(cfg, open(RUNS / exp / "config.json", "w"), indent=1); print("done", exp)
