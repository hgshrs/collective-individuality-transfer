"""Summarise the learning-rate sensitivity sweep (no early stopping): r per coefficient vs lr, training/validation curves."""
import glob, json, sys, numpy as np, pandas as pd
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "phase1_a_z4_eb_lr"   # e.g. phase1_a_z4_eb_ep10k_lr
TAG = sys.argv[2] if len(sys.argv) > 2 else "lr_sweep"
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS, RUNS
out = RESULTS / "phase1"; rows, curves = [], {}
base = pd.read_csv(out / "summary_phase1_a_z4_eb.csv").set_index("coef")     # early-stopped default (lr 1e-3, 3 seeds)
for f in sorted(glob.glob(str(out / f"summary_{PREFIX}*.csv"))):
    lr = float(f.split("_lr")[1].replace(".csv", "")); s = pd.read_csv(f).set_index("coef")
    for c in s.index: rows.append(dict(lr=lr, coef=c, r=s.loc[c, "r_mean"]))
    hs = [pd.read_csv(h) for h in glob.glob(str(RUNS / f"{PREFIX}{f.split('_lr')[1].replace('.csv','')}" / "seed0" / "fold*" / "history.csv"))]
    H = pd.concat(hs).groupby("epoch").agg(train=("train_loss", "mean"), val=("val_fit", "mean")); curves[lr] = H
R = pd.DataFrame(rows); T = R.pivot(index="lr", columns="coef", values="r").sort_index(ascending=False)
T["mean_r"] = T.mean(1); T.loc["early-stop 1e-3 (3 seeds)"] = list(base.loc[T.columns[:-1], "r_mean"]) + [base.r_mean.mean()]
ridge = pd.read_csv(out / "baselines_ridge_eb.csv"); rr = ridge[ridge.model == "ridge_summary_cond"].set_index("coef").r
T.loc["ridge (linear baseline)"] = list(rr.loc[T.columns[:-1]]) + [rr.loc[T.columns[:-1]].mean()]
T.to_csv(out / f"{TAG}_summary.csv"); print(T.round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
lrs = sorted(curves); ax = axs[0]
for lr in lrs: ax.plot(curves[lr].index, curves[lr].train, label=f"lr={lr:g}")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.set_title("training loss (mean over 10 folds)"); ax.legend(fontsize=7)
ax = axs[1]
for lr in lrs: ax.plot(curves[lr].index, curves[lr].val, label=f"lr={lr:g}")
ax.set_xlabel("epoch"); ax.set_ylabel("validation fit loss"); ax.set_title("validation loss (weighted MSE, standardised targets)"); ax.legend(fontsize=7)
ax = axs[2]; Tn = T.loc[[i for i in T.index if isinstance(i, float)]]
for c in Tn.columns[:-1]: ax.plot(Tn.index, Tn[c], "o-", label=c.replace("withSocInfo1:", ""))
ax.plot(Tn.index, Tn.mean_r, "k--", lw=2, label="mean"); ax.axhline(T.loc["ridge (linear baseline)", "mean_r"], color="gray", ls=":", label="ridge mean")
ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("held-out r (final epoch)"); ax.set_title(f"transfer r vs learning rate (no early stopping, {TAG})"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(out / f"fig_{TAG}.png", dpi=150); fig.savefig(out / f"fig_{TAG}.pdf")
