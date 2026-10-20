"""Phase 2 Plan B (AGENT_SPEC §5.1): first-3-min own movement -> social role over the remaining 12 min.
Linear z (Phase 1 verdict: non-linear z unnecessary): out-of-fold PCA-4 of own-movement summary statistics; prediction
by ridge within the fixed group-wise 10-fold split; controls = cell mean, first-3-min score only. Also ridge on all
statistics and supervised PLS-4 as linear upper bounds. Variance decomposition with group random intercept.
Usage: python scripts/31_phase2_planB.py
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS
from collective_id.phase2 import load_phase2, onehot, fold_pca_z, oof_ridge, oof_pls, corr, paired_delta, boot_r_ci, decompose, add_composition, STAT_COLS

OUT = RESULTS / "phase2"; OUT.mkdir(parents=True, exist_ok=True)
F3, T, fold = load_phase2()
meta = T[T.cond != "A"].copy(); parts = [p for p in meta.index if p in F3.index and p in fold.index]; meta = meta.loc[parts]
TARGETS = ["in_deg", "out_deg", "in_minus_out", "n_disc_rest", "disc_gain_rest"]
oh = onehot(meta)
Z, ref = fold_pca_z(F3, fold, parts); Z.to_csv(OUT / "z_planB_pca4_oof.csv")
load = pd.DataFrame(ref.components_, columns=STAT_COLS, index=[f"z{i}" for i in range(4)]); load.to_csv(OUT / "z_planB_pca4_loadings.csv")
print("PCA explained variance (all-sample reference):", ref.explained_variance_ratio_.round(3))
score3 = F3.loc[parts, ["score_3min"]]
preds = {}
for y in TARGETS:
    yy = meta[y].astype(float)
    preds[(y, "cell_mean")] = oof_ridge(None, yy, fold, parts, extra=oh)
    preds[(y, "score3_only")] = oof_ridge(score3, yy, fold, parts, extra=oh)
    preds[(y, "z_pca4")] = oof_ridge(Z, yy, fold, parts, extra=oh)
    preds[(y, "z_pca4+score3")] = oof_ridge(Z.join(score3), yy, fold, parts, extra=oh)
    preds[(y, "ridge_allstats")] = oof_ridge(F3.loc[parts, STAT_COLS], yy, fold, parts, extra=oh)
    preds[(y, "ridge_allstats+score3")] = oof_ridge(F3.loc[parts, STAT_COLS + ["score_3min"]], yy, fold, parts, extra=oh)
    preds[(y, "pls4")] = oof_pls(F3.loc[parts, STAT_COLS], yy, fold, parts, oh)
P = pd.DataFrame({f"{y}|{m}": v for (y, m), v in preds.items()}); P.to_csv(OUT / "planB_oof_predictions.csv")
rows, deltas, bycond = [], [], []
for y in TARGETS:
    yy = meta[y].astype(float)
    for m in ["cell_mean", "score3_only", "z_pca4", "z_pca4+score3", "ridge_allstats", "ridge_allstats+score3", "pls4"]:
        r, lo, hi, n = boot_r_ci(preds[(y, m)], yy); rows.append(dict(target=y, model=m, r=r, ci_lo=lo, ci_hi=hi, n=n))
    for m in ["z_pca4", "z_pca4+score3", "ridge_allstats", "ridge_allstats+score3", "pls4"]:
        for b in ["score3_only", "cell_mean"]:
            d = paired_delta(preds[(y, m)], preds[(y, b)], yy, meta.group); d.update(target=y, model=m, baseline=b); deltas.append(d)
    # B-H2: by signalling condition (and speed)
    for m in ["z_pca4", "z_pca4+score3", "score3_only", "ridge_allstats"]:
        for c in ["NP", "VP", "FP"]:
            idx = meta.index[meta.cond == c]; r, lo, hi, n = boot_r_ci(preds[(y, m)].loc[idx], yy.loc[idx]); bycond.append(dict(target=y, model=m, cond=c, r=r, ci_lo=lo, ci_hi=hi, n=n))
        for s in ["fast", "slow"]:
            idx = meta.index[meta.speed == s]; r, lo, hi, n = boot_r_ci(preds[(y, m)].loc[idx], yy.loc[idx]); bycond.append(dict(target=y, model=m, cond=s, r=r, ci_lo=lo, ci_hi=hi, n=n))
R = pd.DataFrame(rows); R.to_csv(OUT / "planB_r.csv", index=False)
D = pd.DataFrame(deltas); D.to_csv(OUT / "planB_delta.csv", index=False)
B = pd.DataFrame(bycond); B.to_csv(OUT / "planB_by_condition.csv", index=False)
# ---- variance decomposition (all-sample PCA z, so that components are identical across participants) ----
X = F3.loc[parts, STAT_COLS]; Zall = pd.DataFrame(ref.transform((X - X.mean()) / (X.std() + 1e-9)), index=parts, columns=[f"z{i}" for i in range(4)])
df = meta.join(Zall).join(score3); df["score_3min_z"] = (df.score_3min - df.score_3min.mean()) / df.score_3min.std(); df = add_composition(df, list(Zall.columns))
dec = []
for y in TARGETS:
    dec.append(decompose(df, y, list(Zall.columns), extra_fixed=[]).assign(control="none"))
    dec.append(decompose(df, y, list(Zall.columns), extra_fixed=["score_3min_z"]).assign(control="score_3min"))
dec = pd.concat(dec); dec.to_csv(OUT / "planB_decomposition.csv", index=False)
# ---- verdict (pre-registered §5.1: z beats score-3-min baseline by Δr >= 0.05, p < 0.01) ----
main = D[(D.model == "z_pca4") & (D.baseline == "score3_only") & D.target.isin(["in_deg", "out_deg"])]
verdict = dict(targets=main.target.tolist(), delta_r=main.delta_r.round(3).tolist(), mixed_p=main.mixed_p.tolist(), p_boot=main.p_boot.tolist(),
               pass_each=[(d >= 0.05) and (p < 0.01) for d, p in zip(main.delta_r, main.mixed_p)])
verdict["B_H1"] = "PASS" if all(verdict["pass_each"]) else ("PARTIAL" if any(verdict["pass_each"]) else "FAIL")
json.dump(verdict, open(OUT / "planB_verdict.json", "w"), indent=1, default=float)
# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]; piv = R.pivot(index="target", columns="model", values="r")[["cell_mean", "score3_only", "z_pca4", "z_pca4+score3", "ridge_allstats", "pls4"]]
piv.plot.bar(ax=ax, width=0.85); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("out-of-fold r"); ax.set_title("Plan B: role prediction"); ax.legend(fontsize=6)
ax = axes[1]
for m, c in (("z_pca4", "tab:blue"), ("score3_only", "tab:gray"), ("ridge_allstats", "tab:green")):
    d = B[(B.model == m) & (B.target == "in_deg") & B.cond.isin(["NP", "VP", "FP"])]; x = np.arange(3) + {"z_pca4": -0.25, "score3_only": 0, "ridge_allstats": 0.25}[m]
    ax.errorbar(x, d.r, yerr=[d.r - d.ci_lo, d.ci_hi - d.r], fmt="o", color=c, label=m, capsize=3)
ax.set_xticks(range(3)); ax.set_xticklabels(["NP", "VP", "FP"]); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("r (in-degree)"); ax.set_title("B-H2: by condition"); ax.legend(fontsize=7)
ax = axes[2]; dd = dec[(dec.control == "none")]
for y, c in zip(TARGETS, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]):
    d = dd[dd.outcome == y]; ax.plot(range(4), d.r2_marginal, marker="o", color=c, label=y)
ax.set_xticks(range(4)); ax.set_xticklabels(["cell", "+own z", "+others mean", "+group SD"], fontsize=8); ax.set_ylabel("marginal R²"); ax.set_title("Variance decomposition (mixed, group RI)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_planB.png", dpi=150); fig.savefig(OUT / "fig_planB.pdf")
pd.set_option("display.width", 220)
print(R.pivot(index="target", columns="model", values="r").round(3)); print(D[D.baseline == "score3_only"].round(3).to_string()); print(B[B.target == "in_deg"].round(3).to_string()); print(dec.round(3).to_string()); print(verdict)
