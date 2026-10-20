"""Phase 2 Plan C (AGENT_SPEC §5.2, exploratory): VP participants (n = 165) — predict voluntary payoff sharing from z.
z sources: (a) Phase 1 linear search-phase z (results/phase1/z_linear_pls4_ridge_eb.csv; out-of-fold PLS scores),
(b) Plan B first-3-min own-movement z (PCA-4, out-of-fold; targets restricted to t >= 180 for this z),
(c) tracking-phase random effects (EB teacher, 6 coefficients) as the tracking-phase z.
Controls: final score only, tracking time only, speed only. Ridge on logit-transformed targets within the fixed group-wise 10-fold split (LOPO + Pearson r is dominated by the held-out-mean anti-correlation artefact for weak predictors). Effect sizes with bootstrap 95 % CI; no pass/fail.
C-H2: correlation of z dimensions with payoff-selectivity weights (teacher quality interactions) and Plan B in-degree.
Usage: python scripts/32_phase2_planC.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from collective_id.paths import RESULTS, RUNS
from collective_id.phase2 import load_phase2, oof_ridge, boot_r_ci, STAT_COLS

OUT = RESULTS / "phase2"; OUT.mkdir(parents=True, exist_ok=True)
F3, T, fold = load_phase2()
vp = T[T.cond == "VP"].copy(); parts = list(vp.index)
cv = fold.loc[parts]   # fixed group-wise 10-fold split (LOPO with Pearson r suffers from the held-out-mean anti-correlation artefact; see report)
Zs = pd.read_csv(RESULTS / "phase1" / "z_linear_pls4_ridge_eb.csv", index_col=0)
Zb = pd.read_csv(OUT / "z_planB_pca4_oof.csv", index_col=0)
RE = pd.read_csv(RUNS / "eb" / "tracking_all_ranef.csv").pivot(index="participant", columns="coef", values="estimate")
RE.columns = ["re_" + c.replace("withSocInfo1:", "").replace(":", "_") for c in RE.columns]
ex = pd.DataFrame({"s_slow": (vp.speed == "slow").astype(float)})
TARGETS = {"share_all": "all", "share_track": "all", "share_search": "all", "share_post_entry": "all", "share_post_disc": "all", "share_rest": "first3"}
SOURCES = {"speed_only": None, "score_only": vp[["final_score"]], "track_time_only": vp[["time_on_resource"]],
           "z_search_linear": Zs.reindex(parts), "z_first3_pca4": Zb.reindex(parts), "re_tracking6": RE.reindex(parts),
           "first3_allstats": F3.reindex(parts)[STAT_COLS]}
rows, preds = [], {}
for y in TARGETS:
    yy = vp[y].astype(float); yl = np.log(yy.clip(0.005, 0.995) / (1 - yy.clip(0.005, 0.995)))   # ridge is fitted on the logit (sharing is strongly skewed); evaluation on the raw scale
    for name, X in SOURCES.items():
        if X is not None:
            ok = X.notna().all(1); sub = [p for p in parts if ok[p]]
        else: sub = parts
        pred = oof_ridge(X.loc[sub] if X is not None else None, yl, cv, sub, extra=ex.loc[sub])
        preds[(y, name)] = pred
        r, lo, hi, n = boot_r_ci(pred, yy.loc[sub]); rows.append(dict(target=y, source=name, r=r, ci_lo=lo, ci_hi=hi, n=n))
        # Spearman (primary: sharing is strongly skewed), with participant bootstrap CI
        m = pred.notna() & yy.loc[sub].notna(); pv, yv = pred[m].to_numpy(), yy.loc[sub][m].to_numpy(); rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(pv), len(pv)); bs.append(stats.spearmanr(pv[i], yv[i])[0] if pv[i].std() > 0 else 0.0)
        rows[-1].update(rho=float(stats.spearmanr(pv, yv)[0]) if m.sum() > 5 else np.nan, rho_lo=float(np.nanquantile(bs, .025)), rho_hi=float(np.nanquantile(bs, .975)))
R = pd.DataFrame(rows); R.to_csv(OUT / "planC_r.csv", index=False)
pd.DataFrame({f"{y}|{s}": v for (y, s), v in preds.items()}).to_csv(OUT / "planC_oof_predictions.csv")
# ---- C-H2: correlation table ----
grp = T[T.cond != "A"]
tab = []
for zname, Z in (("z_search_linear", Zs), ("z_first3_pca4", Zb)):
    for zc in Z.columns:
        d = dict(z=f"{zname}:{zc}")
        for tname, series in (("sel_pos (RE PosInFOV:SocQual)", RE["re_SocPosInFOV_SocQual"]), ("sel_mov (RE MovDir:SocQual)", RE["re_SocMovDir_SocQual"]),
                              ("in_deg (groups)", grp["in_deg"]), ("out_deg (groups)", grp["out_deg"]), ("share_all (VP)", vp["share_all"])):
            m = Z[zc].reindex(series.index).notna() & series.notna(); x = Z[zc].reindex(series.index)[m]; yv = series[m]
            rho, p = stats.spearmanr(x, yv); d[tname] = f"{rho:.2f} (p={p:.3f}, n={m.sum()})"
        tab.append(d)
C2 = pd.DataFrame(tab); C2.to_csv(OUT / "planC_H2_correlations.csv", index=False)
# descriptive: sharing vs score / tracking time (the trivial explanation)
desc = {k: float(stats.spearmanr(vp[k].fillna(vp[k].median()), vp.share_all)[0]) for k in ["final_score", "time_on_resource"]}
# figure
fig, ax = plt.subplots(figsize=(9, 4))
piv = R.pivot(index="target", columns="source", values="rho")[list(SOURCES)]; piv.plot.bar(ax=ax, width=0.85); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("out-of-fold Spearman ρ"); ax.set_title("Plan C: voluntary sharing (VP, exploratory)"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_planC.png", dpi=150); fig.savefig(OUT / "fig_planC.pdf")
pd.set_option("display.width", 220); print(R.round(3).to_string()); print(C2.to_string()); print("spearman share_all vs", desc)
