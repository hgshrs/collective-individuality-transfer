"""Phase 3 evaluation (AGENT_SPEC §6.3-4/5, §6.4): predicted (seed-mean) vs observed collective metrics for every
analysed group; main model vs controls; D-H1 / D-H2 tests; composition regression; figures.
Primary correlation = Pearson r pooled over groups after z-scoring predictions and observations WITHIN condition x speed
cell (controls that are constant within a cell therefore get r ≈ 0, as intended: they carry no group-specific
information). Also reported: r within speed, raw pooled r, MAE. Uncertainty: group bootstrap (3,000).
Usage: python scripts/43_phase3_evaluate.py
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from collective_id.paths import RESULTS

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--sim", default="sim_runs.parquet"); ap.add_argument("--suffix", default=""); a = ap.parse_args()
OUT = RESULTS / "phase3"; SUF = a.suffix
S = pd.read_parquet(OUT / a.sim); O = pd.read_csv(OUT / "observed_group_metrics.csv").set_index("group"); GP = pd.read_csv(OUT / "sim_group_params.csv").set_index("group")
METRICS = ["score", "tracking_time", "pcr", "in_out_corr", "in_deg", "out_deg", "med_dist", "disc_gain"]
POINT = [v for v in ["main", "ctrl1_paper", "ctrl1b_pooled", "ctrl1_pop", "ctrl2_avg", "ctrl2b_gmean", "ctrl3_nominal"] if v in set(S.variant)]
HAS_SHUFFLE = "ctrl4_shuffle" in set(S.variant)
rng = np.random.default_rng(0)

# ---------- aggregate predictions ----------
pred = {}
for v in POINT:
    d = S[S.variant == v]
    pred[v] = d.groupby("group")[[m for m in METRICS if m in d.columns and d[m].notna().any()]].mean()
draws = {v: S[S.variant == v].groupby(["group", "draw"])[METRICS].mean() for v in ("ctrl4_shuffle", "ctrl4b_mps") if v in set(S.variant)}
for v in draws: pred[v] = draws[v].groupby("group").mean()
groups = sorted(set(O.index) & set(pred["main"].index)); O = O.loc[groups]
cell = O.cell


def zcell(s):
    s = pd.Series(s, index=groups, dtype=float)
    return s.groupby(cell).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))


def r_of(p, o):
    m = p.notna() & o.notna()
    return float(np.corrcoef(p[m], o[m])[0, 1]) if m.sum() > 3 and p[m].std() > 0 else np.nan


rows = []
for v in pred:
    for met in METRICS:
        if met not in pred[v].columns: continue
        p = pred[v][met].reindex(groups); o = O[met]
        pz, oz = zcell(p), zcell(o)
        bs = []
        for _ in range(3000):
            i = rng.integers(0, len(groups), len(groups)); bs.append(r_of(pz.iloc[i].reset_index(drop=True), oz.iloc[i].reset_index(drop=True)))
        rows.append(dict(variant=v, metric=met, r_within_cell=r_of(pz, oz), ci_lo=float(np.nanquantile(bs, .025)), ci_hi=float(np.nanquantile(bs, .975)),
                         r_slow=r_of(p[O.speed == "slow"], o[O.speed == "slow"]), r_fast=r_of(p[O.speed == "fast"], o[O.speed == "fast"]), r_raw=r_of(p, o),
                         mae=float((p - o).abs().mean()), mae_z=float((pz - oz).abs().mean()), n=int((p.notna() & o.notna()).sum())))
R = pd.DataFrame(rows); R.to_csv(OUT / f"eval_r{SUF}.csv", index=False)

# ---------- paired comparisons: main vs each control (Δr within-cell z; group bootstrap; sign-flip permutation on squared error) ----------
rows = []
for v in [x for x in pred if x != "main"]:
    for met in METRICS:
        if met not in pred[v].columns or met not in pred["main"].columns: continue
        pm, pc, o = zcell(pred["main"][met].reindex(groups)), zcell(pred[v][met].reindex(groups)), zcell(O[met])
        m = pm.notna() & pc.notna() & o.notna(); pm, pc, o = pm[m].to_numpy(), pc[m].to_numpy(), o[m].to_numpy(); n = m.sum()
        if n < 10: continue
        obs = r_of(pd.Series(pm), pd.Series(o)) - (r_of(pd.Series(pc), pd.Series(o)) if np.std(pc) > 0 else 0.0)
        bs = []
        for _ in range(3000):
            i = rng.integers(0, n, n); rc = np.corrcoef(pc[i], o[i])[0, 1] if np.std(pc[i]) > 0 else 0.0
            bs.append(np.corrcoef(pm[i], o[i])[0, 1] - rc)
        bs = np.array(bs); p_boot = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        d_err = (pc - o) ** 2 - (pm - o) ** 2; perm = []
        for _ in range(5000): perm.append((d_err * rng.choice([-1, 1], n)).mean())
        p_perm = float((np.abs(perm) >= abs(d_err.mean())).mean())
        rows.append(dict(control=v, metric=met, delta_r=obs, ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)), p_boot=float(p_boot), d_sqerr=float(d_err.mean()), p_perm=p_perm, n=int(n)))
D = pd.DataFrame(rows); D.to_csv(OUT / f"eval_delta{SUF}.csv", index=False)

# ---------- D-H2: shuffle tests (null distribution of r from randomly re-composed groups) ----------
rows = []
for v in [x for x in ("ctrl4_shuffle", "ctrl4b_mps") if x in draws]:
    dr = draws[v]
    for met in METRICS:
        o = zcell(O[met]); pm = zcell(pred["main"][met].reindex(groups)); r_main = r_of(pm, o)
        gl = [g for g in groups if g in dr.index.get_level_values(0)]
        null = []
        for _ in range(2000):
            pick = pd.Series({g: dr.loc[g, met].sample(1, random_state=int(rng.integers(1e9))).iloc[0] for g in gl}).reindex(groups)
            null.append(r_of(zcell(pick), o))
        null = np.array(null)
        # per-group error comparison: |pred_main - obs| vs mean_d |pred_shuffle_d - obs| (z scale)
        e_main = (pm - o).abs(); e_sh = pd.Series({g: (zcell(dr.loc[g, met].reindex(groups, fill_value=np.nan)) if False else np.nan) for g in gl})
        em, es = [], []
        for g in gl:
            if np.isnan(o[g]) or np.isnan(pm[g]): continue
            em.append(abs(pm[g] - o[g])); es.append(np.nanmean(np.abs(((dr.loc[g, met] - pred[v][met].reindex(groups).groupby(cell).transform("mean")[g]) / (pred[v][met].reindex(groups).groupby(cell).transform(lambda x: x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)[g])) - o[g])))
        rows.append(dict(shuffle=v, metric=met, r_main=r_main, r_shuffle_mean=float(np.nanmean(null)), r_shuffle_q95=float(np.nanquantile(null, .95)), p_perm=float(np.mean(null >= r_main)), n_groups=len(gl),
                         err_main=float(np.mean(em)), err_shuffle=float(np.mean(es))))
SH = pd.DataFrame(rows); SH.to_csv(OUT / f"shuffle_test{SUF}.csv", index=False)

# ---------- composition regression: observed ~ nominal (or group mean params) + parameter SDs ----------
rows = []
sdcols = [c for c in GP.columns if c.endswith("_sd")]; mcols = [c for c in GP.columns if c.endswith("_mean") and c != "sigma_mean"]
for met in ["score", "tracking_time", "pcr"]:
    o = zcell(O[met]); X0 = pd.DataFrame({"nominal": zcell(pred["ctrl3_nominal"][met].reindex(groups))}, index=groups)
    Xm = GP.loc[groups, mcols].apply(lambda s: (s - s.mean()) / s.std()); Xs = GP.loc[groups, sdcols].apply(lambda s: (s - s.mean()) / s.std())
    for base_name, Xb in (("nominal", X0), ("group_mean_params", Xm)):
        m0 = sm.OLS(o, sm.add_constant(Xb)).fit(); m1 = sm.OLS(o, sm.add_constant(pd.concat([Xb, Xs], axis=1))).fit()
        rows.append(dict(metric=met, base=base_name, r2_base=m0.rsquared, r2_plus_sd=m1.rsquared, adj_r2_base=m0.rsquared_adj, adj_r2_plus_sd=m1.rsquared_adj, p_sd_block=float(m1.compare_f_test(m0)[1]), n=int(m1.nobs),
                         **{f"b_{c}": float(m1.params[c]) for c in sdcols}, **{f"p_{c}": float(m1.pvalues[c]) for c in sdcols}))
CR = pd.DataFrame(rows); CR.to_csv(OUT / f"composition_regression{SUF}.csv", index=False)

# ---------- verdicts (§6.4) ----------
def get(control, met, col):
    d = D[(D.control == control) & (D.metric == met)]; return float(d[col].iloc[0]) if len(d) else np.nan
v = {}
for met in ("score", "pcr"):
    for c in [x for x in ("ctrl1_paper", "ctrl1_pop", "ctrl2_avg") if x in POINT]:
        v[f"{met}_vs_{c}"] = dict(delta_r=get(c, met, "delta_r"), p_boot=get(c, met, "p_boot"), p_perm=get(c, met, "p_perm"), passes=bool(get(c, met, "delta_r") >= 0.10 and get(c, met, "p_boot") < 0.01))
v["D_H1"] = "PASS" if all(v[k]["passes"] for k in v if k != "D_H1") else "FAIL"
if len(SH):
    sh = SH.set_index(["shuffle", "metric"])
    v["D_H2_shuffle"] = {met: dict(p_perm=float(sh.loc[("ctrl4_shuffle", met), "p_perm"]), r_main=float(sh.loc[("ctrl4_shuffle", met), "r_main"]), r_shuffle_mean=float(sh.loc[("ctrl4_shuffle", met), "r_shuffle_mean"])) for met in ("score", "pcr", "tracking_time")}
    v["D_H2_mean_preserving"] = {met: dict(p_perm=float(sh.loc[("ctrl4b_mps", met), "p_perm"]), r_shuffle_mean=float(sh.loc[("ctrl4b_mps", met), "r_shuffle_mean"])) for met in ("score", "pcr", "tracking_time")}
    v["D_H2"] = "PASS" if all(v["D_H2_shuffle"][m]["p_perm"] < 0.01 for m in ("score", "pcr")) else "FAIL"
    v["D_H2_mps"] = "PASS" if all(v["D_H2_mean_preserving"][m]["p_perm"] < 0.01 for m in ("score", "pcr")) else "FAIL"
# substantive composition test (decision 2026-10-20): group-mean-agent control
v["D_H2_groupmean_control"] = {met: dict(delta_r=get("ctrl2b_gmean", met, "delta_r"), ci_lo=get("ctrl2b_gmean", met, "ci_lo"), ci_hi=get("ctrl2b_gmean", met, "ci_hi"), p_boot=get("ctrl2b_gmean", met, "p_boot")) for met in ("score", "pcr", "tracking_time")}
json.dump(v, open(OUT / f"verdict{SUF}.json", "w"), indent=1, default=float)

# ---------- figures ----------
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for ax, met in zip(axes.flat, METRICS):
    p = pred["main"][met].reindex(groups) if met in pred["main"].columns else pd.Series(np.nan, index=groups); o = O[met]
    for spd, c in (("slow", "tab:blue"), ("fast", "tab:red")):
        m = O.speed == spd; ax.scatter(p[m], o[m], c=c, s=18, alpha=.7, label=spd)
    rr = R[(R.variant == "main") & (R.metric == met)]
    ax.set_title(f"{met}: r_within-cell={rr.r_within_cell.iloc[0]:.2f}  slow={rr.r_slow.iloc[0]:.2f} fast={rr.r_fast.iloc[0]:.2f}", fontsize=9); ax.set_xlabel("predicted (main, seed mean)"); ax.set_ylabel("observed")
axes[0, 0].legend(); fig.suptitle("Phase 3 D-1 (reduced): leave-one-group-out prediction of collective metrics from members' individually fitted ABM policies"); fig.tight_layout(); fig.savefig(OUT / f"fig_pred_vs_obs{SUF}.png", dpi=150); fig.savefig(OUT / f"fig_pred_vs_obs{SUF}.pdf")
fig, ax = plt.subplots(figsize=(11, 4.5))
piv = R.pivot(index="metric", columns="variant", values="r_within_cell").reindex(METRICS); piv = piv[[c for c in ["main", "ctrl2b_gmean", "ctrl3_nominal", "ctrl4_shuffle", "ctrl4b_mps", "ctrl1_paper", "ctrl1b_pooled", "ctrl1_pop", "ctrl2_avg"] if c in piv.columns]]
piv.plot.bar(ax=ax, width=0.85); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("r (pooled, z within cell)"); ax.set_title("Prediction of observed collective metrics: main model vs controls"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / f"fig_controls{SUF}.png", dpi=150); fig.savefig(OUT / f"fig_controls{SUF}.pdf")
pd.set_option("display.width", 250)
print(R.pivot(index="metric", columns="variant", values="r_within_cell").round(3)); print(R[R.variant == "main"].round(3).to_string()); print(D[D.metric.isin(["score", "pcr", "tracking_time"])].round(3).to_string()); print(SH.round(3).to_string()); print(CR.round(3).to_string()); print(json.dumps(v, indent=1, default=float))
