"""Pre-Phase-3 diagnostic 2: in the observed data, does the *composition* of measured individual parameters explain
group performance beyond the group's mean parameters?

Design (no simulation): individual tracking-phase random effects are estimated from the FIRST half of the session
(runs/eb/tracking_first_ranef.csv); group outcomes are computed from the SECOND half (t >= 450 s): mean per-member
gross score (40 - d points/s inside 40 m) and tracking-time proportion. Outcomes are z-scored within condition x speed.

Group-level nested OLS (n = 103 groups):
  M0: cell + n_members
  M1: M0 + group MEAN of the 6 random effects
  M2: M1 + group SD of the 6 random effects                 (composition)
  M3: M2 + first-half group outcome                         (strict control for group-session persistence)
Reported: adjusted R2, leave-one-group-out CV R2, nested F-test p, and a membership-permutation null for the
composition increment (participants re-assigned to groups within cell, 2000 permutations).
Also a 2-RE version (Intercept, Detector only: the reliably measured dimensions).

Individual-level mixed model (n = 477 group participants): second-half score ~ cell + own RE + others' mean RE +
others' SD RE + (1 | group); LRT for the others' terms.

Usage: python scripts/25_diag_composition_observed.py
Outputs: results/diagnostics/composition_*.csv, composition_summary.json, fig_composition.png
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS, PROCESSED, RUNS

OUT = RESULTS / "diagnostics"; OUT.mkdir(parents=True, exist_ok=True)
COEFS = ["Intercept", "Detector", "withSocInfo1:SocPosInFOV", "withSocInfo1:SocMovDir", "withSocInfo1:SocPosInFOV:SocQual", "withSocInfo1:SocMovDir:SocQual"]
SHORT = dict(zip(COEFS, ["int", "det", "pos", "mov", "posq", "movq"]))
T_SPLIT = 450
rng = np.random.default_rng(0)


def load():
    f = pd.read_parquet(PROCESSED / "features_1hz.parquet", columns=["participant", "group", "cond", "speed", "t", "state", "dist_raw"])
    f = f[f.cond != "A"].copy()
    f["pts"] = np.where(f.dist_raw <= 40, 40 - f.dist_raw, 0.0); f["trk"] = (f.state == "tracking").astype(float)
    f["half"] = np.where(f.t < T_SPLIT, "h1", "h2")
    ind = f.groupby(["participant", "group", "cond", "speed", "half"]).agg(score=("pts", "sum"), track=("trk", "mean"), n_t=("t", "size")).reset_index()
    ind = ind.pivot_table(index=["participant", "group", "cond", "speed"], columns="half", values=["score", "track"]); ind.columns = [f"{a}_{b}" for a, b in ind.columns]; ind = ind.reset_index()
    re = pd.read_csv(RUNS / "eb" / "tracking_first_ranef.csv").pivot(index="participant", columns="coef", values="estimate")[COEFS]
    re.columns = [SHORT[c] for c in COEFS]
    ind = ind.merge(re, left_on="participant", right_index=True, how="inner")
    ind["cell"] = ind.cond + "_" + ind.speed
    return ind


def group_table(ind):
    g = ind.groupby(["group", "cond", "speed", "cell"])
    G = g.agg(n=("participant", "size"), y_score=("score_h2", "mean"), y_track=("track_h2", "mean"), y1_score=("score_h1", "mean"), y1_track=("track_h1", "mean"),
              **{f"m_{c}": (c, "mean") for c in SHORT.values()}, **{f"s_{c}": (c, lambda x: x.std(ddof=0)) for c in SHORT.values()}).reset_index()
    for y in ("y_score", "y_track", "y1_score", "y1_track"):
        G[y + "_z"] = G.groupby("cell")[y].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    return G


def fit_ols(G, y, terms):
    X = pd.get_dummies(G[["cell"]], drop_first=True).astype(float); X["n"] = G.n
    for t in terms: X[t] = G[t]
    X = sm.add_constant(X); m = sm.OLS(G[y], X).fit()
    # LOO-CV R2
    yv = G[y].to_numpy(); Xv = X.to_numpy(); pred = np.empty(len(yv))
    for i in range(len(yv)):
        msk = np.arange(len(yv)) != i
        beta, *_ = np.linalg.lstsq(Xv[msk], yv[msk], rcond=None); pred[i] = Xv[i] @ beta
    cv_r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return m, float(cv_r2)


def nested(G, y, re_set, label):
    means = [f"m_{c}" for c in re_set]; sds = [f"s_{c}" for c in re_set]
    specs = {"M0_cell": [], "M1_+mean": means, "M2_+sd": means + sds, "M3_+h1": means + sds + [y.replace("y_", "y1_")]}
    fits = {k: fit_ols(G, y, v) for k, v in specs.items()}
    rows, prev = [], None
    for k, (m, cv) in fits.items():
        p_nested = float(m.compare_f_test(prev)[1]) if prev is not None else np.nan
        rows.append(dict(set=label, outcome=y, model=k, r2=m.rsquared, adj_r2=m.rsquared_adj, cv_r2=cv, aic=m.aic, p_vs_previous=p_nested, df_model=m.df_model)); prev = m
    # composition increment beyond mean, also after h1 control
    m_ctrl0, cv0 = fit_ols(G, y, means + [y.replace("y_", "y1_")]); m_ctrl1, cv1 = fit_ols(G, y, means + sds + [y.replace("y_", "y1_")])
    rows.append(dict(set=label, outcome=y, model="SD|mean+h1", r2=m_ctrl1.rsquared, adj_r2=m_ctrl1.rsquared_adj, cv_r2=cv1, aic=m_ctrl1.aic, p_vs_previous=float(m_ctrl1.compare_f_test(m_ctrl0)[1]), df_model=m_ctrl1.df_model, r2_increment=m_ctrl1.rsquared - m_ctrl0.rsquared, cv_r2_increment=cv1 - cv0))
    return pd.DataFrame(rows), fits


def permutation_null(ind, y, re_set, n_perm=2000):
    """Re-assign participants to groups within cell (group sizes preserved); recompute mean/SD; ΔR2 (M2 - M1) and (M1 - M0)."""
    G0 = group_table(ind); means = [f"m_{c}" for c in re_set]; sds = [f"s_{c}" for c in re_set]
    def incs(G):
        m0, _ = fit_ols(G, y, []); m1, _ = fit_ols(G, y, means); m2, _ = fit_ols(G, y, means + sds)
        return m1.rsquared - m0.rsquared, m2.rsquared - m1.rsquared
    obs_mean, obs_sd = incs(G0)
    null = []
    cols = list(SHORT.values()); base = ind.copy()
    for _ in range(n_perm):
        perm = base.copy()
        for cell, idx in base.groupby("cell").indices.items():
            perm.loc[base.index[idx], cols] = base.loc[base.index[rng.permutation(idx)], cols].to_numpy()
        null.append(incs(group_table(perm)))
    null = np.array(null)
    return dict(outcome=y, set="+".join(re_set), obs_dR2_mean=obs_mean, p_mean=float((null[:, 0] >= obs_mean).mean()), null_dR2_mean_q95=float(np.quantile(null[:, 0], .95)),
                obs_dR2_sd=obs_sd, p_sd=float((null[:, 1] >= obs_sd).mean()), null_dR2_sd_q95=float(np.quantile(null[:, 1], .95)), n_perm=n_perm)


def individual_mixed(ind):
    """second-half individual outcome ~ cell + own RE + others' mean RE + group SD RE + (1|group)."""
    cols = list(SHORT.values()); d = ind.copy()
    for c in cols:
        gsum = d.groupby("group")[c].transform("sum"); gn = d.groupby("group")[c].transform("size")
        d[f"om_{c}"] = (gsum - d[c]) / (gn - 1)                      # others' mean (leave-self-out)
        d[f"os_{c}"] = d.groupby("group")[c].transform(lambda x: x.std(ddof=0))  # group SD (a group property; the
        # leave-self-out SD was tried first and gave an anti-conservative LRT, p = 2e-4 vs 0.07 here / 0.076 permutation)
    for y in ("score_h2", "track_h2", "score_h1"):
        d[y + "_z"] = d.groupby("cell")[y].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    rows = []
    for y in ("score_h2_z", "track_h2_z"):
        own = " + ".join(cols); oth_m = " + ".join(f"om_{c}" for c in cols); oth_s = " + ".join(f"os_{c}" for c in cols)
        f0 = f"{y} ~ C(cell) + {own}"; f1 = f0 + " + " + oth_m; f2 = f1 + " + " + oth_s; f3 = f2 + " + score_h1_z"
        fits = [smf.mixedlm(f, d, groups=d.group).fit(reml=False, method="lbfgs") for f in (f0, f1, f2, f3)]
        names = ["own", "+others_mean", "+others_sd", "+h1"]
        for i, m in enumerate(fits):
            lrt = 2 * (m.llf - fits[i - 1].llf) if i else np.nan; dfd = (len(m.params) - len(fits[i - 1].params)) if i else np.nan
            rows.append(dict(outcome=y, model=names[i], llf=m.llf, aic=m.aic, lrt=lrt, df=dfd, p_lrt=float(stats.chi2.sf(lrt, dfd)) if i else np.nan,
                             group_var=float(m.cov_re.iloc[0, 0]), resid_var=float(m.scale)))
        own_p = fits[0].pvalues[cols]; rows.append(dict(outcome=y, model="own_coef_p", **{c: float(own_p[c]) for c in cols}))
        own_b = fits[0].params[cols]; rows.append(dict(outcome=y, model="own_coef_b", **{c: float(own_b[c]) for c in cols}))
    return pd.DataFrame(rows)


def main():
    ind = load(); G = group_table(ind)
    ind.to_csv(OUT / "composition_individuals.csv", index=False); G.to_csv(OUT / "composition_groups.csv", index=False)
    print(f"groups {len(G)} (n=5: {(G.n==5).sum()}, n=4: {(G.n==4).sum()}), participants {len(ind)}")
    tabs, perms = [], []
    for y in ("y_score_z", "y_track_z"):
        for label, rs in (("all6", list(SHORT.values())), ("int+det", ["int", "det"])):
            t, _ = nested(G, y, rs, label); tabs.append(t)
            perms.append(permutation_null(ind, y, rs, n_perm=2000)); print(f"perm done {y} {label}", flush=True)
    tabs = pd.concat(tabs); tabs.to_csv(OUT / "composition_group_models.csv", index=False)
    perms = pd.DataFrame(perms); perms.to_csv(OUT / "composition_permutation.csv", index=False)
    # also: 5-person groups only
    G5 = G[G.n == 5].reset_index(drop=True); t5 = pd.concat([nested(G5, y, ["int", "det"], "int+det_n5")[0] for y in ("y_score_z", "y_track_z")]); t5.to_csv(OUT / "composition_group_models_n5.csv", index=False)
    # simple correlations for the reader
    cor = []
    for y in ("y_score_z", "y_track_z"):
        for x in [c for c in G.columns if c.startswith(("m_", "s_"))] + ["y1_score_z", "y1_track_z"]:
            r, p = stats.pearsonr(G[x], G[y]); cor.append(dict(outcome=y, x=x, r=r, p=p))
    cor = pd.DataFrame(cor); cor.to_csv(OUT / "composition_correlations.csv", index=False)
    mixed = individual_mixed(ind); mixed.to_csv(OUT / "composition_individual_mixed.csv", index=False)
    # reliability context: first vs second half RE
    re2 = pd.read_csv(RUNS / "eb" / "tracking_second_ranef.csv").pivot(index="participant", columns="coef", values="estimate")[COEFS]; re2.columns = [SHORT[c] for c in COEFS]
    rel = {c: float(ind.set_index("participant")[c].corr(re2[c])) for c in SHORT.values()}
    summ = dict(n_groups=int(len(G)), n_groups5=int((G.n == 5).sum()), n_participants=int(len(ind)), split_half_r_group_participants=rel,
                group_models=tabs.to_dict("records"), permutation=perms.to_dict("records"), mixed=mixed.to_dict("records"))
    json.dump(summ, open(OUT / "composition_summary.json", "w"), indent=1, default=float)
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    for lab, mk in (("all6", "o"), ("int+det", "s")):
        for y, c in (("y_score_z", "tab:blue"), ("y_track_z", "tab:orange")):
            t = tabs[(tabs.set == lab) & (tabs.outcome == y) & tabs.model.str.startswith("M")]
            ax.plot(range(4), t.cv_r2, marker=mk, color=c, label=f"{lab} {y[2:-2]}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["cell", "+mean", "+SD", "+h1"]); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("LOO-CV R²"); ax.legend(fontsize=7); ax.set_title("Group-level nested models")
    ax = axes[1]; ax.scatter(G.m_det, G.y_score_z, c=np.where(G.speed == "slow", "tab:blue", "tab:red"), s=18); ax.set_xlabel("group mean Detector RE (h1)"); ax.set_ylabel("group score h2 (z within cell)")
    ax = axes[2]; ax.scatter(G.s_det, G.y_score_z, c=np.where(G.speed == "slow", "tab:blue", "tab:red"), s=18); ax.set_xlabel("group SD Detector RE (h1)"); ax.set_ylabel("group score h2 (z within cell)")
    fig.suptitle("Diagnostic 2: observed composition effects (blue slow, red fast)"); fig.tight_layout(); fig.savefig(OUT / "fig_composition.png", dpi=150); fig.savefig(OUT / "fig_composition.pdf")
    pd.set_option("display.width", 200); print(tabs.round(3).to_string()); print(perms.round(3).to_string()); print(mixed.round(4).to_string()); print("split-half r:", {k: round(v, 2) for k, v in rel.items()})


if __name__ == "__main__":
    main()
