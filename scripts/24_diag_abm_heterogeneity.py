"""Pre-Phase-3 diagnostic 1: is the paper's ABM sensitive to individual heterogeneity of the size we can actually
measure (tracking-phase Intercept and Detector random effects)?

Mapping of the paper's statistical model (|rot| ~ Exp(mean = exp(a_i + b_i * det))) onto the ABM tracking rule
(rot = I_i + S_i * det, degrees): I_i = 50 * exp(u_i), S_i = 35 * exp(u_i) * (1 + v_i / b_cell), where u_i and v_i are
the participant's EB random effects for Intercept and Detector and b_cell the condition x speed fixed effect of Detector.
Selectivity k and the other ABM parameters are held at the Fig 5B values (k = 1 unless stated).

Part A (real groups): each of the 103 analysed groups is simulated with (i) its members' mapped parameters
('hetero'), (ii) all members at the group's mean random effects ('homo_mean'), (iii) default agents ('homo_default').
Part B (parametric sweep): 5 agents with random effects drawn from the empirical pool and scaled by
s in {0, 0.5, 1, 2}, k in {1, 10}, speed in {0.4, 0.8}.

Usage: python scripts/24_diag_abm_heterogeneity.py [--seeds 100] [--seeds_b 200] [--procs 20]
Outputs: results/diagnostics/abm_het_real_groups.csv, abm_het_sweep.csv, abm_het_summary.json, figures.
"""
import argparse, sys, time, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS, EXTERNAL, PROCESSED, RUNS
sys.path.insert(0, str(EXTERNAL))
from agent_based_model.abm import SearcherResourceModel

BASE = dict(width=400, height=400, resource_movement='random', noize_degree=10, p_s=0.15, p_u=0.35,
            search_max_rotation=90, social_info_quality_sensitivity_intercept=0, persistence=0,
            p_perceive_social_info_quality=1, social_alignment='position', social_info_quality_func='sigmoid')
I0, S0 = 50.0, 35.0
OUT = RESULTS / "diagnostics"; OUT.mkdir(parents=True, exist_ok=True)


def map_params(u, v, b_cell):
    """u = Intercept RE, v = Detector RE (arrays) -> (intercepts deg, slopes deg)."""
    I = I0 * np.exp(u)
    S = np.clip(S0 * np.exp(u) * (1 + v / b_cell), 0.0, None)
    return I, S


def run_one(job):
    tag, speed, k, I, S, seed = job
    np.random.seed(seed)
    n = len(I)
    m = SearcherResourceModel(N=n + 1, resource_speed=speed, social_info_quality_sensitivity_slope=k,
                              rotation_tracking_intercept=[float(x) for x in I],
                              detector_sensitivity_slope=[float(x) for x in S], **BASE)
    for _ in range(900):
        m.step()
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    last = df[(df.Step == df.Step.max()) & (df.AgentID != 0)]
    score = last.score.to_numpy(float); ton = last.n_times_on_res.to_numpy(float)
    return dict(tag=tag, speed=speed, k=k, n=n, seed=seed, score=float(score.mean()),
                tracking_time=float(ton.mean() / 900), pcr=float(np.nanmean(score / np.where(ton > 0, ton, np.nan))),
                score_sd_within=float(score.std()))


def load_re():
    r = pd.read_csv(RUNS / "eb" / "tracking_all_ranef.csv").pivot(index="participant", columns="coef", values="estimate")
    fx = pd.read_csv(RUNS / "eb" / "tracking_all_fixef.csv")
    b = {}
    for _, row in fx.iterrows():
        parts = row.coef.split(":")
        if parts[-1] == "Detector" and len(parts) == 3:
            b[(parts[0].replace("SignalingType", ""), parts[1].replace("ResourceSpeed", ""))] = float(row.Estimate)
    meta = pd.read_parquet(PROCESSED / "features_1hz.parquet", columns=["participant", "group", "cond", "speed"]).drop_duplicates("participant").set_index("participant")
    return r, b, meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=100); ap.add_argument("--seeds_b", type=int, default=200)
    ap.add_argument("--procs", type=int, default=20); ap.add_argument("--k", type=float, default=1.0)
    a = ap.parse_args()
    r, b, meta = load_re()
    # ---------------- Part A: real groups ----------------
    jobs, ginfo = [], []
    grp_members = meta[meta.cond != "A"].groupby("group")
    for g, mm in grp_members:
        ids = [p for p in mm.index if p in r.index]
        cond, speed = mm.cond.iloc[0], mm.speed.iloc[0]
        sp = 0.4 if speed == "slow" else 0.8
        u = r.loc[ids, "Intercept"].to_numpy(); v = r.loc[ids, "Detector"].to_numpy()
        I, S = map_params(u, v, b[(cond, speed)])
        Im, Sm = map_params(np.full(len(ids), u.mean()), np.full(len(ids), v.mean()), b[(cond, speed)])
        ginfo.append(dict(group=g, cond=cond, speed=speed, n=len(ids), u_mean=u.mean(), u_sd=u.std(ddof=0), v_mean=v.mean(), v_sd=v.std(ddof=0),
                          I_mean=I.mean(), I_sd=I.std(ddof=0), S_mean=S.mean(), S_sd=S.std(ddof=0)))
        for s in range(a.seeds):
            seed = 100000 + 1000 * len(ginfo) + s
            jobs.append((f"{g}|hetero", sp, a.k, I, S, seed))
            jobs.append((f"{g}|homo_mean", sp, a.k, Im, Sm, seed))
            jobs.append((f"{g}|homo_default", sp, a.k, np.full(len(ids), I0), np.full(len(ids), S0), seed))
    ginfo = pd.DataFrame(ginfo); ginfo.to_csv(OUT / "abm_het_group_params.csv", index=False)
    print(f"Part A: {len(jobs)} runs ({len(ginfo)} groups x 3 x {a.seeds} seeds)", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, res in enumerate(ex.map(run_one, jobs, chunksize=8)):
            rows.append(res)
            if i % 2000 == 0: print(f"  {i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    A = pd.DataFrame(rows); A[["group", "variant"]] = A.tag.str.split("|", expand=True)
    A = A.merge(ginfo, on="group", suffixes=("", "_g")); A.to_csv(OUT / "abm_het_real_groups.csv", index=False)
    print(f"Part A done in {time.time()-t0:.0f}s", flush=True)

    # ---------------- Part B: parametric sweep ----------------
    pool = r[["Intercept", "Detector"]].to_numpy(); rng = np.random.default_rng(0)
    b_mean = float(np.mean(list(b.values())))
    jobs = []
    for s_scale in (0.0, 0.5, 1.0, 2.0):
        for k in (1.0, 10.0):
            for sp in (0.4, 0.8):
                for s in range(a.seeds_b):
                    idx = rng.integers(0, len(pool), 5); u = pool[idx, 0] * s_scale; v = pool[idx, 1] * s_scale
                    I, S = map_params(u, v, b_mean)
                    jobs.append((f"sweep|{s_scale}", sp, k, I, S, 500000 + s))
    print(f"Part B: {len(jobs)} runs", flush=True); t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for res in ex.map(run_one, jobs, chunksize=8): rows.append(res)
    B = pd.DataFrame(rows); B["scale"] = B.tag.str.split("|").str[1].astype(float); B.to_csv(OUT / "abm_het_sweep.csv", index=False)
    print(f"Part B done in {time.time()-t0:.0f}s", flush=True)
    summarise(A, B, ginfo, a)


def icc_group(df, y):
    """One-way ICC(1) of group identity across seeds, computed within speed x n cells and pooled by run count."""
    out = []
    for (sp, n), d in df.groupby(["speed", "n"]):
        g = d.groupby("group")[y]; k = g.size().mean(); G = g.ngroups
        msb = k * g.mean().var(ddof=1); msw = g.var(ddof=1).mean()
        icc = (msb - msw) / (msb + (k - 1) * msw)
        out.append(dict(speed=sp, n=n, n_groups=G, icc=float(icc), sd_between_of_means=float(g.mean().std(ddof=1)), se_mean=float(np.sqrt(msw / k))))
    return pd.DataFrame(out)


def summarise(A, B, ginfo, a):
    import scipy.stats as st
    summ = {}
    # 1. between-group ICC under each variant
    icc = []
    for v, d in A.groupby("variant"):
        for y in ("score", "tracking_time", "pcr"):
            t = icc_group(d, y); t["variant"] = v; t["metric"] = y; icc.append(t)
    icc = pd.concat(icc); icc.to_csv(OUT / "abm_het_icc.csv", index=False)
    # 2. per-group seed means and paired differences
    gm = A.groupby(["group", "variant"])[["score", "tracking_time", "pcr"]].mean().unstack("variant")
    gm.columns = [f"{m}_{v}" for m, v in gm.columns]; gm = gm.reset_index().merge(ginfo, on="group")
    for m in ("score", "tracking_time", "pcr"):
        gm[f"{m}_d_het_mean"] = gm[f"{m}_hetero"] - gm[f"{m}_homo_mean"]
        gm[f"{m}_d_het_def"] = gm[f"{m}_hetero"] - gm[f"{m}_homo_default"]
        gm[f"{m}_d_mean_def"] = gm[f"{m}_homo_mean"] - gm[f"{m}_homo_default"]
    gm.to_csv(OUT / "abm_het_group_means.csv", index=False)
    corr = []
    for sp in ("slow", "fast"):
        d = gm[gm.speed == sp]
        for m in ("score", "tracking_time", "pcr"):
            # composition effect: (hetero - homo_mean) vs within-group SD of parameters
            for x in ("u_sd", "v_sd", "u_mean", "v_mean"):
                for dcol in (f"{m}_d_het_mean", f"{m}_d_het_def", f"{m}_hetero"):
                    rr, p = st.pearsonr(d[x], d[dcol]); corr.append(dict(speed=sp, metric=m, x=x, y=dcol, r=rr, p=p, n=len(d)))
            t, p = st.ttest_rel(d[f"{m}_hetero"], d[f"{m}_homo_mean"])
            corr.append(dict(speed=sp, metric=m, x="paired_t_hetero_vs_homo_mean", y=f"{m}", r=float(d[f"{m}_d_het_mean"].mean()), p=p, n=len(d)))
    corr = pd.DataFrame(corr); corr.to_csv(OUT / "abm_het_correlations.csv", index=False)
    # 3. sweep summary
    sw = B.groupby(["scale", "k", "speed"]).agg(score=("score", "mean"), score_se=("score", lambda x: x.std() / np.sqrt(len(x))),
                                               tracking_time=("tracking_time", "mean"), tt_se=("tracking_time", lambda x: x.std() / np.sqrt(len(x))),
                                               pcr=("pcr", "mean"), pcr_se=("pcr", lambda x: x.std() / np.sqrt(len(x))), n=("seed", "size")).reset_index()
    sw.to_csv(OUT / "abm_het_sweep_summary.csv", index=False)
    summ["icc"] = icc.to_dict("records"); summ["sweep"] = sw.to_dict("records"); summ["seeds"] = a.seeds; summ["k_partA"] = a.k
    json.dump(summ, open(OUT / "abm_het_summary.json", "w"), indent=1, default=float)
    # figures
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, m in enumerate(("score", "tracking_time", "pcr")):
        ax = axes[0, j]
        for sp, c in (("slow", "tab:blue"), ("fast", "tab:red")):
            d = gm[gm.speed == sp]
            ax.scatter(d[f"{m}_homo_mean"], d[f"{m}_hetero"], c=c, s=18, label=sp, alpha=.7)
        lim = [gm[[f"{m}_homo_mean", f"{m}_hetero"]].min().min(), gm[[f"{m}_homo_mean", f"{m}_hetero"]].max().max()]
        ax.plot(lim, lim, "k--", lw=.7); ax.set_xlabel(f"{m}: homogeneous (group-mean agents)"); ax.set_ylabel(f"{m}: heterogeneous (members)")
        if j == 0: ax.legend()
        ax = axes[1, j]
        for k, ls in ((1.0, "-"), (10.0, "--")):
            for sp, c in ((0.4, "tab:blue"), (0.8, "tab:red")):
                d = sw[(sw.k == k) & (sw.speed == sp)]
                se = d[{"score": "score_se", "tracking_time": "tt_se", "pcr": "pcr_se"}[m]]
                ax.errorbar(d["scale"], d[m], yerr=1.96 * se, color=c, ls=ls, marker="o", label=f"k={k:g}, speed={sp}")
        ax.set_xlabel("heterogeneity scale (x measured RE SD)"); ax.set_ylabel(m)
        if j == 0: ax.legend(fontsize=7)
    fig.suptitle("Diagnostic 1: ABM sensitivity to measured heterogeneity (top: real groups; bottom: parametric sweep)")
    fig.tight_layout(); fig.savefig(OUT / "fig_abm_heterogeneity.png", dpi=150); fig.savefig(OUT / "fig_abm_heterogeneity.pdf")
    print(icc.pivot_table(index=["metric", "variant"], columns=["speed", "n"], values="icc").round(3))
    print(corr[corr.x.isin(["u_sd", "v_sd"]) & corr.y.str.endswith("d_het_mean")].round(3).to_string())
    print(sw.round(3).to_string())


if __name__ == "__main__":
    main()
