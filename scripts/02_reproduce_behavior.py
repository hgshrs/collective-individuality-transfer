"""Phase 0 §3.1 items 2–4: reproduce Fig 2A (performance), Fig 3A/B + Supp Fig 9 (visibility networks),
Fig 3C (resource discovery events) from the paper's processed tables. Means + 95% CI (no Bayesian regressions)."""
import numpy as np, pandas as pd, polars as pl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RAW_DATA, RESULTS
out = RESULTS / "phase0"; out.mkdir(parents=True, exist_ok=True)
ORDER = ["A", "NP", "VP", "FP"]; COL = {"A": "k", "NP": "tab:blue", "VP": "tab:orange", "FP": "tab:red"}

def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

ts = pl.read_csv(RAW_DATA / "processed/time_series_data.csv").to_pandas()
s = pd.read_csv(RAW_DATA / "processed/subject_data.csv")
ids = ts.participant.unique(); s = s[s.player_id.isin(ids)].copy()
s["tracking_time_prop"] = s.time_on_resource / 15
s["tracking_eff_pts_s"] = s.points_collection_speed / 60   # points per second on resource (paper: PCS/60)

# ---- Fig 2A: performance ----
perf = s.groupby(["condition_resource", "condition_signaling"]).agg(
    n=("player_id", "size"),
    score_mean=("final_score", "mean"), score_ci=("final_score", ci95), score_sd=("final_score", "std"),
    ttime_mean=("tracking_time_prop", "mean"), ttime_ci=("tracking_time_prop", ci95),
    teff_mean=("tracking_eff_pts_s", "mean"), teff_ci=("tracking_eff_pts_s", ci95)).reset_index()
perf["condition_signaling"] = pd.Categorical(perf.condition_signaling, ORDER); perf = perf.sort_values(["condition_resource", "condition_signaling"])
perf.to_csv(out / "fig2A_performance_by_condition.csv", index=False)
# paper-direction checks
chk = []
for res, exp in [("slow", "FP>VP>NP>=A"), ("fast", "FP>A>VP>NP")]:
    m = perf[perf.condition_resource == res].set_index("condition_signaling").score_mean
    order = list(m.sort_values(ascending=False).index)
    chk.append({"metric": "total score", "speed": res, "expected": exp, "observed": ">".join(order), "match": ">".join(order).replace(">=", ">") == exp.replace(">=", ">")})
for res in ["slow", "fast"]:
    m = perf[perf.condition_resource == res].set_index("condition_signaling")
    chk.append({"metric": "tracking time", "speed": res, "expected": "groups > A", "observed": ">".join(m.ttime_mean.sort_values(ascending=False).index), "match": bool(all(m.ttime_mean[c] > m.ttime_mean["A"] for c in ["NP", "VP", "FP"]))})
    exp = "FP>A>VP,NP" if res == "slow" else "A>=FP>VP,NP"
    o = ">".join(m.teff_mean.sort_values(ascending=False).index)
    ok = (m.teff_mean["FP"] > m.teff_mean["A"] if res == "slow" else abs(m.teff_mean["FP"] - m.teff_mean["A"]) < 0.5) and m.teff_mean["A"] > max(m.teff_mean["NP"], m.teff_mean["VP"])
    chk.append({"metric": "tracking efficiency", "speed": res, "expected": exp, "observed": o, "match": bool(ok)})
pd.DataFrame(chk).to_csv(out / "fig2A_direction_checks.csv", index=False)
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
for _, r in perf.iterrows():
    mk = "o" if r.condition_resource == "slow" else "s"
    ax.errorbar(r.ttime_mean, r.teff_mean, xerr=r.ttime_ci, yerr=r.teff_ci, fmt=mk, color=COL[r.condition_signaling], capsize=2, label=f"{r.condition_signaling} {r.condition_resource}")
ax.set_xlabel("Tracking time (proportion)"); ax.set_ylabel("Tracking efficiency (points/s on resource)"); ax.legend(fontsize=7, ncol=2); ax.set_title("Fig 2A reproduction (mean ± 95% CI)")
fig.tight_layout(); fig.savefig(out / "fig2A_performance.png", dpi=150); fig.savefig(out / "fig2A_performance.pdf"); plt.close(fig)

# ---- Fig 3A: in/out-degree per condition (group conditions) ----
g = ts[ts.signaling != "A"]
pp = g.groupby(["participant", "signaling", "resource"]).agg(in_deg=("in_degree", "mean"), out_deg=("out_degree", "mean"), score=("participant", "size")).reset_index()
pp = pp.merge(s[["player_id", "final_score", "group"]], left_on="participant", right_on="player_id")
deg = pp.groupby(["resource", "signaling"]).agg(n=("participant", "size"), in_mean=("in_deg", "mean"), in_ci=("in_deg", ci95), out_mean=("out_deg", "mean"), out_ci=("out_deg", ci95)).reset_index()
deg.to_csv(out / "fig3A_degree_by_condition.csv", index=False)
# Supp Fig 9: in vs out correlation (participant level, within condition)
cors = []
for (res, sig), d in pp.groupby(["resource", "signaling"]):
    cors.append({"resource": res, "signaling": sig, "n": len(d), "r_in_out": d.in_deg.corr(d.out_deg),
                 "r_in_score": d.in_deg.corr(d.final_score), "r_out_score": d.out_deg.corr(d.final_score)})
cors = pd.DataFrame(cors); cors.to_csv(out / "suppfig9_in_out_correlation.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, res in zip(axs, ["fast", "slow"]):
    d = deg[deg.resource == res].set_index("signaling").loc[["NP", "VP", "FP"]]
    x = np.arange(3); ax.bar(x - 0.2, d.in_mean, 0.4, yerr=d.in_ci, label="in-degree", capsize=2); ax.bar(x + 0.2, d.out_mean, 0.4, yerr=d.out_ci, label="out-degree", capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(d.index); ax.set_title(f"{res} resource"); ax.set_ylabel("mean degree")
axs[0].legend(); fig.suptitle("Fig 3A reproduction"); fig.tight_layout(); fig.savefig(out / "fig3A_degree.png", dpi=150); fig.savefig(out / "fig3A_degree.pdf"); plt.close(fig)

# ---- Fig 3B: adaptive social information: P(at least one visible other closer than self | ≥1 visible) ----
gg = g[(g.out_degree > 0) & g.state.isin(["tracking", "searching"])].copy()
gg["adaptive"] = (gg.soc_info_quality_raw_best < 0).astype(float)   # best visible other's distance − own distance < 0
pa = gg.groupby(["participant", "signaling", "resource", "state"]).adaptive.mean().reset_index()
ad = pa.groupby(["resource", "state", "signaling"]).agg(n=("participant", "size"), p_mean=("adaptive", "mean"), p_ci=("adaptive", ci95)).reset_index()
ad.to_csv(out / "fig3B_adaptive_social_info.csv", index=False)
contrasts = []
for (res, st), d in ad.groupby(["resource", "state"]):
    m = d.set_index("signaling").p_mean
    contrasts.append({"resource": res, "state": st, "FP-NP": m["FP"] - m["NP"], "FP-VP": m["FP"] - m["VP"]})
pd.DataFrame(contrasts).to_csv(out / "fig3B_contrasts.csv", index=False)

# ---- Fig 3C: resource discovery events (fast): discoverer in-degree, others' out-degree, P(on resource) over time ----
rd = pl.read_csv(RAW_DATA / "processed/resource_discoveries_data.csv").to_pandas()
rd["Timepoint"] = rd.timepoint - 2; rd = rd[rd.Timepoint >= 0]
rd["on_res"] = (rd.distance_from_resource <= 40).astype(float)
rd["role"] = np.where(rd.player_order == 1, "discoverer", "others")
rd["tbin"] = np.floor(rd.Timepoint).astype(int)
ev = rd.groupby(["condition_resource", "condition_signaling"]).events.nunique().reset_index().rename(columns={"events": "n_events"})
ev.to_csv(out / "fig3C_event_counts.csv", index=False)
cur = rd.groupby(["condition_resource", "condition_signaling", "role", "tbin"]).agg(in_deg=("in_degree", "mean"), out_deg=("out_degree", "mean"), p_on=("on_res", "mean"), n=("events", "nunique")).reset_index()
cur.to_csv(out / "fig3C_discovery_curves.csv", index=False)
peaks = []
for (res, sig), d in cur[cur.role == "discoverer"].groupby(["condition_resource", "condition_signaling"]):
    i = d.in_deg.idxmax(); peaks.append({"resource": res, "signaling": sig, "peak_in_degree": d.loc[i, "in_deg"], "peak_time_s": d.loc[i, "tbin"], "in_degree_at_10s": d[d.tbin == 10].in_deg.mean()})
pd.DataFrame(peaks).to_csv(out / "fig3C_discoverer_indegree_peaks.csv", index=False)
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
for sig in ORDER:
    d = cur[(cur.condition_resource == "fast") & (cur.condition_signaling == sig)]
    dd = d[d.role == "discoverer"]; do = d[d.role == "others"]
    axs[0, 0].plot(dd.tbin, dd.p_on, color=COL[sig], label=sig); axs[0, 1].plot(do.tbin, do.p_on, color=COL[sig], label=sig)
    if sig != "A":
        axs[1, 0].plot(dd.tbin, dd.in_deg, color=COL[sig], label=sig); axs[1, 1].plot(do.tbin, do.out_deg, color=COL[sig], label=sig)
axs[0, 0].set_title("P(discoverer on resource)"); axs[0, 1].set_title("P(others on resource)"); axs[1, 0].set_title("discoverer in-degree"); axs[1, 1].set_title("others out-degree")
for ax in axs.flat: ax.set_xlabel("time since discovery (s)"); ax.legend(fontsize=7)
fig.suptitle("Fig 3C reproduction (fast resource; raw means)"); fig.tight_layout(); fig.savefig(out / "fig3C_discovery.png", dpi=150); fig.savefig(out / "fig3C_discovery.pdf"); plt.close(fig)

pd.set_option("display.width", 200)
print(perf.to_string(index=False)); print(pd.DataFrame(chk).to_string(index=False)); print(deg.to_string(index=False)); print(cors.to_string(index=False)); print(ad.to_string(index=False)); print(pd.DataFrame(contrasts).to_string(index=False)); print(ev.to_string(index=False)); print(pd.DataFrame(peaks).to_string(index=False))
