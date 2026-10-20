"""Phase 3 §6.3-7 extrapolation (prediction only, not validated): resource speed {0.2..1.0} x group size {3, 5, 10, 25} x
heterogeneity {homogeneous (mean fitted parameters), sampled from the fitted participant distribution, 2x spread}.
Plus solitary agents (N=2) at each speed for the Fig 5B-style difference. Parameters from scripts/40 (all participants).
Usage: python scripts/44_phase3_extrapolate.py [--seeds 50] [--procs 20]
Output: results/phase3/extrapolation.csv, fig_extrapolation.png
"""
import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS, EXTERNAL
sys.path.insert(0, str(EXTERNAL))
from agent_based_model.abm import SearcherResourceModel

OUT = RESULTS / "phase3"
BASE = dict(width=400, height=400, resource_movement='random', p_s=0.15, p_u=0.35, p_perceive_social_info_quality=1, social_alignment='position', social_info_quality_func='sigmoid')
P5 = ["I", "S", "k", "x0", "r_max"]


def run_one(job):
    het, n, speed, P, (sigma, pers), seed = job
    np.random.seed(seed)
    m = SearcherResourceModel(N=n + 1, resource_speed=speed, noize_degree=float(sigma), persistence=float(pers), rotation_tracking_intercept=[float(v) for v in P[:, 0]],
                              detector_sensitivity_slope=[float(v * (1 + speed)) for v in P[:, 1]], social_info_quality_sensitivity_slope=[float(v) for v in P[:, 2]],
                              social_info_quality_sensitivity_intercept=[float(v) for v in P[:, 3]], search_max_rotation=[float(v) for v in P[:, 4]], **BASE)
    t0 = time.perf_counter()
    for _ in range(900): m.step()
    df = m.datacollector.get_agent_vars_dataframe().reset_index(); last = df[(df.Step == df.Step.max()) & (df.AgentID != 0)]
    sc = last.score.to_numpy(float); ton = last.n_times_on_res.to_numpy(float)
    return dict(het=het, n=n, speed=speed, seed=seed, score=float(sc.mean()), tracking_time=float(ton.mean() / 900), pcr=float(np.nanmean(sc / np.where(ton > 0, ton, np.nan))), sec=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=50); ap.add_argument("--procs", type=int, default=20); ap.add_argument("--sigma", default="fixed10"); a = ap.parse_args()
    par = pd.read_csv(OUT / "policy_params_all.csv")
    X = par[P5].copy(); L = X.copy(); L[["k", "r_max"]] = np.log(L[["k", "r_max"]]); mu = L.mean(); sig = (10.0 if a.sigma == "fixed10" else float(par.sigma.mean()), float(par.persistence.mean()))
    rng = np.random.default_rng(0)
    def sample(n, spread):
        idx = rng.integers(0, len(L), n); Z = L.iloc[idx].copy()
        Z = mu + spread * (Z - mu); Z[["k", "r_max"]] = np.exp(Z[["k", "r_max"]]); Z["r_max"] = Z.r_max.clip(5, 180); Z["k"] = Z.k.clip(0.05, 50)
        return Z[P5].to_numpy(float)
    hom = mu.copy(); hom[["k", "r_max"]] = np.exp(hom[["k", "r_max"]]); hom = hom[P5].to_numpy(float)
    jobs = []
    for speed in (0.2, 0.4, 0.6, 0.8, 1.0):
        for sd in range(a.seeds):
            jobs.append(("homogeneous", 1, speed, hom[None, :], sig, 1000 + sd)); jobs.append(("sampled", 1, speed, sample(1, 1.0), sig, 1000 + sd))
        for n in (3, 5, 10, 25):
            for sd in range(a.seeds):
                jobs.append(("homogeneous", n, speed, np.tile(hom, (n, 1)), sig, 2000 + sd)); jobs.append(("sampled", n, speed, sample(n, 1.0), sig, 2000 + sd)); jobs.append(("spread_x2", n, speed, sample(n, 2.0), sig, 2000 + sd))
    print(f"{len(jobs)} runs", flush=True); t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
            rows.append(r)
            if i % 500 == 0: print(f"  {i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "extrapolation_runs.csv", index=False)
    agg = df.groupby(["het", "n", "speed"]).agg(score=("score", "mean"), score_se=("score", lambda x: x.std() / np.sqrt(len(x))), tracking_time=("tracking_time", "mean"), pcr=("pcr", "mean"), sec=("sec", "mean")).reset_index()
    solo = agg[agg.n == 1].set_index(["het", "speed"])
    agg["d_tracking_vs_solo"] = [r.tracking_time - solo.loc[("homogeneous" if r.het == "homogeneous" else "sampled", r.speed), "tracking_time"] for r in agg.itertuples()]
    agg["d_pcr_vs_solo"] = [r.pcr - solo.loc[("homogeneous" if r.het == "homogeneous" else "sampled", r.speed), "pcr"] for r in agg.itertuples()]
    agg.to_csv(OUT / "extrapolation.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, met in zip(axes, ["score", "tracking_time", "pcr"]):
        for het, ls in (("homogeneous", "-"), ("sampled", "--"), ("spread_x2", ":")):
            for n, c in zip((3, 5, 10, 25), ("tab:blue", "tab:green", "tab:orange", "tab:red")):
                d = agg[(agg.het == het) & (agg.n == n)]; ax.plot(d.speed, d[met], ls=ls, color=c, marker="o", ms=3, label=f"{het} n={n}" if met == "score" else None)
        ax.set_xlabel("resource speed (fraction of agent speed)"); ax.set_ylabel(met)
    axes[0].legend(fontsize=6, ncol=3); fig.suptitle("Extrapolation (prediction only): speed x group size x heterogeneity of fitted individual policies"); fig.tight_layout(); fig.savefig(OUT / "fig_extrapolation.png", dpi=150); fig.savefig(OUT / "fig_extrapolation.pdf")
    pd.set_option("display.width", 200); print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
