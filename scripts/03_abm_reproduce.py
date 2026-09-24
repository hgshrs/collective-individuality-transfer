"""Phase 0 §3.1-6: run the paper's mesa ABM (external/analysis, unmodified) for selectivity k in {0.1,1,10},
resource speed in {0.2..1.0}, N in {2 (solo), 6 (5 agents)}; reproduce Fig 5B qualitatively and time one run.
Usage: python scripts/03_abm_reproduce.py [--iters 100] [--procs 4]"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import ROOT, RESULTS, EXTERNAL
sys.path.insert(0, str(EXTERNAL))  # paper code: `from agent_based_model.abm import SearcherResourceModel`
from agent_based_model.abm import SearcherResourceModel
from concurrent.futures import ProcessPoolExecutor

PARAMS = dict(width=400, height=400, resource_movement='random', noize_degree=10, p_s=0.15, p_u=0.35,
              rotation_tracking_intercept=None, detector_sensitivity_slope=35, search_max_rotation=90,
              social_info_quality_sensitivity_intercept=0, persistence=0, p_perceive_social_info_quality=1,
              social_alignment='position', social_info_quality_func='sigmoid')

def run_one(args):
    n, speed, k, seed = args
    np.random.seed(seed)
    t0 = time.perf_counter()
    m = SearcherResourceModel(N=n, resource_speed=speed, social_info_quality_sensitivity_slope=k, **PARAMS)
    for _ in range(900):
        m.step()
    dt = time.perf_counter() - t0
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    last = df[(df.Step == df.Step.max()) & (df.AgentID != 0)]
    return [dict(N=n, resource_speed=speed, k=k, seed=seed, agent=int(a.AgentID), score=float(a.score),
                 n_times_on_res=int(a.n_times_on_res), steps=int(df.Step.max()), sec=dt) for a in last.itertuples()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=100); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    out = RESULTS / "phase0"; out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, sp, k, 1000 * i + j) for j, (n, sp, k) in enumerate([(n, sp, k) for n in (2, 6) for sp in (0.2, 0.4, 0.6, 0.8, 1.0) for k in (0.1, 1.0, 10.0)]) for i in range(a.iters)]
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
            rows += r
            if i % 200 == 0: print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "abm_fig5B_runs.csv", index=False)
    df["pcr"] = df.score / df.n_times_on_res.replace(0, np.nan); df["tracking_time"] = df.n_times_on_res / df.steps
    agg = df.groupby(["N", "resource_speed", "k"]).agg(pcr=("pcr", "mean"), tracking_time=("tracking_time", "mean"), score=("score", "mean"), n_runs=("seed", "nunique"), sec_per_run=("sec", "mean")).reset_index()
    solo = agg[agg.N == 2].set_index("resource_speed")
    grp = agg[agg.N == 6].copy()
    grp["d_pcr"] = grp.apply(lambda r: r.pcr - solo.loc[r.resource_speed].pcr.mean(), axis=1)
    grp["d_tracking_time"] = grp.apply(lambda r: r.tracking_time - solo.loc[r.resource_speed].tracking_time.mean(), axis=1)
    agg.to_csv(out / "abm_fig5B_summary.csv", index=False); grp.to_csv(out / "abm_fig5B_group_minus_solo.csv", index=False)
    timing = dict(sec_per_run_solo=float(df[df.N == 2].drop_duplicates(["seed", "resource_speed", "k"]).sec.mean()),
                  sec_per_run_group5=float(df[df.N == 6].drop_duplicates(["seed", "resource_speed", "k"]).sec.mean()),
                  procs=a.procs, iters_per_combo=a.iters, total_wall_s=time.time() - t0, mesa_version=__import__("mesa").__version__)
    json.dump(timing, open(out / "abm_timing.json", "w"), indent=1); print(timing)
    # qualitative check: at 40% only k=10 beats solo on both metrics
    chk = grp[grp.resource_speed == 0.4][["k", "d_pcr", "d_tracking_time"]]; print(chk)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mk = {0.2: "o", 0.4: "s", 0.6: "D", 0.8: "^", 1.0: "P"}; cm = {0.1: "tab:blue", 1.0: "tab:green", 10.0: "gold"}
    for _, r in grp.iterrows():
        ax.scatter(r.d_tracking_time, r.d_pcr, marker=mk[r.resource_speed], color=cm[r.k], edgecolor="k", s=60, alpha=1 if r.resource_speed in (0.4, 0.8) else 0.35)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("Δ tracking time (group − solo)"); ax.set_ylabel("Δ tracking efficiency (group − solo)")
    ax.set_title(f"Fig 5B reproduction (paper ABM, {a.iters} runs/combo)\ncolor: k=0.1 blue, 1 green, 10 yellow; marker: speed")
    fig.tight_layout(); fig.savefig(out / "abm_fig5B.png", dpi=150); fig.savefig(out / "abm_fig5B.pdf")
