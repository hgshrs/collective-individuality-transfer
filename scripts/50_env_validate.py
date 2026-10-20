"""§6.3-1: validate the vectorised environment (src/collective_id/env.py) against the paper's mesa ABM on the Fig 5B grid
(homogeneous agents, k in {0.1, 1, 10}, speed in {0.2..1.0}, solo and 5 agents; paper parameters I 50, S 35, r_max 90,
x0 0, sigma 10, persistence 0). Compares tracking time and pcr with results/phase0/abm_fig5B_summary.csv (mesa, 100 runs).
Usage: python scripts/50_env_validate.py [--runs 200]
Output: results/phase3/env_validation.csv, fig_env_validation.png
"""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RESULTS
from collective_id.env import run

ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=200); a = ap.parse_args()
mesa = pd.read_csv(RESULTS / "phase0" / "abm_fig5B_summary.csv")
rows = []
for N in (1, 5):
    for sp in (0.2, 0.4, 0.6, 0.8, 1.0):
        for k in (0.1, 1.0, 10.0):
            pol = lambda env, o, t, k=k: env.abm_policy(o, I=50.0, S=35.0, k=k, x0=0.0, rmax=90.0, sigma=10.0, pers=0.0)
            res = run(a.runs, N, sp, pol, seed=int(1000 * sp + 10 * k + N), record=False)
            sc, ton = res["score"], res["n_on"]
            rows.append(dict(N=N + 1, resource_speed=sp, k=k, pcr=float(np.nanmean(sc / np.where(ton > 0, ton, np.nan))), tracking_time=float((ton / 900).mean()), score=float(sc.mean()),
                             tt_se=float((ton / 900).mean(1).std() / np.sqrt(a.runs)), pcr_se=float(np.nanmean(sc / np.where(ton > 0, ton, np.nan), 1).std() / np.sqrt(a.runs))))
            print(rows[-1], flush=True)
V = pd.DataFrame(rows).merge(mesa[["N", "resource_speed", "k", "pcr", "tracking_time", "score"]], on=["N", "resource_speed", "k"], suffixes=("_np", "_mesa"))
V["d_tt"] = V.tracking_time_np - V.tracking_time_mesa; V["d_pcr"] = V.pcr_np - V.pcr_mesa
V.to_csv(RESULTS / "phase3" / "env_validation.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, m in zip(axes, ["tracking_time", "pcr"]):
    ax.scatter(V[f"{m}_mesa"], V[f"{m}_np"], c=np.where(V.N == 2, "tab:gray", "tab:blue"), s=25)
    lim = [V[[f"{m}_mesa", f"{m}_np"]].min().min(), V[[f"{m}_mesa", f"{m}_np"]].max().max()]; ax.plot(lim, lim, "k--", lw=.7); ax.set_xlabel(f"{m} (mesa ABM)"); ax.set_ylabel(f"{m} (numpy env)")
    ax.set_title(f"r = {np.corrcoef(V[f'{m}_mesa'], V[f'{m}_np'])[0,1]:.3f}")
fig.suptitle("Environment validation: Fig 5B grid, homogeneous ABM policy"); fig.tight_layout(); fig.savefig(RESULTS / "phase3" / "fig_env_validation.png", dpi=150)
pd.set_option("display.width", 200); print(V.round(3).to_string())
print("mean |Δ tracking time|:", V.d_tt.abs().mean().round(4), " mean |Δ pcr|:", V.d_pcr.abs().mean().round(3), " r_tt:", np.corrcoef(V.tracking_time_mesa, V.tracking_time_np)[0, 1].round(3), " r_pcr:", np.corrcoef(V.pcr_mesa, V.pcr_np)[0, 1].round(3))
# the qualitative Fig 5B check: at 40 % only k = 10 beats solo on both metrics
g = V[V.N == 6].set_index(["resource_speed", "k"]); s = V[V.N == 2].set_index(["resource_speed", "k"])
for k in (0.1, 1.0, 10.0):
    print(f"speed 0.4 k={k}: Δtt={g.loc[(0.4, k), 'tracking_time_np'] - s.loc[(0.4, k), 'tracking_time_np']:.3f} Δpcr={g.loc[(0.4, k), 'pcr_np'] - s.loc[(0.4, k), 'pcr_np']:.3f}")
