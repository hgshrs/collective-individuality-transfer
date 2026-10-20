"""Phase 3: observed group-level metrics (AGENT_SPEC §6.1 'collective metrics'), one row per analysed group.
score = mean member gross score (40 - d points/s inside 40 m, from the 1 Hz table; comparable with the ABM score),
tracking_time = mean member fraction of seconds on the resource, pcr = points per tracking second (mean over members),
in_out_corr = correlation across members of mean in- and out-degree, in_deg/out_deg means, median distance from the
resource centre (median over members of their median distance), disc_gain = mean discoverer in-degree gain (10 s after -
10 s before; events from phase2_discoveries.csv), n_disc.
Output: results/phase3/observed_group_metrics.csv
"""
import numpy as np, pandas as pd
from collective_id.paths import PROCESSED, RESULTS

OUT = RESULTS / "phase3"; OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_parquet(PROCESSED / "features_1hz.parquet"); f = f[f.cond != "A"].copy()
f["pts"] = np.where(f.dist_raw <= 40, 40 - f.dist_raw, 0.0); f["trk"] = (f.state == "tracking").astype(float)
ind = f.groupby(["participant", "group", "cond", "speed"]).agg(score=("pts", "sum"), track=("trk", "mean"), n_track=("trk", "sum"), in_deg=("in_deg", "mean"), out_deg=("n_vis", "mean"), med_dist=("dist_raw", "median"), n_sec=("t", "size")).reset_index()
ind["pcr"] = ind.score / ind.n_track.replace(0, np.nan)
ev = pd.read_csv(PROCESSED / "phase2_discoveries.csv")
indeg = f.pivot(index="participant", columns="t", values="in_deg"); tmin, tmax = indeg.columns.min(), indeg.columns.max()
gains = []
for r in ev.itertuples():
    td = int(round(r.t_disc)); pre = [t for t in range(td - 10, td) if tmin <= t <= tmax]; post = [t for t in range(td + 1, td + 11) if tmin <= t <= tmax]
    if len(pre) >= 5 and len(post) >= 5 and r.participant in indeg.index:
        gains.append(dict(group=r.group, gain=indeg.loc[r.participant, post].mean() - indeg.loc[r.participant, pre].mean()))
gains = pd.DataFrame(gains).groupby("group").gain.agg(disc_gain="mean", n_disc="size").reset_index()
G = ind.groupby(["group", "cond", "speed"]).agg(n=("participant", "size"), score=("score", "mean"), tracking_time=("track", "mean"), pcr=("pcr", "mean"), in_deg=("in_deg", "mean"), out_deg=("out_deg", "mean"), med_dist=("med_dist", "median"),
                                                in_out_corr=("in_deg", lambda s: np.nan)).reset_index()
cor = ind.groupby("group").apply(lambda g: g.in_deg.corr(g.out_deg), include_groups=False).rename("in_out_corr").reset_index()
G = G.drop(columns="in_out_corr").merge(cor, on="group").merge(gains, on="group", how="left"); G["n_disc"] = G.n_disc.fillna(0).astype(int)
G["cell"] = G.cond + "_" + G.speed
G.to_csv(OUT / "observed_group_metrics.csv", index=False)
pd.set_option("display.width", 200); print(G.groupby("cell")[["score", "tracking_time", "pcr", "in_out_corr", "med_dist", "disc_gain"]].agg(["mean", "std"]).round(2)); print(G.n_disc.describe())
