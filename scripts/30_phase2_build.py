"""Phase 2 data build (AGENT_SPEC §5): per-participant tables for Plan B (roles) and Plan C (voluntary sharing).

Sources: data/processed/features_1hz.parquet (paper's 1 Hz table), data/processed/traces_10hz_subset.parquet (10 Hz
positions, in/out-degree, signaling and discovery-event flags; cached from processed_traces.csv).

Outputs (data/processed/):
  phase2_first3min_stats.csv  — own-movement summary statistics of the first 3 min (t in [20, 180) s; the 1 Hz table
                                starts at 20 s). No visible-other features (§5.1). Includes score_3min (control).
  phase2_first3min_seq.parquet — 1 s own-movement sequences of the first 3 min for the CNN variant.
  phase2_targets.csv          — Plan B role targets over the remaining 12 min (t >= 180): in-degree mean, out-degree
                                mean, #discoveries as discoverer, discoverer in-degree gain (10 s after − 10 s before);
                                Plan C sharing targets (VP): sharing fraction (all / t>=180 / tracking / searching),
                                P(sharing in the 10 s after own discovery), P(sharing within 10 s of resource entry).
  phase2_discoveries.csv      — one row per discovery event (discoverer, time).
"""
import numpy as np, pandas as pd
from collective_id.paths import PROCESSED

T0, T3 = 20, 180
f = pd.read_parquet(PROCESSED / "features_1hz.parquet")
parts = f.participant.unique()
meta = f.groupby("participant").agg(group=("group", "first"), cond=("cond", "first"), speed=("speed", "first"), final_score=("final_score", "first"), time_on_resource=("time_on_resource", "first")).reset_index()
f["pts"] = np.where(f.dist_raw <= 40, 40 - f.dist_raw, 0.0); f["trk"] = (f.state == "tracking").astype(float)

# ---------------- 10 Hz: per-second speed and discovery events ----------------
tr = pd.read_parquet(PROCESSED / "traces_10hz_subset.parquet"); tr = tr[tr.participant.isin(parts)].sort_values(["participant", "time"])
tr["sec"] = np.floor(tr.time + 1e-6).astype(int)
dx = tr.groupby("participant").x.diff(); dz = tr.groupby("participant").z.diff(); dt = tr.groupby("participant").time.diff()
tr["disp"] = np.sqrt(dx ** 2 + dz ** 2); tr.loc[(dt <= 0) | (dt > 0.15), "disp"] = np.nan
sp = tr.groupby(["participant", "sec"]).agg(disp=("disp", "sum"), nvalid=("disp", lambda x: x.notna().sum())).reset_index()
sp["speed"] = np.where(sp.nvalid > 0, sp.disp / (sp.nvalid * 0.1), np.nan)
mv = tr.assign(moving=(tr.disp > 0.2).astype(float).where(tr.disp.notna())).groupby(["participant", "sec"]).moving.mean().reset_index()
sp = sp.merge(mv, on=["participant", "sec"]).rename(columns={"sec": "t", "speed": "spd", "moving": "mov"})   # "speed" is the resource-speed condition in f
f = f.merge(sp[["participant", "t", "spd", "mov"]], on=["participant", "t"], how="left")
f.to_parquet(PROCESSED / "features_1hz_speed.parquet", index=False)

ev = tr[(tr.resource_finding_event_1 != 0) & (tr.resource_finding_event_1_order == 1)].groupby(["participant", "resource_finding_event_1"]).time.min().reset_index()
ev["t_disc"] = ev.time + 2.0          # the flag window starts 2 s before the discovery (paper's R code removes a 2 s offset)
ev = ev.merge(meta[["participant", "group", "cond", "speed"]], on="participant").rename(columns={"resource_finding_event_1": "event_idx"})
ev.to_csv(PROCESSED / "phase2_discoveries.csv", index=False)
print("discovery events (analysed participants):", len(ev), ev.groupby(["speed", "cond"]).size().to_dict())

# ---------------- first-3-min own-movement summary statistics ----------------
def first3(g):
    g = g.sort_values("t"); rs = g.rot_signed.to_numpy(); nz = rs[np.abs(rs) > 0.005]
    flips = np.mean(np.sign(nz[1:]) != np.sign(nz[:-1])) if len(nz) > 2 else 0.0
    sw = (g.trk.diff().abs() > 0).sum()
    return pd.Series({
        "rot_abs_mean": g.rot_abs.mean(), "rot_abs_sd": g.rot_abs.std(), "rot_abs_p90": g.rot_abs.quantile(0.9),
        "rot_zero_frac": (g.rot_abs <= 0.011).mean(), "rot_large_frac": (g.rot_abs > 0.5).mean(),
        "rot_autocorr": g.rot_signed.autocorr(1) if len(g) > 3 else 0.0, "rot_flip_rate": flips,
        "no_move_frac": (g.no_move > 0).mean(), "speed_mean": g.spd.mean(), "speed_sd": g.spd.std(), "moving_frac": g.mov.mean(),
        "dist_mean": g.dist_raw.mean(), "dist_sd": g.dist_raw.std(), "dist_min": g.dist_raw.min(), "dist_max": g.dist_raw.max(),
        "track_frac": g.trk.mean(), "n_switch": sw, "score_3min": g.pts.sum(), "n_sec": len(g),
    })
F3 = f[(f.t >= T0) & (f.t < T3)].groupby("participant").apply(first3, include_groups=False)
F3 = F3.fillna(F3.median(numeric_only=True)); F3.to_csv(PROCESSED / "phase2_first3min_stats.csv")

seq = f[(f.t >= T0) & (f.t < T3)][["participant", "t", "rot_abs", "rot_signed", "no_move", "spd", "mov", "dist_raw", "trk"]].copy()
seq["no_move"] = (seq.no_move > 0).astype(float); seq["spd"] = (seq.spd.fillna(seq.spd.median()) / 7.0); seq["mov"] = seq.mov.fillna(seq.mov.median())
seq["dist_n"] = (seq.dist_raw - 200) / 200
seq.to_parquet(PROCESSED / "phase2_first3min_seq.parquet", index=False)

# ---------------- Plan B targets (t >= 180) ----------------
rest = f[f.t >= T3]
TB = rest.groupby("participant").agg(in_deg=("in_deg", "mean"), out_deg=("n_vis", "mean"), score_rest=("pts", "sum"), track_rest=("trk", "mean")).reset_index()
TB["in_minus_out"] = TB.in_deg - TB.out_deg
indeg = f.pivot(index="participant", columns="t", values="in_deg")
sig = f.pivot(index="participant", columns="t", values="is_signaling")
tmin, tmax = indeg.columns.min(), indeg.columns.max()
rows = []
for p, g in ev.groupby("participant"):
    gains, gains_rest, share_post = [], [], []
    for td in g.t_disc:
        td = int(round(td)); pre = [t for t in range(td - 10, td) if tmin <= t <= tmax]; post = [t for t in range(td + 1, td + 11) if tmin <= t <= tmax]
        if len(pre) >= 5 and len(post) >= 5:
            gain = indeg.loc[p, post].mean() - indeg.loc[p, pre].mean(); gains.append(gain)
            if td >= T3: gains_rest.append(gain)
            if p in sig.index: share_post.append(sig.loc[p, [t for t in range(td, td + 11) if tmin <= t <= tmax]].mean())
    rows.append(dict(participant=p, n_disc_all=len(g), n_disc_rest=int((g.t_disc >= T3).sum()), disc_gain_all=np.mean(gains) if gains else np.nan,
                     disc_gain_rest=np.mean(gains_rest) if gains_rest else np.nan, share_post_disc=np.mean(share_post) if share_post else np.nan))
D = pd.DataFrame(rows)
TB = TB.merge(D, on="participant", how="left"); TB[["n_disc_all", "n_disc_rest"]] = TB[["n_disc_all", "n_disc_rest"]].fillna(0).astype(int)

# ---------------- Plan C targets (VP) ----------------
vp = f[f.cond == "VP"].copy()
TC = vp.groupby("participant").agg(share_all=("is_signaling", "mean")).reset_index()
TC = TC.merge(vp[vp.t >= T3].groupby("participant").is_signaling.mean().rename("share_rest").reset_index(), how="left")
TC = TC.merge(vp[vp.state == "tracking"].groupby("participant").is_signaling.mean().rename("share_track").reset_index(), how="left")
TC = TC.merge(vp[vp.state == "searching"].groupby("participant").is_signaling.mean().rename("share_search").reset_index(), how="left")
# resource entries (searching -> tracking) and sharing within 10 s after entry
ent = []
for p, g in vp.sort_values("t").groupby("participant"):
    trk = g.trk.to_numpy(); t = g.t.to_numpy(); s = g.is_signaling.to_numpy()
    idx = np.where((trk[1:] == 1) & (trk[:-1] == 0))[0] + 1
    post = [s[i:i + 11].max() for i in idx if i + 1 < len(s)]
    ent.append(dict(participant=p, n_entries=len(idx), share_post_entry=np.mean(post) if post else np.nan))
TC = TC.merge(pd.DataFrame(ent), on="participant", how="left")

T = meta.merge(TB, on="participant", how="left").merge(TC, on="participant", how="left")
T.to_csv(PROCESSED / "phase2_targets.csv", index=False)
pd.set_option("display.width", 200)
print(F3.describe().T[["mean", "std", "min", "max"]].round(3))
print(T[T.cond != "A"][["in_deg", "out_deg", "n_disc_rest", "disc_gain_rest", "disc_gain_all"]].describe().round(3))
print("participants with >=1 own discovery (t>=180):", int((T[T.cond != "A"].n_disc_rest > 0).sum()), "of", int((T.cond != "A").sum()))
print(T[T.cond == "VP"][["share_all", "share_rest", "share_track", "share_search", "share_post_disc", "share_post_entry", "n_entries"]].describe().round(3))
