"""Convert logged agent trajectories (x, y, heading in radians, resource x,y; 1 Hz) into the paper-style 1 Hz feature
schema used by the Phase 1 encoder (search-phase features). Generic angle convention: heading vector (cos θ, sin θ)."""
import numpy as np, pandas as pd
FOV_HALF_DEG = 91.49284 / 2

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def features_from_trajectories(X, Y, TH, RX, RY, ids, group_id, cond="NP", speed="slow"):
    """X, Y, TH: (T, N) agent positions/headings; RX, RY: (T,) resource. Returns DataFrame in features_1hz schema (subset)."""
    T, N = X.shape
    rows = []
    dist = np.sqrt((X - RX[:, None]) ** 2 + (Y - RY[:, None]) ** 2)
    dth = wrap(np.diff(TH, axis=0, prepend=TH[:1]))                        # net rotation per step (rad)
    rot = np.clip(np.degrees(dth) / 150.0, -1, 1)                            # normalised as paper (150 deg/s)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        if others:
            dx = X[:, others] - X[:, [i]]; dy = Y[:, others] - Y[:, [i]]
            ang = wrap(np.arctan2(dy, dx) - TH[:, [i]])                       # signed angle to other, relative to heading
            vis = np.abs(np.degrees(ang)) < FOV_HALF_DEG
            n_vis = vis.sum(1)
            with np.errstate(invalid="ignore"):
                mean_ang = np.where(n_vis > 0, np.nansum(np.where(vis, ang, np.nan), 1) / np.maximum(n_vis, 1), np.nan)
                # mean movement direction of visible others (vector mean), relative to own heading
                cx = np.nansum(np.where(vis, np.cos(TH[:, others]), np.nan), 1); cy = np.nansum(np.where(vis, np.sin(TH[:, others]), np.nan), 1)
                mov = wrap(np.arctan2(cy, cx) - TH[:, i]); mov = np.where(n_vis > 0, mov, np.nan)
                mean_dist_o = np.where(n_vis > 0, np.nansum(np.where(vis, dist[:, others], np.nan), 1) / np.maximum(n_vis, 1), np.nan)
                best_dist_o = np.where(n_vis > 0, np.nanmin(np.where(vis, dist[:, others], np.inf), 1), np.nan)
        else:
            n_vis = np.zeros(T, int); mean_ang = np.full(T, np.nan); mov = np.full(T, np.nan); mean_dist_o = np.full(T, np.nan); best_dist_o = np.full(T, np.nan)
        with_soc = n_vis > 0
        pos = np.clip(np.abs(np.degrees(mean_ang)), 0, FOV_HALF_DEG) / FOV_HALF_DEG; movn = np.clip(np.abs(np.degrees(mov)), 0, 180) / 180
        tracking = dist[:, i] <= 40
        # state per 1 s step: tracking if on resource at both this and previous step (approximation of min/max<=40)
        prev = np.roll(tracking, 1); prev[0] = tracking[0]
        state = np.where(tracking & prev, "tracking", np.where(~tracking & ~prev, "searching", "mixed"))
        q = mean_dist_o - dist[:, i]
        qt = np.select([~with_soc, q < -10, q < 0, q < 10], [0.0, 1.0, 0.5, -0.5], default=-1.0)
        df = pd.DataFrame(dict(participant=ids[i], group=group_id, cond=cond, speed=speed, t=np.arange(T), state=state, no_move=0,
                               rot_signed=rot[:, i], rot_abs=np.abs(rot[:, i]) + 0.01, with_soc=with_soc.astype(int), n_vis=n_vis, in_deg=0,
                               detector=0.0, dist_raw=dist[:, i], dist_track=(dist[:, i] - 20) / 20, dist_search=(dist[:, i] - 200) / 200,
                               soc_pos=np.where(with_soc, (np.nan_to_num(pos) - 0.5) * 2, 0.0), soc_mov=np.where(with_soc, (np.nan_to_num(movn) - 0.5) * 2, 0.0),
                               soc_qual_track=qt, soc_qual_search=np.where(with_soc, np.where(np.nan_to_num(best_dist_o, nan=1e9) < 40, 1.0, -1.0), 0.0)))
        df["soc_qual"] = np.where(df.state == "tracking", df.soc_qual_track, df.soc_qual_search)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
