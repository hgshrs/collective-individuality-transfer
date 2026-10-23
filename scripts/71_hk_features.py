"""Per participant x half x state behavioural statistics for Hawkins et al. (2023) Exp 1 and 3 (8 Hz).

States: 'search' (outside the scoring region: bg == 0 in exp 1, bg < 0.5 in exp 3) vs 'score' (inside).
Halves: ticks < 1200 ('first') vs >= 1200 ('second'), as in the authors' analysis.
Private statistics: speed-state fractions, path length, turning rate, click rate / click distance / click turn (exp 1),
  stop latency after entering the scoring region, fraction of scoring ticks spent stopped, exits from the region.
Social statistics (n_others >= 1): distance to nearest other / centroid, fraction of ticks facing another (45 deg cone),
  facing a stopped (exp 1: v = 0; exp 3: slow) other, being faced by others (in-degree), approach events
  (fast + facing the same other for >= 4 ticks), n_others.
Output: data/processed/hawkins2023/features.parquet (one row per participant x half x state, plus state='all').
"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[1]; D = P / 'data/processed/hawkins2023'
t = pd.read_parquet(D / 'ticks.parquet'); games = pd.read_parquet(D / 'games.parquet')
t = t.merge(games[['experiment', 'gameid', 'complete', 'n_players']], on=['experiment', 'gameid'])
t = t[t.complete & ~t.dropped_first_half]
FACE = np.radians(45)
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
rows = []
for (exp, gid), g in t.groupby(['experiment', 'gameid']):
    g = g.sort_values(['pid', 'tick'])
    pids = sorted(g.pid.unique()); T = int(g.tick.max()) + 1
    A = {}
    for p in pids:
        d = g[g.pid == p].set_index('tick').reindex(range(T))
        A[p] = d[['x', 'y', 'v', 'angle', 'bg', 'goal_x', 'goal_y']].to_numpy(float)
    for p in pids:
        a = A[p]; x, y, v, ang, bg = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        valid = ~np.isnan(x)
        head = np.radians((ang - 90) % 360)  # authors' convention: angle w.r.t. top
        dhead = np.abs(wrap(np.diff(head, prepend=head[0]))) * 8  # rad/s
        inspot = (bg >= 0.5) if exp == 3 else (bg >= 1)
        stop = v == 0; slow = v == 2.125; fast = v == 7.125
        step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
        # clicks
        if exp == 1:
            gx, gy = a[:, 5], a[:, 6]
            click = (np.diff(gx, prepend=gx[0]) != 0) | (np.diff(gy, prepend=gy[0]) != 0)
            cdist = np.hypot(gx - x, gy - y)
            cang = np.arctan2(gy - y, gx - x)
            head_prev = np.roll(head, 1); head_prev[0] = head[0]
            cturn = np.abs(wrap(cang - head_prev))
        # entering/exiting the region, stop latency
        enter = np.where(inspot & ~np.roll(inspot, 1))[0]; exit_ = np.where(~inspot & np.roll(inspot, 1))[0]
        lat = []
        for e in enter:
            w = np.where(stop[e:e + 40])[0]
            lat.append(w[0] / 8 if len(w) else 5.0)
        others = [q for q in pids if q != p]
        if others:
            ox = np.stack([A[q][:, 0] for q in others], 1); oy = np.stack([A[q][:, 1] for q in others], 1); ov = np.stack([A[q][:, 2] for q in others], 1)
            oh = np.radians((np.stack([A[q][:, 3] for q in others], 1) - 90) % 360)
            dx, dy = ox - x[:, None], oy - y[:, None]; dist = np.hypot(dx, dy)
            ang_to = np.arctan2(dy, dx); facing = (np.abs(wrap(ang_to - head[:, None])) < FACE) & ~np.isnan(dist)
            ostop = (ov == 0) if exp == 1 else (ov == 2.125)
            facing_stop = facing & ostop; facing_mov = facing & ~ostop & ~np.isnan(ov)
            # being faced
            ang_from = np.arctan2(-dy, -dx); faced_by = (np.abs(wrap(ang_from - oh)) < FACE) & ~np.isnan(dist)
            nn = np.nanmin(np.where(np.isnan(dist), np.inf, dist), 1); nn[np.isinf(nn)] = np.nan
            cen = np.hypot(np.nanmean(ox, 1) - x, np.nanmean(oy, 1) - y)
            # approach events: fast and facing the same other for >= 4 consecutive ticks
            appr = np.zeros(T, bool)
            for j in range(len(others)):
                f = fast & facing[:, j]
                run = 0
                for i in range(T):
                    run = run + 1 if f[i] else 0
                    if run >= 4: appr[i] = True
            n_oth = (~np.isnan(dist)).sum(1)
        for half, hm in [('first', np.arange(T) < 1200), ('second', np.arange(T) >= 1200), ('all', np.ones(T, bool))]:
            for state, sm in [('search', ~inspot), ('score', inspot), ('all', np.ones(T, bool))]:
                m = valid & hm & sm
                if m.sum() < 40: continue
                r = dict(experiment=exp, gameid=gid, pid=p, half=half, state=state, n_ticks=int(m.sum()), n_players=len(pids),
                         score=np.nanmean(bg[valid & hm]), frac_inspot=inspot[valid & hm].mean(),
                         frac_stop=stop[m].mean(), frac_slow=slow[m].mean(), frac_fast=fast[m].mean(), path_len=step[m].sum() / m.sum() * 8,
                         turn_rate=np.nanmean(dhead[m]), turn_sd=np.nanstd(dhead[m]), frac_turning=(dhead[m] > 0.1).mean(),
                         n_cells=len(np.unique(np.floor(x[m] / 40) * 100 + np.floor(y[m] / 40))))
                if exp == 1:
                    cm = m & click
                    r.update(click_rate=click[m].sum() / m.sum() * 8, click_dist=np.nanmean(cdist[cm]) if cm.sum() else np.nan,
                             click_turn=np.nanmean(cturn[cm]) if cm.sum() else np.nan, click_dist_cv=np.nanstd(cdist[cm]) / np.nanmean(cdist[cm]) if cm.sum() > 2 else np.nan)
                if state == 'all':
                    em = enter[hm[enter]] if len(enter) else enter
                    r.update(n_enter=len(em), n_exit=int(hm[exit_].sum()) if len(exit_) else 0,
                             stop_latency=np.mean([lat[i] for i in range(len(enter)) if hm[enter[i]]]) if len(em) else np.nan,
                             frac_stop_inspot=stop[valid & hm & inspot].mean() if (valid & hm & inspot).sum() > 8 else np.nan,
                             frac_stop_search=stop[valid & hm & ~inspot].mean() if (valid & hm & ~inspot).sum() > 8 else np.nan)
                if others and (m & (n_oth > 0)).sum() > 40:
                    mo = m & (n_oth > 0)
                    r.update(n_others_mean=n_oth[mo].mean(), nn_dist=np.nanmean(nn[mo]), cen_dist=np.nanmean(cen[mo]),
                             out_face=facing[mo].any(1).mean(), out_face_stop=facing_stop[mo].any(1).mean(), out_face_mov=facing_mov[mo].any(1).mean(),
                             face_select=(facing_stop[mo].any(1).mean() / max(ostop[mo].any(1).mean(), 1e-3)) - (facing_mov[mo].any(1).mean() / max((~ostop[mo] & ~np.isnan(ov[mo])).any(1).mean(), 1e-3)),
                             in_face=faced_by[mo].any(1).mean(), in_face_n=faced_by[mo].sum(1).mean(), approach=appr[mo].mean(), frac_near50=(nn[mo] < 50).mean())
                rows.append(r)
out = pd.DataFrame(rows); out.to_parquet(D / 'features.parquet', index=False)
# waiting-room (exp 3) movement statistics per participant: solo-like pre-game phase (no score field)
w = pd.read_parquet(D / 'waiting.parquet'); wrows = []
for p, d in w.groupby('pid'):
    d = d.sort_values('tick'); x, y, v, ang = d.x.to_numpy(float), d.y.to_numpy(float), d.v.to_numpy(float), d.angle.to_numpy(float)
    if len(d) < 160: continue
    head = np.radians((ang - 90) % 360); dhead = np.abs(wrap(np.diff(head))) * 8; step = np.hypot(np.diff(x), np.diff(y))
    wrows.append(dict(pid=p, w_ticks=len(d), w_frac_fast=(v == 7.125).mean(), w_frac_slow=(v == 2.125).mean(), w_path_len=step.sum() / len(d) * 8,
                      w_turn_rate=dhead.mean(), w_turn_sd=dhead.std(), w_frac_turning=(dhead > 0.1).mean(),
                      w_n_cells=len(np.unique(np.floor(x / 40) * 100 + np.floor(y / 40))) / (len(d) / 8) * 60,
                      w_wall=((x < 15) | (x > 470) | (y < 15) | (y > 270)).mean()))
wf = pd.DataFrame(wrows); wf.to_parquet(D / 'waiting_features.parquet', index=False); print('waiting features', wf.shape)
print(out.shape); print(out[(out.half == 'all') & (out.state == 'all')].groupby('experiment').size())
print(out[(out.half == 'all') & (out.state == 'all')].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
