"""Per participant x round behavioural summary statistics for the Wu et al. (2025) Minecraft data.

Movement (20 Hz positions + look vector), foraging (block destruction events), and social
(Unity visual-field occupancy, reward-splash observations, inter-player distances; group rounds only).
Output: data/processed/wu2025/round_features.parquet (one row per participant x round).
"""
import numpy as np, pandas as pd
from pathlib import Path

P = Path('/home/higashi/Research/20260924_collective'); D = P / 'data/processed/wu2025'
players = pd.read_parquet(D / 'players.parquet')
blocks = pd.read_parquet(D / 'blocks.parquet')
evis = pd.read_parquet(D / 'evis.parquet')
rounds = pd.read_parquet(D / 'rounds.parquet')
DT = 0.05

def yaw(xl, zl):
    return np.arctan2(zl, xl)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

rows = []
for (s, k), g in players.groupby(['session', 'round']):
    g = g.sort_values(['name', 't'])
    names = sorted(g['name'].unique())
    pos = {n: g[g['name'] == n][['t', 'x', 'z', 'xlook', 'ylook', 'zlook', 'occ_P1', 'occ_P2', 'occ_P3', 'occ_P4']].to_numpy() for n in names}
    T = min(len(v) for v in pos.values())
    pos = {n: v[:T] for n, v in pos.items()}
    rtype = g['type'].iloc[0]; env = g['env'].iloc[0]
    bl = blocks[(blocks.session == s) & (blocks['round'] == k)]
    ev = evis[(evis.session == s) & (evis['round'] == k)] if rtype == 'group' else None
    for n in names:
        v = pos[n]; t, x, z, xl, yl, zl = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        dx, dz = np.diff(x), np.diff(z); step = np.hypot(dx, dz)
        moving = step > 0.02
        speed = step / DT
        path_len = step.sum()
        # movement heading changes while moving
        hd = np.arctan2(dz, dx); mv_idx = np.where(moving)[0]
        turn_move = np.abs(wrap(np.diff(hd[mv_idx]))) if len(mv_idx) > 2 else np.array([np.nan])
        # look-vector yaw rate (rad/s) and pitch
        yw = yaw(xl, zl); dyaw = np.abs(wrap(np.diff(yw))) / DT
        pitch = np.arcsin(np.clip(yl, -1, 1))
        # exploration extent
        cells = np.unique(np.floor(x / 3).astype(int) * 1000 + np.floor(z / 3).astype(int))
        rg = np.sqrt(((x - x.mean()) ** 2 + (z - z.mean()) ** 2).mean())
        # straightness over 10 s windows
        W = int(10 / DT); strs = []
        for a in range(0, T - W, W):
            pl = step[a:a + W - 1].sum(); nd = np.hypot(x[a + W - 1] - x[a], z[a + W - 1] - z[a])
            if pl > 1: strs.append(nd / pl)
        # bouts of movement
        bouts = np.sum(np.diff(moving.astype(int)) == 1)
        # foraging
        b = bl[bl['name'] == n].sort_values('time')
        nb = len(b); nr = int(b['reward'].sum())
        ibd = np.hypot(np.diff(b['x'].to_numpy()), np.diff(b['z'].to_numpy())) if nb > 1 else np.array([np.nan])
        ibi = np.diff(b['time'].to_numpy()) if nb > 1 else np.array([np.nan])
        rew = b['reward'].to_numpy()
        ars = np.nan; ars_t = np.nan
        if nb > 2:
            after_r = ibd[rew[:-1]]; after_n = ibd[~rew[:-1]]
            if len(after_r) > 0 and len(after_n) > 0:
                ars = after_n.mean() - after_r.mean()
                ars_t = ibi[~rew[:-1]].mean() - ibi[rew[:-1]].mean()
        row = dict(session=s, round=k, type=rtype, env=env, name=n, pid=f'{n}session{s}.json',
                   path_len=path_len, speed_mean=speed.mean(), frac_moving=moving.mean(), n_bouts=bouts,
                   turn_move_mean=np.nanmean(turn_move), yaw_rate_mean=dyaw.mean(), yaw_rate_sd=dyaw.std(),
                   pitch_mean=pitch.mean(), n_cells=len(cells), rgyr=rg, straightness=np.mean(strs) if strs else np.nan,
                   n_blocks=nb, n_rewards=nr, reward_rate=nr / max(nb, 1), ibd_mean=np.nanmean(ibd), ibd_median=np.nanmedian(ibd),
                   ibd_cv=np.nanstd(ibd) / np.nanmean(ibd) if nb > 2 else np.nan, ibi_mean=np.nanmean(ibi),
                   ars_dist=ars, ars_time=ars_t, t_first_block=b['time'].iloc[0] if nb else np.nan)
        if rtype == 'group':
            j = int(n[-1]); others = [m for m in names if m != n]
            occ_self = v[:, 6:10]  # occupancy of P1..P4 in n's view
            occ_others = np.stack([occ_self[:, int(m[-1]) - 1] for m in others], 1)
            out_vis = (occ_others > 0).any(1).mean(); n_vis = (occ_others > 0).sum(1).mean()
            in_vis = np.stack([(pos[m][:, 5 + j] > 0) for m in others], 1)
            d_oth = np.stack([np.hypot(x - pos[m][:, 1], z - pos[m][:, 2]) for m in others], 1)
            cx = np.mean([pos[m][:, 1] for m in names], 0); cz = np.mean([pos[m][:, 2] for m in names], 0)
            e = ev[(ev['name'] == n) & (ev['trigger'] != n) & (ev['occupancy'] > 0)]
            e_all = ev[(ev['trigger'] != n)]
            n_splash_avail = e_all['eventid'].nunique()
            row.update(out_vis=out_vis, n_vis_mean=n_vis, in_vis=in_vis.any(1).mean(), in_vis_n=in_vis.sum(1).mean(),
                       dist_others_mean=d_oth.mean(), nn_dist_mean=d_oth.min(1).mean(), dist_centroid=np.hypot(x - cx, z - cz).mean(),
                       frac_near10=(d_oth.min(1) < 10).mean(), n_splash_seen=e['eventid'].nunique(),
                       frac_splash_seen=e['eventid'].nunique() / max(n_splash_avail, 1))
        rows.append(row)
    if k == 15: print('session', s, flush=True)
out = pd.DataFrame(rows).merge(rounds[['session', 'round', 'block', 'pos_in_block', 'env_file']], on=['session', 'round'])
out.to_parquet(D / 'round_features.parquet', index=False)
print(out.shape); print(out.describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
