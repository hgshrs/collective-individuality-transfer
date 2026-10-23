"""Parse Hawkins et al. (2023) Experiment 1 and 3 raw tick-by-tick logs (8 Hz) into parquet.

Exp 1: binary hidden spotlight, click-to-set-goal, 5 min (2400 ticks); file name date_nplanned_field_pid.csv.
Exp 3: gradient score field (spotlight + noise), keyboard turning, 6 min (2880 ticks); file name date_nplanned_field-noise_gameid.csv;
       waiting-room logs (no score field) are kept as a pre-game solo-like phase.
Completeness as in the authors' code: game complete if max tick >= 2399 (exp1) / 2879 (exp3); participant dropped in first half if max tick < 1200.
Outputs: data/processed/hawkins2023/{ticks,games,waiting}.parquet
"""
import glob, re
import numpy as np, pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[1]; H = P / 'external/hawkins2023/data'; OUT = P / 'data/processed/hawkins2023'
rows = []; games = []
def load(files, exp):
    for f in files:
        name = Path(f).stem; parts = name.split('_')
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if len(d) == 0: continue
        d = d.rename(columns={'x_pos': 'x', 'y_pos': 'y', 'velocity': 'v', 'bg_val': 'bg'})
        if exp == 1:
            n_planned, field, noise = int(parts[1]), int(parts[2]), 'none'
        else:
            n_planned = int(parts[1]); field, noise = parts[2].split('-'); field = int(field); noise = 'high' if noise == '2en01' else 'low'
        d['experiment'] = exp; d['gameid'] = name; d['n_planned'] = n_planned; d['field'] = field; d['noise'] = noise
        d['date'] = parts[0]
        for c in ['obs_bg_val', 'goal_x', 'goal_y']:
            if c not in d: d[c] = np.nan
        rows.append(d[['experiment', 'gameid', 'pid', 'tick', 'x', 'y', 'v', 'angle', 'bg', 'total_points', 'obs_bg_val', 'goal_x', 'goal_y', 'n_planned', 'field', 'noise', 'active']])
load(sorted(glob.glob(str(H / 'experiment1/raw/*/games/*.csv'))), 1)
load(sorted(glob.glob(str(H / 'experiment3/*/games/*.csv'))), 3)
t = pd.concat(rows, ignore_index=True)
t['tick'] = t['tick'].astype(int)
g = t.groupby(['experiment', 'gameid']).agg(max_tick=('tick', 'max'), n_pids=('pid', 'nunique'), n_planned=('n_planned', 'first'), field=('field', 'first'), noise=('noise', 'first')).reset_index()
g['complete'] = ((g.experiment == 1) & (g.max_tick >= 2399)) | ((g.experiment == 3) & (g.max_tick >= 2879))
pm = t.groupby(['experiment', 'gameid', 'pid'])['tick'].max().reset_index().rename(columns={'tick': 'pid_max_tick'})
pm['dropped_first_half'] = pm.pid_max_tick < 1200
n2 = pm[~pm.dropped_first_half].groupby(['experiment', 'gameid']).size().rename('n_players').reset_index()
g = g.merge(n2, how='left').fillna({'n_players': 0}); g['n_players'] = g.n_players.astype(int)
t = t.merge(pm[['experiment', 'gameid', 'pid', 'dropped_first_half']], on=['experiment', 'gameid', 'pid'])
t.to_parquet(OUT / 'ticks.parquet', index=False); g.to_parquet(OUT / 'games.parquet', index=False)
print(t.shape); print(g.groupby(['experiment', 'complete']).size()); print(g[g.complete].groupby(['experiment', 'n_players']).size())
# waiting rooms (exp 3)
w = []
for f in sorted(glob.glob(str(H / 'experiment3/*/waiting_games/*.csv'))):
    try: d = pd.read_csv(f)
    except pd.errors.EmptyDataError: continue
    if len(d) == 0: continue
    d = d.rename(columns={'x_pos': 'x', 'y_pos': 'y', 'velocity': 'v', 'bg_val': 'bg'}); d['waitid'] = Path(f).stem; w.append(d)
w = pd.concat(w, ignore_index=True); w.to_parquet(OUT / 'waiting.parquet', index=False)
print('waiting', w.shape, w.pid.nunique(), 'pids; ticks per pid median', w.groupby('pid').size().median())
