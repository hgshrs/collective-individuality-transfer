"""Parse the Wu et al. (2025) Minecraft foraging raw logs into parquet tables.

Outputs (data/processed/wu2025/):
  rounds.parquet   one row per session x round (0-15): type (solo/group), env (random/smooth), env_file, block (order of 4-round blocks)
  players.parquet  20 Hz player states with server-side visibility flags and Unity visual-field occupancy of each other player
  blocks.parquet   block destruction events (from the authors' feather cache) + env_file
  evis.parquet     Unity-transcribed event visibility (reward splashes seen by each player)
Raw round file index ge_k corresponds to round k-2 (ge_0, ge_1 are training rounds).
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, polars as pl

P = Path(__file__).resolve().parents[1]
RAW = P / 'external/wu2025/analysis/data/2021batch'
SESS = P / 'external/wu2025/experiment/run/sessions'
OUT = P / 'data/processed/wu2025'; OUT.mkdir(parents=True, exist_ok=True)

# ---- rounds -------------------------------------------------------------
rows = []
for s in range(1, 33):
    rd = json.load(open(SESS / f'session{s}.json'))['roundData']
    k = 0
    for i, r in enumerate(rd):
        if r['trainingRound'] == 'TRUE':
            continue
        rows.append(dict(session=s, round=k, ge=i, type='solo' if r['soloRound'] == 'TRUE' else 'group',
                         env=r['environment'], env_file=r['envFile'], block=k // 4, pos_in_block=k % 4))
        k += 1
    assert k == 16, (s, k)
rounds = pd.DataFrame(rows)
rounds.to_parquet(OUT / 'rounds.parquet', index=False)
print('rounds', rounds.shape)

# ---- blocks ---------------------------------------------------------------
b = pd.read_feather(RAW / 'all_blocks_pd_cache.feather')
b['session'] = b['session'].str.extract(r'session(\d+)\.json').astype(int)
b = b.merge(rounds[['session', 'round', 'env_file', 'ge']], on=['session', 'round'], how='left', validate='m:1')
chk = b.groupby(['session', 'round'])[['type', 'env']].first().join(rounds.set_index(['session', 'round'])[['type', 'env']], rsuffix='_json')
assert (chk['type'] == chk['type_json']).all() and (chk['env'] == chk['env_json']).all()
b['pid'] = b['name'] + 'session' + b['session'].astype(str) + '.json'
b.to_parquet(OUT / 'blocks.parquet', index=False)
print('blocks', b.shape)

# ---- players + pvis -------------------------------------------------------
def parse_vis(s):
    s = s.strip()[1:-1]
    return [x for x in s.split(',') if x]

parts = []
evis_parts = []
for r in rounds.to_dict('records'):
    s, k, ge = int(r['session']), int(r['round']), int(r['ge'])
    f = RAW / f'session{s}_ge_{ge}_players.log.csv'
    df = pd.read_csv(f, sep=';', usecols=range(8), dtype={'Visible Players': str})
    df.columns = ['t', 'name', 'x', 'z', 'xlook', 'ylook', 'zlook', 'vis']
    vis = df['vis'].map(parse_vis)
    for j in range(1, 5):
        df[f'srv_vis_P{j}'] = vis.map(lambda v, j=j: f'MPIB{j}' in v).astype(np.int8)
    df = df.drop(columns='vis')
    df['t'] = df['t'].round(2)
    pf = RAW / f'session{s}_ge_{ge}_pvis.log.csv'
    if pf.exists():
        pv = pd.read_csv(pf, sep=';')
        pv.columns = ['t', 'name', 'occ_P1', 'occ_P2', 'occ_P3', 'occ_P4']
        pv['t'] = pv['t'].astype(float).round(2)
        df = df.merge(pv, on=['t', 'name'], how='left', validate='1:1')
    else:
        for j in range(1, 5):
            df[f'occ_P{j}'] = np.nan
    df['session'] = s; df['round'] = k; df['type'] = r['type']; df['env'] = r['env']
    parts.append(df)
    ef = RAW / f'session{s}_ge_{ge}_evis.log.csv'
    if ef.exists():
        ev = pd.read_csv(ef, sep=';')
        ev.columns = ['t', 'name', 'eventtype', 'eventid', 'x', 'y', 'z', 'occupancy', 'trigger']
        ev['session'] = s; ev['round'] = k; ev['type'] = r['type']; ev['env'] = r['env']
        evis_parts.append(ev)
    if k == 15:
        print('session', s, 'done', flush=True)

players = pd.concat(parts, ignore_index=True)
players['pid'] = players['name'] + 'session' + players['session'].astype(str) + '.json'
players = players.sort_values(['session', 'round', 'name', 't']).reset_index(drop=True)
players.to_parquet(OUT / 'players.parquet', index=False)
print('players', players.shape, 'pvis missing frac', players['occ_P1'].isna().mean())
evis = pd.concat(evis_parts, ignore_index=True)
evis.to_parquet(OUT / 'evis.parquet', index=False)
print('evis', evis.shape, evis['eventtype'].value_counts().to_dict())
