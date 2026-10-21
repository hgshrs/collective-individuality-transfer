"""Evaluate the mobility-augmented grid ABM (scripts/83) against observed Wu et al. group x env outcomes, side by side with
the original ABM (scripts/66-67). r within env (z within env, 64 session x env cells), session-cluster bootstrap CI;
Delta r for composition (main_mob - gmean_mob) and for the mobility term itself (main_mob - main, pop_mob - pop).
Also: calibration of absolute reward, and p_move vs solo / group reward. Outputs: results/wu2025/sim_mobility_eval.csv,
sim_mobility_delta.csv, sim_mobility_calibration.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path('/home/higashi/Research/20260924_collective'); R = P / 'results/wu2025'
rng = np.random.default_rng(5)
gr = pd.read_csv(R / 'group_rounds.csv'); sim0 = pd.read_parquet(R / 'sim_runs.parquet'); sim1 = pd.read_parquet(R / 'sim_runs_mobility.parquet')
obs = gr.groupby(['session', 'env'])[['total_reward', 'nn_dist_mean', 'reward_sd']].mean()
def nominal(sim, name):
    nom = sim[sim.variant == name].groupby(['session', 'env', 'env_file', 'seed'])[['total_reward']].sum().reset_index(); nom['variant'] = name; return nom
simv = pd.concat([sim0[sim0.variant != 'nominal'], nominal(sim0, 'nominal'), sim1[sim1.variant != 'nominal_mob'], nominal(sim1, 'nominal_mob')])
agg = simv.groupby(['session', 'env', 'variant'])[['total_reward', 'nn_dist', 'reward_sd']].mean()
sessions = obs.index.get_level_values('session').to_numpy(); envs = obs.index.get_level_values('env').to_numpy(); us = np.unique(sessions)
def zenv(s): return s.groupby(level='env').transform(lambda v: (v - v.mean()) / v.std()).to_numpy()
def get(variant, col): return zenv(agg.xs(variant, level='variant')[col].reindex(obs.index))
def r(a, b, idx=None):
    if idx is not None: a, b = a[idx], b[idx]
    return float(np.corrcoef(a, b)[0, 1])
def boot(fn, B=3000):
    vals = []
    for _ in range(B):
        take = rng.choice(us, len(us), replace=True); idx = np.concatenate([np.where(sessions == s)[0] for s in take]); vals.append(fn(idx))
    vals = np.array(vals); return np.percentile(vals, [2.5, 97.5]), 2 * min((vals <= 0).mean(), (vals >= 0).mean())
VARS = ['main', 'gmean', 'pop', 'nominal', 'main_mob', 'gmean_mob', 'pop_mob', 'nominal_mob']
rows, drows = [], []
for oc, sc in [('total_reward', 'total_reward'), ('nn_dist_mean', 'nn_dist'), ('reward_sd', 'reward_sd')]:
    y = zenv(obs[oc])
    for v in VARS:
        if v.startswith('nominal') and sc != 'total_reward': continue
        p = get(v, sc); ci, _ = boot(lambda i: r(y, p, i))
        rows.append(dict(observed=oc, simulated=sc, variant=v, r=r(y, p), ci_lo=ci[0], ci_hi=ci[1], r_random=r(y[envs == 'random'], p[envs == 'random']), r_smooth=r(y[envs == 'smooth'], p[envs == 'smooth']), n=len(y)))
    for a, b in [('main_mob', 'gmean_mob'), ('main_mob', 'pop_mob'), ('main_mob', 'main'), ('pop_mob', 'pop'), ('gmean_mob', 'gmean'), ('main_mob', 'nominal_mob'), ('nominal_mob', 'nominal')]:
        if 'nominal' in a + b and sc != 'total_reward': continue
        pa, pb = get(a, sc), get(b, sc); ci, pv = boot(lambda i: r(y, pa, i) - r(y, pb, i))
        drows.append(dict(observed=oc, simulated=sc, contrast=f'{a} - {b}', delta_r=r(y, pa) - r(y, pb), ci_lo=ci[0], ci_hi=ci[1], p_boot=pv))
res = pd.DataFrame(rows); res.to_csv(R / 'sim_mobility_eval.csv', index=False); dres = pd.DataFrame(drows); dres.to_csv(R / 'sim_mobility_delta.csv', index=False)
pd.set_option('display.width', 250); print(res.round(3).to_string()); print(dres.round(3).to_string())
cal = agg[['total_reward']].unstack('variant').droplevel(0, axis=1)[VARS].join(obs[['total_reward']].rename(columns={'total_reward': 'observed'})).groupby(level='env').mean()
cal.to_csv(R / 'sim_mobility_calibration.csv'); print(cal.round(1).to_string())
pm = pd.read_csv(R / 'sim_mobility_params.csv', index_col=0); feats = pd.read_parquet(P / 'data/processed/wu2025/round_features.parquet')
solo = feats[feats.type == 'solo'].groupby('pid')['n_rewards'].mean(); grp = feats[feats.type == 'group'].groupby('pid')['n_rewards'].mean()
print('p_move vs solo reward r =', round(pm.p_move.corr(solo), 3), '; vs group reward r =', round(pm.p_move.corr(grp), 3))
