"""Evaluate the zero-shot group simulations (scripts/66) against observed group x env outcomes (Wu et al. 2025).

Observed (group_rounds.csv, mean of the 4 group rounds per session x env): total_reward, nn_dist_mean, out_vis_mean.
Simulated (sim_runs.parquet, mean over 100 runs per session x env x variant): total_reward, nn_dist, pair_dist.
r within env (z within env, pooled over 64 session x env cells); session-cluster bootstrap CI; Delta r vs controls;
membership-shuffle null for r. Output: results/wu2025/sim_eval.csv, sim_eval_delta.csv, sim_shuffle.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[1]; R = P / 'results/wu2025'
rng = np.random.default_rng(2)
gr = pd.read_csv(R / 'group_rounds.csv'); sim = pd.read_parquet(R / 'sim_runs.parquet')
obs = gr.groupby(['session', 'env'])[['total_reward', 'nn_dist_mean', 'out_vis_mean', 'reward_sd']].mean()
def zenv(df, cols):
    out = df.copy()
    for c in cols: out[c] = out.groupby(level='env')[c].transform(lambda v: (v - v.mean()) / v.std())
    return out
obs_z = zenv(obs, obs.columns)
# nominal: sum of the four solitary runs sharing (session, env, env_file, seed)
nom = sim[sim.variant == 'nominal'].groupby(['session', 'env', 'env_file', 'seed'])['total_reward'].sum().reset_index()
nom['variant'] = 'nominal'
simv = pd.concat([sim[sim.variant != 'nominal'][['session', 'env', 'variant', 'env_file', 'seed', 'tag', 'total_reward', 'nn_dist', 'pair_dist', 'reward_sd']], nom])
agg = simv.groupby(['session', 'env', 'variant'])[['total_reward', 'nn_dist', 'pair_dist', 'reward_sd']].mean()
sessions = obs.index.get_level_values('session').to_numpy(); envs = obs.index.get_level_values('env').to_numpy()
def get(variant, col):
    a = agg.xs(variant, level='variant')[col].reindex(obs.index)
    return a.groupby(level='env').transform(lambda v: (v - v.mean()) / v.std()).to_numpy()
def r(a, b, idx=None):
    if idx is not None: a, b = a[idx], b[idx]
    return np.corrcoef(a, b)[0, 1]
us = np.unique(sessions)
def boot(fn, B=3000):
    vals = []
    for _ in range(B):
        take = rng.choice(us, len(us), replace=True); idx = np.concatenate([np.where(sessions == s)[0] for s in take]); vals.append(fn(idx))
    vals = np.array(vals); return np.percentile(vals, [2.5, 97.5]), (vals <= 0).mean()
pairs = [('total_reward', 'total_reward'), ('nn_dist_mean', 'nn_dist'), ('nn_dist_mean', 'pair_dist'), ('reward_sd', 'reward_sd')]
rows = []; drows = []
for oc, sc in pairs:
    y = obs_z[oc].to_numpy()
    for variant in ['main', 'gmean', 'pop', 'nominal', 'shuffle']:
        if variant == 'nominal' and sc != 'total_reward': continue
        p = get(variant, sc)
        if np.isnan(p).all(): continue
        ci, _ = boot(lambda i: r(y, p, i))
        for e in ['random', 'smooth']:
            m = envs == e
        rows.append(dict(observed=oc, simulated=sc, variant=variant, r=r(y, p), ci_lo=ci[0], ci_hi=ci[1],
                         r_random=r(y[envs == 'random'], p[envs == 'random']), r_smooth=r(y[envs == 'smooth'], p[envs == 'smooth']), n=len(y)))
    pm = get('main', sc)
    for ctrl in ['gmean', 'pop', 'nominal', 'shuffle']:
        if ctrl == 'nominal' and sc != 'total_reward': continue
        pc = get(ctrl, sc)
        if np.isnan(pc).all(): continue
        d = r(y, pm) - r(y, pc); ci, pb = boot(lambda i: r(y, pm, i) - r(y, pc, i))
        drows.append(dict(observed=oc, simulated=sc, control=ctrl, delta_r=d, ci_lo=ci[0], ci_hi=ci[1], p_boot=pb))
res = pd.DataFrame(rows); res.to_csv(R / 'sim_eval.csv', index=False); print(res.round(3).to_string())
dres = pd.DataFrame(drows); dres.to_csv(R / 'sim_eval_delta.csv', index=False); print(dres.round(3).to_string())
# shuffle null: one random shuffled composition per session x env per draw
sh = sim[sim.variant == 'shuffle'].groupby(['session', 'env', 'tag'])[['total_reward', 'nn_dist']].mean().reset_index()
srows = []
for oc, sc in [('total_reward', 'total_reward'), ('nn_dist_mean', 'nn_dist')]:
    y = obs_z[oc].to_numpy(); r_main = r(y, get('main', sc)); null = []
    groups = {k: g[sc].to_numpy() for k, g in sh.groupby(['session', 'env'])}
    for _ in range(2000):
        p = np.array([rng.choice(groups[(s, e)]) for s, e in zip(sessions, envs)])
        p = pd.Series(p, index=obs.index).groupby(level='env').transform(lambda v: (v - v.mean()) / v.std()).to_numpy()
        null.append(r(y, p))
    null = np.array(null)
    srows.append(dict(observed=oc, r_main=r_main, r_shuffle_mean=null.mean(), r_shuffle_q95=np.percentile(null, 95), p_perm=(null >= r_main).mean()))
sres = pd.DataFrame(srows); sres.to_csv(R / 'sim_shuffle.csv', index=False); print(sres.round(3).to_string())
# absolute calibration
cal = agg.xs('main', level='variant')[['total_reward']].join(obs[['total_reward']], lsuffix='_sim', rsuffix='_obs').groupby(level='env').mean()
cal.to_csv(R / 'sim_calibration.csv'); print(cal.round(2))
