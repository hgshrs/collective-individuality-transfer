"""Group-level analyses on Wu et al. (2025): predicting group-round outcomes from members' solo measurements.

Unit: group round (32 sessions x 8 group rounds; 4 random + 4 smooth). Outcomes standardised within env:
  total_reward (sum of members' rewards), out_vis_mean (visibility-network density), nn_dist_mean, splash_seen_mean.
Member aggregates from solo rounds: mean and SD of solo z (PCA4, all-data), solo reward, solo choice weights.
1. Zero-shot nominal prediction: members' mean solo reward (same env) vs observed group reward.
2. Leave-one-session-out ridge: means only vs means + SDs; Delta r with session bootstrap; permutation of the SD block.
3. Collective gain (group reward - nominal expectation) vs SD of z.
"""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats
warnings.filterwarnings('ignore')
P = Path('/home/higashi/Research/20260924_collective'); D = P / 'data/processed/wu2025'; R = P / 'results/wu2025'
rng = np.random.default_rng(1)
feats = pd.read_parquet(D / 'round_features.parquet'); cw = pd.read_csv(R / 'choice_weights.csv')
Z = pd.read_csv(R / 'z_solo_pca4.csv', index_col=0)
SW = ['Locality', 'BlockVisibility', 'RewardPrediction', 'AdaptLocality']
# member-level solo measurements (per env and pooled)
solo = feats[feats.type == 'solo']
solo_rew_env = solo.groupby(['pid', 'env'])['n_rewards'].mean().unstack()
solo_rew = solo.groupby('pid')['n_rewards'].mean()
wsolo = cw[(cw.type == 'solo') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value')[SW]
member = Z.join(solo_rew.rename('solo_reward')).join(wsolo)
# group-round outcomes
g = feats[feats.type == 'group']
gr = g.groupby(['session', 'round', 'env']).agg(total_reward=('n_rewards', 'sum'), out_vis_mean=('out_vis', 'mean'), in_vis_sd=('in_vis', 'std'),
                                                  nn_dist_mean=('nn_dist_mean', 'mean'), splash_seen_mean=('n_splash_seen', 'mean'),
                                                  reward_sd=('n_rewards', 'std')).reset_index()
members = g.groupby(['session', 'round'])['pid'].apply(list)
def agg_members(pids, cols):
    m = member.loc[pids, cols]; return pd.concat([m.mean().add_prefix('mean_'), m.std().add_prefix('sd_')])
cols = list(Z.columns) + ['solo_reward'] + SW
M = pd.DataFrame([agg_members(members[(s, r)], cols) for s, r in zip(gr.session, gr['round'])])
gr = pd.concat([gr, M], axis=1)
# nominal expectation: sum of members' mean solo reward in the same env
gr['nominal_reward'] = [solo_rew_env.loc[members[(s, r)], e].sum() for s, r, e in zip(gr.session, gr['round'], gr.env)]
OUT = ['total_reward', 'out_vis_mean', 'nn_dist_mean', 'splash_seen_mean', 'reward_sd', 'in_vis_sd']
for c in OUT + ['nominal_reward']:
    gr[c + '_z'] = gr.groupby('env')[c].transform(lambda v: (v - v.mean()) / v.std())
gr['gain'] = gr['total_reward'] - gr['nominal_reward']
gr['gain_z'] = gr.groupby('env')['gain'].transform(lambda v: (v - v.mean()) / v.std())
gr.to_csv(R / 'group_rounds.csv', index=False)
sess = gr['session'].to_numpy(); us = np.unique(sess)

def boot(fn, B=2000):
    vals = []
    for _ in range(B):
        take = rng.choice(us, len(us), replace=True); idx = np.concatenate([np.where(sess == s)[0] for s in take]); vals.append(fn(idx))
    vals = np.array(vals); return np.percentile(vals, [2.5, 97.5]), (vals <= 0).mean()

rows = []
# 1. zero-shot nominal
y = gr['total_reward_z'].to_numpy(); p = gr['nominal_reward_z'].to_numpy()
r = np.corrcoef(y, p)[0, 1]; ci, _ = boot(lambda i: np.corrcoef(y[i], p[i])[0, 1])
rows.append(dict(outcome='total_reward', model='nominal_solo_reward (zero-shot)', r=r, ci_lo=ci[0], ci_hi=ci[1], delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan))
# group x env aggregation (4 rounds averaged)
ge = gr.groupby(['session', 'env'])[['total_reward_z', 'nominal_reward_z']].mean()
r_ge = ge['total_reward_z'].corr(ge['nominal_reward_z'])
rows.append(dict(outcome='total_reward (group x env mean)', model='nominal_solo_reward (zero-shot)', r=r_ge, ci_lo=np.nan, ci_hi=np.nan, delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan))

def loso(Xcols, y):
    X = gr[Xcols].to_numpy(); pred = np.full(len(y), np.nan)
    for s in us:
        tr, te = sess != s, sess == s
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred
zc = list(Z.columns)
sets = {'mean_solo_reward': ['mean_solo_reward'], 'mean_z4': [f'mean_{c}' for c in zc], 'mean_z4+reward': [f'mean_{c}' for c in zc] + ['mean_solo_reward'],
        'mean_z4+reward+SD': [f'mean_{c}' for c in zc] + ['mean_solo_reward'] + [f'sd_{c}' for c in zc] + ['sd_solo_reward'],
        'mean_choice4': [f'mean_{c}' for c in SW], 'mean_choice4+SD': [f'mean_{c}' for c in SW] + [f'sd_{c}' for c in SW],
        'SD_only_z4+reward': [f'sd_{c}' for c in zc] + ['sd_solo_reward']}
comp = []
for outc in OUT + ['gain']:
    y = gr[outc + '_z'].to_numpy(); preds = {k: loso(v, y) for k, v in sets.items()}
    for k, p in preds.items():
        r = np.corrcoef(y, p)[0, 1]; ci, _ = boot(lambda i: np.corrcoef(y[i], p[i])[0, 1])
        rows.append(dict(outcome=outc, model=k, r=r, ci_lo=ci[0], ci_hi=ci[1], delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan))
    for base, full in [('mean_z4+reward', 'mean_z4+reward+SD'), ('mean_choice4', 'mean_choice4+SD'), ('mean_solo_reward', 'mean_z4+reward')]:
        p0, p1 = preds[base], preds[full]
        d = np.corrcoef(y, p1)[0, 1] - np.corrcoef(y, p0)[0, 1]
        ci, pb = boot(lambda i: np.corrcoef(y[i], p1[i])[0, 1] - np.corrcoef(y[i], p0[i])[0, 1])
        rows.append(dict(outcome=outc, model=f'{full} - {base}', r=np.nan, ci_lo=np.nan, ci_hi=np.nan, delta_r=d, d_ci_lo=ci[0], d_ci_hi=ci[1], p=pb))
    # composition regression: in-sample R2 of mean block vs + SD block, permutation of SD rows within env
    Xm = gr[sets['mean_z4+reward']].to_numpy(); Xs = gr[[f'sd_{c}' for c in zc] + ['sd_solo_reward']].to_numpy()
    def r2(X): return LinearRegression().fit(X, y).score(X, y)
    r2m = r2(Xm); r2f = r2(np.hstack([Xm, Xs])); dR2 = r2f - r2m
    null = []
    env = gr['env'].to_numpy()
    for _ in range(2000):
        Xp = Xs.copy()
        for e in np.unique(env):
            idx = np.where(env == e)[0]; Xp[idx] = Xs[rng.permutation(idx)]
        null.append(r2(np.hstack([Xm, Xp])) - r2m)
    comp.append(dict(outcome=outc, r2_mean=r2m, r2_mean_plus_sd=r2f, delta_r2=dR2, p_perm=(np.array(null) >= dR2).mean(), n=len(y)))
    # collective gain vs SD z: univariate Spearman per SD column
    if outc == 'gain':
        for c in [f'sd_{c}' for c in zc] + ['sd_solo_reward', 'mean_solo_reward'] + [f'mean_{c}' for c in zc]:
            rho, pv = stats.spearmanr(gr[c], y); comp.append(dict(outcome=f'gain~{c}', r2_mean=np.nan, r2_mean_plus_sd=np.nan, delta_r2=rho, p_perm=pv, n=len(y)))
res = pd.DataFrame(rows); res.to_csv(R / 'group_pred.csv', index=False); print(res.round(3).to_string())
comp = pd.DataFrame(comp); comp.to_csv(R / 'group_composition.csv', index=False); print(comp.round(3).to_string())
