"""Individual-level analyses on Wu et al. (2025): reliability, solo -> group transfer, role decomposition.

1. Reliability (split-half A/B = rounds 1-2 vs 3-4 of each 4-round block, Spearman-Brown; ICC(1) over rounds)
   of movement/foraging statistics (solo and group rounds), social statistics (group rounds) and choice weights.
2. Solo z: PCA (4) of standardised solo movement+foraging statistics, fitted within training folds
   (8 folds x 4 sessions; whole sessions held out). Transfer: solo z -> group-round social behaviour,
   group-round reward and group-round choice weights (own MAP fit and the authors' Stan posterior means),
   ridge with inner CV; r of out-of-fold predictions; session-cluster bootstrap for r and for Delta r
   against the solo-reward baseline. Noise ceiling = split-half reliability of each target.
3. Decomposition (participant x group round): outcome ~ env + own solo reward + own z + others' mean z + group SD z,
   mixed model (session random intercept + participant variance component), LRT per block.
"""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
P = Path('/home/higashi/Research/20260924_collective'); D = P / 'data/processed/wu2025'; R = P / 'results/wu2025'
rng = np.random.default_rng(0)
feats = pd.read_parquet(D / 'round_features.parquet')
cw = pd.read_csv(R / 'choice_weights.csv')
MOV = ['path_len', 'speed_mean', 'frac_moving', 'n_bouts', 'turn_move_mean', 'yaw_rate_mean', 'yaw_rate_sd', 'pitch_mean', 'n_cells', 'rgyr', 'straightness']
FOR = ['n_blocks', 'n_rewards', 'reward_rate', 'ibd_mean', 'ibd_median', 'ibd_cv', 'ibi_mean', 'ars_dist', 'ars_time', 't_first_block']
SOC = ['out_vis', 'n_vis_mean', 'in_vis', 'in_vis_n', 'dist_others_mean', 'nn_dist_mean', 'dist_centroid', 'frac_near10', 'n_splash_seen', 'frac_splash_seen']
GW = ['Locality', 'BlockVisibility', 'RewardPrediction', 'SuccessfulProximity', 'UnsuccessfulProximity', 'AdaptLocality', 'AdaptSuccessful', 'AdaptUnsuccessful']
SW = ['Locality', 'BlockVisibility', 'RewardPrediction', 'AdaptLocality']

# ---- folds: 8 x 4 sessions ------------------------------------------------
sessions = np.arange(1, 33); perm = rng.permutation(sessions)
folds = {int(s): int(i % 8) for i, s in enumerate(perm)}
json.dump(folds, open(D / 'split_v1.json', 'w'))

def zin_env(df, cols, by=('type', 'env')):
    out = df.copy()
    for c in cols:
        out[c] = out.groupby(list(by))[c].transform(lambda v: (v - v.mean()) / v.std())
    return out

def agg(df, rtype, cols, sel=None):
    d = df[df.type == rtype]
    if sel is not None: d = d[sel(d)]
    d = zin_env(d, cols)
    return d.groupby('pid')[cols].mean()

fz = feats.copy()
# ---- 1. reliability ---------------------------------------------------------
rel = []
def icc1(d, col):
    g = d.groupby('pid')[col]; k = g.size().mean(); msb = g.mean().var() * k; msw = g.var().mean()
    return (msb - msw) / (msb + (k - 1) * msw)
for rtype, cols in [('solo', MOV + FOR), ('group', MOV + FOR + SOC)]:
    A = agg(fz, rtype, cols, lambda d: d.pos_in_block < 2); B = agg(fz, rtype, cols, lambda d: d.pos_in_block >= 2)
    dz = zin_env(fz[fz.type == rtype], cols)
    for c in cols:
        r = A[c].corr(B[c]); rel.append(dict(domain=f'stats_{rtype}', param=c, r_split=r, spearman_brown=2 * r / (1 + r), icc1=icc1(dz, c), n=len(A)))
for rtype, names in [('solo', SW), ('group', GW)]:
    w = cw[cw.type == rtype].pivot_table(index=['pid', 'param'], columns='split', values='value').reset_index()
    for c in names:
        ww = w[w.param == c]; r = ww['A'].corr(ww['B'])
        rel.append(dict(domain=f'choice_{rtype}', param=c, r_split=r, spearman_brown=2 * r / (1 + r), icc1=np.nan, n=len(ww)))
# cross-context consistency of the same statistic: solo mean vs group mean
S_all = agg(fz, 'solo', MOV + FOR); G_all = agg(fz, 'group', MOV + FOR)
for c in MOV + FOR:
    rel.append(dict(domain='solo_vs_group', param=c, r_split=S_all[c].corr(G_all[c]), spearman_brown=np.nan, icc1=np.nan, n=len(S_all)))
rel = pd.DataFrame(rel); rel.to_csv(R / 'ind_reliability.csv', index=False)
print(rel.round(3).to_string())

# ---- 2. transfer --------------------------------------------------------------
pids = S_all.index.to_numpy(); sess = np.array([int(p.split('session')[1].split('.')[0]) for p in pids]); fold = np.array([folds[s] for s in sess])
X_stats = S_all[MOV + FOR].fillna(S_all[MOV + FOR].mean()).to_numpy()
w_solo = cw[(cw.type == 'solo') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value').reindex(pids)[SW].to_numpy()
w_group = cw[(cw.type == 'group') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value').reindex(pids)[GW]
w_group_A = cw[(cw.type == 'group') & (cw.split == 'A')].pivot(index='pid', columns='param', values='value').reindex(pids)[GW]
w_group_B = cw[(cw.type == 'group') & (cw.split == 'B')].pivot(index='pid', columns='param', values='value').reindex(pids)[GW]
stan_g = pd.read_csv(R / 'stan_vID_conditional_ARS.csv').set_index('id').reindex(pids)[GW]
G_soc = agg(fz, 'group', SOC + ['n_rewards']).reindex(pids)
G_soc_A = agg(fz, 'group', SOC + ['n_rewards'], lambda d: d.pos_in_block < 2).reindex(pids)
G_soc_B = agg(fz, 'group', SOC + ['n_rewards'], lambda d: d.pos_in_block >= 2).reindex(pids)
solo_reward = S_all[['n_rewards']].to_numpy()

# solo z: PCA within folds (out-of-fold scores) + an all-data version for descriptive use
Z_oof = np.zeros((len(pids), 4)); 
for f in range(8):
    tr, te = fold != f, fold == f
    pipe = make_pipeline(StandardScaler(), PCA(4, random_state=0)).fit(X_stats[tr])
    Z_oof[te] = pipe.transform(X_stats[te])
pipe_all = make_pipeline(StandardScaler(), PCA(4, random_state=0)).fit(X_stats)
Z_all = pipe_all.transform(X_stats)
load = pd.DataFrame(pipe_all[-1].components_.T, index=MOV + FOR, columns=[f'z{i}' for i in range(4)])
load['expl_var'] = np.nan; load.to_csv(R / 'z_solo_pca_loadings.csv')
print('PCA explained variance', pipe_all[-1].explained_variance_ratio_.round(3))
pd.DataFrame(Z_all, index=pids, columns=[f'z{i}' for i in range(4)]).to_csv(R / 'z_solo_pca4.csv')

targets = {}
for c in SOC + ['n_rewards']: targets[f'group_{c}'] = (G_soc[c].to_numpy(), G_soc_A[c].corr(G_soc_B[c]))
for c in GW: targets[f'choice_{c}'] = (w_group[c].to_numpy(), w_group_A[c].corr(w_group_B[c]))
for c in GW: targets[f'stan_{c}'] = (stan_g[c].to_numpy(), np.nan)
sources = {'solo_reward': solo_reward, 'solo_z4': Z_oof, 'solo_z4+reward': np.hstack([Z_oof, solo_reward]), 'solo_stats_ridge': X_stats,
           'solo_choice4': w_solo, 'solo_z4+choice4': np.hstack([Z_oof, w_solo])}

def oof_predict(X, y):
    pred = np.full(len(y), np.nan)
    for f in range(8):
        tr, te = fold != f, fold == f
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred

def rboot(y, p1, p0=None, B=2000):
    us = np.unique(sess); rs = []
    for _ in range(B):
        take = rng.choice(us, len(us), replace=True); idx = np.concatenate([np.where(sess == s)[0] for s in take])
        r1 = np.corrcoef(y[idx], p1[idx])[0, 1]
        rs.append(r1 if p0 is None else r1 - np.corrcoef(y[idx], p0[idx])[0, 1])
    rs = np.array(rs); return np.percentile(rs, [2.5, 97.5]), (rs <= 0).mean()

rows = []; preds = {}
for tname, (y, ceil) in targets.items():
    ok = ~np.isnan(y)
    for sname, X in sources.items():
        p = oof_predict(X[ok], y[ok]); preds[(tname, sname)] = (p, ok)
        r = np.corrcoef(y[ok], p)[0, 1]; ci, _ = rboot(y[ok], p) if False else ((np.nan, np.nan), np.nan)
        rows.append(dict(target=tname, source=sname, r=r, noise_ceiling=ceil, n=int(ok.sum())))
tr_df = pd.DataFrame(rows).pivot(index='target', columns='source', values='r').join(pd.Series({k: v[1] for k, v in targets.items()}, name='noise_ceiling'))
tr_df.to_csv(R / 'transfer_r.csv'); print(tr_df.round(3).to_string())
# Delta r vs solo-reward baseline with session-cluster bootstrap
drows = []
for tname, (y, ceil) in targets.items():
    ok = ~np.isnan(y); p0 = preds[(tname, 'solo_reward')][0]
    for sname in ['solo_z4', 'solo_z4+reward', 'solo_stats_ridge', 'solo_choice4', 'solo_z4+choice4']:
        p1 = preds[(tname, sname)][0]
        sess_bak = sess.copy()
        globals()['sess'] = sess_bak[ok]
        ci, pb = rboot(y[ok], p1, p0); ci_r, _ = rboot(y[ok], p1)
        globals()['sess'] = sess_bak
        drows.append(dict(target=tname, source=sname, r=np.corrcoef(y[ok], p1)[0, 1], r_ci_lo=ci_r[0], r_ci_hi=ci_r[1],
                          delta_r=np.corrcoef(y[ok], p1)[0, 1] - np.corrcoef(y[ok], p0)[0, 1], ci_lo=ci[0], ci_hi=ci[1], p_boot=pb, noise_ceiling=ceil))
d_df = pd.DataFrame(drows); d_df.to_csv(R / 'transfer_delta.csv', index=False)
print(d_df[d_df.source.isin(['solo_z4', 'solo_z4+choice4'])].round(3).to_string())

# ---- 3. decomposition ----------------------------------------------------------
zdf = pd.DataFrame(Z_all, index=pids, columns=[f'z{i}' for i in range(4)]); zdf['solo_reward'] = S_all['n_rewards']
g = fz[fz.type == 'group'].copy()
g = zin_env(g, SOC + ['n_rewards'])
g = g.join(zdf, on='pid')
for c in [f'z{i}' for i in range(4)] + ['solo_reward']:
    g[f'oth_{c}'] = g.groupby(['session', 'round'])[c].transform(lambda v: (v.sum() - v) / (len(v) - 1))
    g[f'sd_{c}'] = g.groupby(['session', 'round'])[c].transform('std')
dec = []
own = ' + '.join([f'z{i}' for i in range(4)]); oth = ' + '.join([f'oth_z{i}' for i in range(4)]); sd = ' + '.join([f'sd_z{i}' for i in range(4)])
for outc in ['in_vis', 'out_vis', 'n_rewards', 'n_splash_seen', 'nn_dist_mean', 'frac_near10']:
    forms = [('env+ctrl', f'{outc} ~ C(env) + solo_reward'), ('+own z', f'{outc} ~ C(env) + solo_reward + {own}'),
             ('+others mean z', f'{outc} ~ C(env) + solo_reward + {own} + {oth}'), ('+group SD z', f'{outc} ~ C(env) + solo_reward + {own} + {oth} + {sd}')]
    prev = None
    for name, fm in forms:
        try:
            m = smf.mixedlm(fm, g, groups=g['session'], re_formula='1', vc_formula={'pid': '0 + C(pid)'}).fit(reml=False)
            llf = m.llf; k = len(m.fe_params)
            fe = m.fe_params; yhat = m.model.exog @ fe.to_numpy(); r2m = yhat.var() / g[outc].var()
        except Exception as e:
            m = smf.ols(fm, g).fit(); llf = m.llf; k = len(m.params); r2m = m.rsquared
        from scipy.stats import chi2
        p = np.nan if prev is None else chi2.sf(2 * (llf - prev[0]), k - prev[1])
        dec.append(dict(outcome=outc, model=name, r2_marginal=r2m, llf=llf, p_lrt=p, n=len(g)))
        prev = (llf, k)
dec = pd.DataFrame(dec); dec.to_csv(R / 'decomposition.csv', index=False); print(dec.round(3).to_string())
