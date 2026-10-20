"""Individual- and group-level Collective-ID analyses on Hawkins et al. (2023) Exp 1 and 3.

All variables are residualised within experiment on group size (categorical) and score field before correlation
('within-cell', as in Chirkov). Folds: 10 folds of whole games (seed 0).
1. Reliability: first vs second half of every statistic and teacher coefficient.
2. z = PCA(4) of first-half *search-state* private statistics (fitted within folds). Transfer to second-half targets
   (scoring-state behaviour, social statistics, teacher social weights, score) with ridge; baseline = first-half score;
   Delta r with game-cluster bootstrap; noise ceiling = first-vs-second r of the target.
   Exp 3 only: waiting-room (pre-game, no score field) movement statistics as a true solo source.
3. Decomposition of second-half individual outcomes: own z, others' mean z, group SD z (mixed model, game intercept).
4. Group level (games with >= 2 players; SD block for >= 3): second-half group mean score from members' first-half
   mean z / mean score (+ SD), leave-one-game-out ridge, permutation of the SD block; zero-shot nominal predictor.
Outputs: results/hawkins2023/{reliability,transfer_r,transfer_delta,decomposition,group_pred,group_composition}.csv
"""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr
warnings.filterwarnings('ignore')
P = Path('/home/higashi/Research/20260924_collective'); D = P / 'data/processed/hawkins2023'; R = P / 'results/hawkins2023'
rng = np.random.default_rng(0)
F = pd.read_parquet(D / 'features.parquet'); games = pd.read_parquet(D / 'games.parquet'); W = pd.read_parquet(D / 'waiting_features.parquet')
TW = pd.read_csv(R / 'teacher_weights.csv')
F = F.merge(games[['experiment', 'gameid', 'field', 'noise']], on=['experiment', 'gameid'])
PRIV = ['frac_stop', 'frac_slow', 'frac_fast', 'path_len', 'turn_rate', 'turn_sd', 'frac_turning', 'n_cells', 'click_rate', 'click_dist', 'click_turn', 'click_dist_cv']
SCORE_STATE = ['frac_stop_inspot', 'stop_latency', 'n_exit', 'n_enter']
SOC = ['out_face', 'out_face_stop', 'out_face_mov', 'face_select', 'in_face', 'in_face_n', 'approach', 'nn_dist', 'cen_dist', 'frac_near50']
TC = ['b_cen', 'b_stop', 'b_nn', 'b_cen_spot', 'b_stop_spot', 'b0']
def cell_resid(df, cols):
    """residualise cols within experiment on n_players (categorical) x field."""
    out = df.copy()
    key = out['experiment'].astype(str) + '_' + out['n_players'].astype(str) + '_' + out['field'].astype(str)
    for c in cols:
        if c not in out: continue
        out[c] = out[c] - out.groupby(key)[c].transform('mean')
        sd = out.groupby('experiment')[c].transform('std'); out[c] = out[c] / sd
    return out
def part(half, state):
    return F[(F.half == half) & (F.state == state)].set_index('pid')
A1, A2 = part('first', 'all'), part('second', 'all'); S1 = part('first', 'search'); C2 = part('second', 'score')
tw = {h: TW[TW.half == h].pivot(index='pid', columns='coef', values='value') for h in ['first', 'second', 'all']}
# ---- 1. reliability ---------------------------------------------------------
rel = []
for name, (d1, d2, cols) in {'all_state': (A1, A2, PRIV + SCORE_STATE + SOC + ['score']), 'search_state': (S1, part('second', 'search'), PRIV),
                             'score_state': (part('first', 'score'), C2, PRIV)}.items():
    for exp in [1, 3]:
        x1 = cell_resid(d1[d1.experiment == exp].reset_index(), cols).set_index('pid'); x2 = cell_resid(d2[d2.experiment == exp].reset_index(), cols).set_index('pid')
        j = x1.index.intersection(x2.index)
        for c in cols:
            if c not in x1 or x1.loc[j, c].notna().sum() < 30: continue
            r = x1.loc[j, c].corr(x2.loc[j, c]); rel.append(dict(block=name, experiment=exp, param=c, r_split=r, spearman_brown=2 * r / (1 + r), n=int(x1.loc[j, c].notna().sum())))
for exp in [1, 3]:
    a, b = tw['first'], tw['second']; j = a.index.intersection(b.index); j = [p for p in j if p in A1.index and A1.loc[p, 'experiment'] == exp]
    for c in TC:
        r = a.loc[j, c].corr(b.loc[j, c]); rel.append(dict(block='teacher', experiment=exp, param=c, r_split=r, spearman_brown=2 * r / (1 + r), n=len(j)))
rel = pd.DataFrame(rel); rel.to_csv(R / 'reliability.csv', index=False); print(rel.round(3).to_string())
# ---- 2. transfer -------------------------------------------------------------
base = A1[['experiment', 'gameid', 'n_players', 'field']].copy()
src = cell_resid(S1.reset_index(), PRIV).set_index('pid')[PRIV].reindex(base.index)
src = src.fillna(src.mean())
score1 = cell_resid(A1.reset_index(), ['score']).set_index('pid')['score'].reindex(base.index)
T2 = cell_resid(A2.reset_index(), SCORE_STATE + SOC + ['score']).set_index('pid').reindex(base.index)
tw2 = tw['second'].reindex(base.index); tw1 = tw['first'].reindex(base.index)
# folds by game
gids = base.gameid.unique(); fold_of = {g: i % 10 for i, g in enumerate(rng.permutation(gids))}; fold = base.gameid.map(fold_of).to_numpy()
json.dump(fold_of, open(D / 'split_v1.json', 'w'))
exp = base.experiment.to_numpy(); game = base.gameid.to_numpy()
Xs = src.to_numpy(); Z = np.zeros((len(base), 4))
for e in [1, 3]:
    m = exp == e; cols = PRIV if e == 1 else [c for c in PRIV if not c.startswith('click')]; Xe = src[cols].to_numpy()
    for f in range(10):
        tr, te = m & (fold != f), m & (fold == f)
        pipe = make_pipeline(StandardScaler(), PCA(4, random_state=0)).fit(Xe[tr]); Z[te] = pipe.transform(Xe[te])
    pipe = make_pipeline(StandardScaler(), PCA(4, random_state=0)).fit(Xe[m])
    pd.DataFrame(pipe[-1].components_.T, index=cols, columns=[f'z{i}' for i in range(4)]).round(2).to_csv(R / f'z_loadings_exp{e}.csv')
    print('exp', e, 'PCA var', pipe[-1].explained_variance_ratio_.round(2))
zdf = pd.DataFrame(Z, index=base.index, columns=[f'z{i}' for i in range(4)]); zdf['score1'] = score1; zdf.join(base).to_csv(R / 'z_first_half.csv')
# waiting (exp 3)
wcols = [c for c in W.columns if c.startswith('w_') and c != 'w_ticks']
Wz = W.set_index('pid')[wcols].reindex(base.index)
targets = {}
for c in SCORE_STATE + SOC + ['score']: targets[f'beh_{c}'] = T2[c].to_numpy()
for c in TC: targets[f'teacher_{c}'] = tw2[c].to_numpy()
def oof(X, y, m):
    pred = np.full(len(y), np.nan)
    for f in range(10):
        tr, te = m & (fold != f), m & (fold == f)
        if te.sum() == 0 or tr.sum() < 20: continue
        mod = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = mod.predict(X[te])
    return pred
def cboot(y, p1, p0, g, B=2000):
    ug = np.unique(g); out = []
    for _ in range(B):
        take = rng.choice(ug, len(ug), replace=True); idx = np.concatenate([np.where(g == s)[0] for s in take])
        r1 = np.corrcoef(y[idx], p1[idx])[0, 1]; out.append(r1 if p0 is None else r1 - np.corrcoef(y[idx], p0[idx])[0, 1])
    out = np.array(out); return np.percentile(out, [2.5, 97.5]), (out <= 0).mean()
sources = {'score1': score1.to_numpy()[:, None], 'z4': Z, 'z4+score1': np.hstack([Z, score1.to_numpy()[:, None]]), 'search_stats_ridge': Xs,
           'teacher1': tw1[TC].to_numpy(), 'z4+teacher1': np.hstack([Z, tw1[TC].to_numpy()]), 'waiting': Wz.to_numpy()}
rrows = []; drows = []
for e in [1, 3]:
    me = exp == e
    for tname, y in targets.items():
        ok = me & ~np.isnan(y)
        preds = {}
        for sname, X in sources.items():
            if sname == 'waiting' and e == 1: continue
            okx = ok & ~np.isnan(X).any(1)
            if okx.sum() < 60: continue
            p = oof(X[okx], y[okx], np.ones(okx.sum(), bool)) if False else None
            # fit within experiment with folds restricted
            pred = np.full(len(y), np.nan); 
            for f in range(10):
                tr, te = okx & (fold != f), okx & (fold == f)
                if te.sum() == 0: continue
                mod = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = mod.predict(X[te])
            preds[sname] = pred
            v = ~np.isnan(pred); r = np.corrcoef(y[v], pred[v])[0, 1]
            ceil_block = 'teacher' if tname.startswith('teacher') else 'all_state'; par = tname.split('_', 1)[1]
            rr = rel[(rel.block == ceil_block) & (rel.experiment == e) & (rel.param == par)]
            rrows.append(dict(experiment=e, target=tname, source=sname, r=r, n=int(v.sum()), noise_ceiling=rr.r_split.iloc[0] if len(rr) else np.nan))
        if 'score1' in preds:
            p0 = preds['score1']
            for sname in ['z4', 'z4+score1', 'search_stats_ridge', 'teacher1', 'z4+teacher1', 'waiting']:
                if sname not in preds: continue
                p1 = preds[sname]; v = ~np.isnan(p1) & ~np.isnan(p0)
                ci, pb = cboot(y[v], p1[v], p0[v], game[v]); cir, _ = cboot(y[v], p1[v], None, game[v])
                drows.append(dict(experiment=e, target=tname, source=sname, r=np.corrcoef(y[v], p1[v])[0, 1], r_ci_lo=cir[0], r_ci_hi=cir[1],
                                  delta_r=np.corrcoef(y[v], p1[v])[0, 1] - np.corrcoef(y[v], p0[v])[0, 1], ci_lo=ci[0], ci_hi=ci[1], p_boot=pb, n=int(v.sum())))
tr_df = pd.DataFrame(rrows); tr_df.to_csv(R / 'transfer_r.csv', index=False)
print(tr_df.pivot_table(index=['experiment', 'target'], columns='source', values='r').round(2).join(tr_df.groupby(['experiment', 'target']).noise_ceiling.first()).to_string())
d_df = pd.DataFrame(drows); d_df.to_csv(R / 'transfer_delta.csv', index=False)
# ---- 3. decomposition ---------------------------------------------------------
g = zdf.join(base).join(T2[SOC + ['score', 'frac_stop_inspot']], rsuffix='_t2')
g = g[g.n_players >= 2].copy()
for c in [f'z{i}' for i in range(4)] + ['score1']:
    g[f'oth_{c}'] = g.groupby('gameid')[c].transform(lambda v: (v.sum() - v) / (len(v) - 1))
    g[f'sd_{c}'] = g.groupby('gameid')[c].transform(lambda v: v.std() if len(v) >= 3 else np.nan)
g['sd_z0'] = g['sd_z0'].fillna(0); 
for c in ['sd_z1', 'sd_z2', 'sd_z3', 'sd_score1']: g[c] = g[c].fillna(0)
g['has_sd'] = (g.n_players >= 3).astype(float)
own = ' + '.join([f'z{i}' for i in range(4)]); oth = ' + '.join([f'oth_z{i}' for i in range(4)]); sd = ' + '.join([f'sd_z{i}' for i in range(4)]) + ' + has_sd'
dec = []
for outc in ['in_face', 'out_face', 'out_face_stop', 'approach', 'nn_dist', 'score']:
    prev = None
    for name, fm in [('ctrl', f'{outc} ~ C(experiment) + score1'), ('+own z', f'{outc} ~ C(experiment) + score1 + {own}'),
                     ('+others mean z', f'{outc} ~ C(experiment) + score1 + {own} + {oth}'), ('+group SD z', f'{outc} ~ C(experiment) + score1 + {own} + {oth} + {sd}')]:
        dd = g.dropna(subset=[outc])
        try:
            m = smf.mixedlm(fm, dd, groups=dd['gameid']).fit(reml=False); llf, k = m.llf, len(m.fe_params); r2 = (m.model.exog @ m.fe_params.to_numpy()).var() / dd[outc].var()
        except Exception:
            m = smf.ols(fm, dd).fit(); llf, k, r2 = m.llf, len(m.params), m.rsquared
        p = np.nan if prev is None else chi2.sf(2 * (llf - prev[0]), k - prev[1]); dec.append(dict(outcome=outc, model=name, r2_marginal=r2, llf=llf, p_lrt=p, n=len(dd))); prev = (llf, k)
dec = pd.DataFrame(dec); dec.to_csv(R / 'decomposition.csv', index=False); print(dec.round(3).to_string())
# ---- 4. group level ----------------------------------------------------------
gr = g.groupby('gameid').agg(experiment=('experiment', 'first'), n_players=('n_players', 'first'), field=('field', 'first'), score2=('score', 'mean'),
                             **{f'mean_z{i}': (f'z{i}', 'mean') for i in range(4)}, **{f'sd_z{i}': (f'z{i}', 'std') for i in range(4)},
                             mean_score1=('score1', 'mean'), sd_score1=('score1', 'std')).reset_index()
# raw (non-residualised) second-half group score and first-half member score for the zero-shot comparison
raw2 = A2.groupby('gameid')['score'].mean().rename('score2_raw'); raw1 = A1.groupby('gameid')['score'].mean().rename('score1_raw')
gr = gr.join(raw2, on='gameid').join(raw1, on='gameid')
gr.to_csv(R / 'group_table.csv', index=False)
y = gr.score2.to_numpy(); gid = gr.gameid.to_numpy(); ug = np.unique(gid)
def logo(cols, sub):
    X = gr.loc[sub, cols].to_numpy(); yy = y[sub]; pred = np.full(len(yy), np.nan); gg = gid[sub]
    for s in np.unique(gg):
        tr, te = gg != s, gg == s
        mod = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], yy[tr]); pred[te] = mod.predict(X[te])
    return pred, yy, gg
rows = []
sub_all = np.ones(len(gr), bool); sub3 = (gr.n_players >= 3).to_numpy()
# zero-shot: mean first-half score (residualised) vs second-half group score
r0 = np.corrcoef(gr.mean_score1, y)[0, 1]; ci, _ = cboot(y, gr.mean_score1.to_numpy(), None, gid)
rows.append(dict(subset='n>=2', model='nominal mean first-half score (zero-shot)', r=r0, ci_lo=ci[0], ci_hi=ci[1], delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan, n=len(gr)))
rows.append(dict(subset='n>=2 raw', model='nominal raw mean first-half score vs raw second-half group score', r=np.corrcoef(gr.score1_raw, gr.score2_raw)[0, 1], ci_lo=np.nan, ci_hi=np.nan, delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan, n=len(gr)))
sets = {'mean_score1': ['mean_score1'], 'mean_z4': [f'mean_z{i}' for i in range(4)], 'mean_z4+score1': [f'mean_z{i}' for i in range(4)] + ['mean_score1'],
        'mean_z4+score1+SD': [f'mean_z{i}' for i in range(4)] + ['mean_score1'] + [f'sd_z{i}' for i in range(4)] + ['sd_score1'], 'SD_only': [f'sd_z{i}' for i in range(4)] + ['sd_score1']}
preds = {k: logo(v, sub3) for k, v in sets.items()}
for k, (p, yy, gg) in preds.items():
    r = np.corrcoef(yy, p)[0, 1]; ci, _ = cboot(yy, p, None, gg); rows.append(dict(subset='n>=3', model=k, r=r, ci_lo=ci[0], ci_hi=ci[1], delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p=np.nan, n=len(yy)))
for b_, f_ in [('mean_z4+score1', 'mean_z4+score1+SD'), ('mean_score1', 'mean_z4+score1')]:
    p0, yy, gg = preds[b_]; p1 = preds[f_][0]; d = np.corrcoef(yy, p1)[0, 1] - np.corrcoef(yy, p0)[0, 1]; ci, pb = cboot(yy, p1, p0, gg)
    rows.append(dict(subset='n>=3', model=f'{f_} - {b_}', r=np.nan, ci_lo=np.nan, ci_hi=np.nan, delta_r=d, d_ci_lo=ci[0], d_ci_hi=ci[1], p=pb, n=len(yy)))
res = pd.DataFrame(rows); res.to_csv(R / 'group_pred.csv', index=False); print(res.round(3).to_string())
# composition regression with permutation of the SD block within experiment x n_players
g3 = gr[sub3]; Xm = g3[sets['mean_z4+score1']].to_numpy(); Xs_ = g3[sets['SD_only']].to_numpy(); yy = y[sub3]
def r2(X): return LinearRegression().fit(X, yy).score(X, yy)
r2m, r2f = r2(Xm), r2(np.hstack([Xm, Xs_])); null = []
key = (g3.experiment.astype(str) + '_' + g3.n_players.astype(str)).to_numpy()
for _ in range(2000):
    Xp = Xs_.copy()
    for k in np.unique(key):
        idx = np.where(key == k)[0]; Xp[idx] = Xs_[rng.permutation(idx)]
    null.append(r2(np.hstack([Xm, Xp])) - r2m)
comp = [dict(outcome='score2', r2_mean=r2m, r2_mean_plus_sd=r2f, delta_r2=r2f - r2m, p_perm=(np.array(null) >= r2f - r2m).mean(), n=len(yy))]
for c in sets['SD_only'] + ['mean_score1']:
    rho, pv = spearmanr(g3[c], yy); comp.append(dict(outcome=f'score2~{c}', r2_mean=np.nan, r2_mean_plus_sd=np.nan, delta_r2=rho, p_perm=pv, n=len(yy)))
comp = pd.DataFrame(comp); comp.to_csv(R / 'group_composition.csv', index=False); print(comp.round(3).to_string())
