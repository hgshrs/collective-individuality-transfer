"""Composition test with selective-social-learning traits measured IN THE GROUP (HANDOVER §12 item 3).

Objection addressed: 'the composition effect was missed because the trait that matters (selective social learning) is the
one that cannot be measured in private'. Here that trait is measured from the first half of group play and its within-group
SD is tested as a predictor of the second-half group outcome, over and above member means.
  Wu 2025:     members' group-round choice weights fitted on split A (rounds 1-2 of each 4-round block, same env) ->
               total reward of the split-B rounds (rounds 3-4; 32 sessions x 2 envs x 2 rounds = 128 group rounds, z within env).
               Base = mean solo reward + mean solo z4 + means of the 5 A-weights; added = SD of SuccessfulProximity (A), of both
               social weights, or of all 5 weights. Leave-one-session-out ridge, Delta r with session bootstrap; in-sample
               Delta R2 with permutation of the SD block within env; Spearman of each SD with the B-round reward.
  Hawkins 2023: first-half in-group measures (facing stopped others, selectivity index, teacher weights on stopped others)
               -> second-half group score, games with >= 3 players (169). Base = mean z4 + mean first-half score + means of
               the selectivity measures; added = their SDs. Leave-one-game-out ridge, game bootstrap, permutation within
               experiment x group size.
Outputs: results/pooled/ingroup_selectivity_wu.csv, ingroup_selectivity_hawkins.csv
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
P = Path(__file__).resolve().parents[1]; R = P / 'results'; OUT = R / 'pooled'; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(3)

def oof(X, y, cl):
    pred = np.full(len(y), np.nan)
    for s in np.unique(cl):
        tr, te = cl != s, cl == s
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred
def cboot(y, p1, p0, cl, B=3000):
    uc = np.unique(cl); out = []
    for _ in range(B):
        take = rng.choice(uc, len(uc), replace=True); idx = np.concatenate([np.where(cl == c)[0] for c in take])
        r1 = np.corrcoef(y[idx], p1[idx])[0, 1]; out.append(r1 if p0 is None else r1 - np.corrcoef(y[idx], p0[idx])[0, 1])
    out = np.array(out); return np.percentile(out, [2.5, 97.5]), (out <= 0).mean()
def perm_r2(Xm, Xs, y, strata, B=3000):
    def r2(X): return LinearRegression().fit(X, y).score(X, y)
    r2m = r2(Xm); d = r2(np.hstack([Xm, Xs])) - r2m; null = []
    for _ in range(B):
        Xp = Xs.copy()
        for k in np.unique(strata):
            idx = np.where(strata == k)[0]; Xp[idx] = Xs[rng.permutation(idx)]
        null.append(r2(np.hstack([Xm, Xp])) - r2m)
    return r2m, d, (np.array(null) >= d).mean()
def evaluate(label, df, y, cl, strata, base_cols, added_sets, out_path):
    rows = []
    p0 = oof(df[base_cols].to_numpy(), y, cl); r0 = np.corrcoef(y, p0)[0, 1]; ci0, _ = cboot(y, p0, None, cl)
    rows.append(dict(dataset=label, model='base (means)', r=r0, ci_lo=ci0[0], ci_hi=ci0[1], delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p_boot=np.nan, r2_base=np.nan, delta_r2=np.nan, p_perm=np.nan, n=len(y)))
    for name, sdc in added_sets.items():
        p1 = oof(df[base_cols + sdc].to_numpy(), y, cl); r1 = np.corrcoef(y, p1)[0, 1]; ci, pb = cboot(y, p1, p0, cl)
        r2m, d2, pp = perm_r2(df[base_cols].to_numpy(), df[sdc].to_numpy(), y, strata)
        rows.append(dict(dataset=label, model=f'base + {name}', r=r1, ci_lo=np.nan, ci_hi=np.nan, delta_r=r1 - r0, d_ci_lo=ci[0], d_ci_hi=ci[1], p_boot=pb, r2_base=r2m, delta_r2=d2, p_perm=pp, n=len(y)))
    for c in sorted({c for s in added_sets.values() for c in s}):
        rho, pv = spearmanr(df[c], y); rows.append(dict(dataset=label, model=f'spearman y ~ {c}', r=rho, ci_lo=np.nan, ci_hi=np.nan, delta_r=np.nan, d_ci_lo=np.nan, d_ci_hi=np.nan, p_boot=pv, r2_base=np.nan, delta_r2=np.nan, p_perm=np.nan, n=len(y)))
    res = pd.DataFrame(rows); res.to_csv(out_path, index=False); pd.set_option('display.width', 250); print(res.round(3).to_string()); return res

# ================= Wu 2025 =================
D = P / 'data/processed/wu2025'; RW = R / 'wu2025'
feats = pd.read_parquet(D / 'round_features.parquet'); cw = pd.read_csv(RW / 'choice_weights.csv'); Z = pd.read_csv(RW / 'z_solo_pca4.csv', index_col=0)
GW5 = ['Locality', 'BlockVisibility', 'RewardPrediction', 'SuccessfulProximity', 'UnsuccessfulProximity']
wA = cw[(cw.type == 'group') & (cw.split == 'A')].pivot(index='pid', columns='param', values='value')[GW5]
solo = feats[feats.type == 'solo']; solo_rew = solo.groupby('pid')['n_rewards'].mean().rename('solo_reward')
member = Z.join(solo_rew).join(wA.add_prefix('A_'))
g = feats[(feats.type == 'group') & (feats.pos_in_block >= 2)]
gr = g.groupby(['session', 'round', 'env']).agg(total_reward=('n_rewards', 'sum')).reset_index()
members = g.groupby(['session', 'round'])['pid'].apply(list)
cols = list(Z.columns) + ['solo_reward'] + [f'A_{c}' for c in GW5]
M = pd.DataFrame([pd.concat([member.loc[m, cols].mean().add_prefix('mean_'), member.loc[m, cols].std().add_prefix('sd_')]) for m in [members[(s, r)] for s, r in zip(gr.session, gr['round'])]])
gr = pd.concat([gr, M], axis=1); gr['y'] = gr.groupby('env')['total_reward'].transform(lambda v: (v - v.mean()) / v.std())
gr.to_csv(OUT / 'ingroup_selectivity_wu_table.csv', index=False)
base = [f'mean_{c}' for c in Z.columns] + ['mean_solo_reward'] + [f'mean_A_{c}' for c in GW5]
added = {'SD SuccessfulProximity (A)': ['sd_A_SuccessfulProximity'], 'SD both social weights (A)': ['sd_A_SuccessfulProximity', 'sd_A_UnsuccessfulProximity'], 'SD all 5 weights (A)': [f'sd_A_{c}' for c in GW5]}
evaluate('Wu 2025 (B rounds)', gr, gr.y.to_numpy(), gr.session.to_numpy(), gr.env.to_numpy(), base, added, OUT / 'ingroup_selectivity_wu.csv')
# split-half check of the group-measured trait (A vs B) for reference
wB = cw[(cw.type == 'group') & (cw.split == 'B')].pivot(index='pid', columns='param', values='value')[GW5]
print('A-vs-B reliability of group weights:', {c: round(wA[c].corr(wB[c]), 3) for c in GW5})

# ================= Hawkins 2023 =================
DH = P / 'data/processed/hawkins2023'; RH = R / 'hawkins2023'
F = pd.read_parquet(DH / 'features.parquet'); games = pd.read_parquet(DH / 'games.parquet'); TW = pd.read_csv(RH / 'teacher_weights.csv')
zf = pd.read_csv(RH / 'z_first_half.csv'); gt = pd.read_csv(RH / 'group_table.csv')
A1 = F[(F.half == 'first') & (F.state == 'all')].merge(games[['experiment', 'gameid', 'field']], on=['experiment', 'gameid'])
SEL = ['out_face_stop', 'face_select']
def cell_resid(df, cols):
    out = df.copy(); key = out['experiment'].astype(str) + '_' + out['n_players'].astype(str) + '_' + out['field'].astype(str)
    for c in cols:
        out[c] = out[c] - out.groupby(key)[c].transform('mean'); out[c] = out[c] / out.groupby('experiment')[c].transform('std')
    return out
tw1 = TW[TW.half == 'first'].pivot(index='pid', columns='coef', values='value')[['b_stop', 'b_stop_spot']]
A1 = A1.set_index('pid').join(tw1).reset_index()
TC = ['b_stop', 'b_stop_spot']
A1 = cell_resid(A1, SEL + TC)
per = A1.set_index('pid')[SEL + TC].reindex(zf.pid).set_index(zf.gameid.to_numpy())
agg = per.groupby(level=0).agg(['mean', 'std']); agg.columns = [f"{'sd' if s == 'std' else s}_{c}" for c, s in agg.columns]
gt = gt.set_index('gameid').join(agg).reset_index(); g3 = gt[gt.n_players >= 3].dropna(subset=[f'sd_{c}' for c in SEL + TC]).reset_index(drop=True)
g3.to_csv(OUT / 'ingroup_selectivity_hawkins_table.csv', index=False)
base = [f'mean_z{i}' for i in range(4)] + ['mean_score1'] + [f'mean_{c}' for c in SEL + TC]
added = {'SD facing stopped others': ['sd_out_face_stop'], 'SD selectivity index': ['sd_face_select'], 'SD teacher b_stop': ['sd_b_stop'], 'SD all 4 selectivity measures': [f'sd_{c}' for c in SEL + TC]}
strata = (g3.experiment.astype(str) + '_' + g3.n_players.astype(str)).to_numpy()
evaluate('Hawkins 2023 (second half)', g3, g3.score2.to_numpy(), g3.gameid.to_numpy(), strata, base, added, OUT / 'ingroup_selectivity_hawkins.csv')
