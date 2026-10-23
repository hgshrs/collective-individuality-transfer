"""A-H2 (cross-individual transfer: the error of predicting participant k's social-phase parameters from participant l's
private-phase z grows with ||z_k - z_l||) recomputed with the LINEAR z used in the paper (AGENT_SPEC §4.2-5; the Phase 1
report computed it with the sequence-encoder z). Same procedure as scripts/12: within every held-out fold, every ordered
pair (k, l), k != l: standardised squared error of the fold's decoder applied to z_l against the teacher of k, averaged over
coefficients, log-transformed; mixed model log_err ~ standardised distance with a random intercept per k; distance = ||z_k - z_l||
(z), Euclidean distance of the standardised private-phase summary statistics (s), or both.
  Chirkov: PLS-4 z and decoder refitted per fold exactly as scripts/11 (search summary statistics + condition one-hot ->
           6 tracking random effects); folds = split_v1 (participants). Only a light regression refit; no ABM, no encoder.
  Wu:      solo z (results/wu2025/z_solo_pca4.csv) -> 5 group-round choice weights; decoder = RidgeCV per session fold (8 folds).
  Hawkins: Exp 1 first-half search-state z (results/hawkins2023/z_first_half.csv) -> second-half targets (stopping, social,
           score; residualised within experiment x group size x field as scripts/73); decoder = RidgeCV per game fold (10).
Outputs: results/phase1/ah2_linear_z.csv (+ per-coefficient rows), results/pooled/ah2_wu.csv, results/pooled/ah2_hawkins.csv
Usage: OMP_NUM_THREADS=1 uv run python scripts/90_ah2_linear_z.py
"""
import json, warnings, numpy as np, pandas as pd, yaml
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from pathlib import Path
warnings.filterwarnings('ignore')
P = Path(__file__).resolve().parents[1]; R = P / 'results'
import sys; sys.path.insert(0, str(P / 'src'))
from collective_id.paths import CONFIGS, PROCESSED
from collective_id.data import load_features
from collective_id.split import load_split
from collective_id.phase1 import load_teacher

def fit_models(X, label, per_coef=None):
    """X: pairs with log_err, dz, ds (+ optional per-coef log errors). Returns rows of (dataset, model, term, slope, se, p, n_pairs, n_k)."""
    X = X.copy(); X['dz_s'] = (X.dz - X.dz.mean()) / X.dz.std(); X['ds_s'] = (X.ds - X.ds.mean()) / X.ds.std(); rows = []
    for name, form in [('z distance only', 'log_err ~ dz_s'), ('summary-statistic distance only', 'log_err ~ ds_s'), ('both', 'log_err ~ dz_s + ds_s')]:
        m = smf.mixedlm(form, X, groups=X['k']).fit(reml=False)
        for term in ['dz_s', 'ds_s']:
            if term in m.params: rows.append(dict(dataset=label, model=name, target='mean over coefficients', term=term, slope=m.params[term], se=m.bse[term], p=m.pvalues[term], n_pairs=len(X), n_k=X.k.nunique(), llf=m.llf))
    for c in (per_coef or []):
        m = smf.mixedlm(f'Q("{c}") ~ dz_s', X, groups=X['k']).fit(reml=False)
        rows.append(dict(dataset=label, model='z distance only', target=c, term='dz_s', slope=m.params['dz_s'], se=m.bse['dz_s'], p=m.pvalues['dz_s'], n_pairs=len(X), n_k=X.k.nunique(), llf=m.llf))
    return pd.DataFrame(rows)

def pairs_from(te, Z, pred_of, T, sd, S, coefs):
    """te: list of ids in the fold; Z: DataFrame of z; pred_of(id) -> predicted target vector (decoder applied to that id's z); T: teacher DataFrame; S: standardised private stats."""
    rows = []
    preds = {l: pred_of(l) for l in te}
    for k in te:
        tk = T.loc[k, coefs].to_numpy(float)
        for l in te:
            if k == l: continue
            e = ((preds[l] - tk) / sd) ** 2
            rows.append(dict(k=k, l=l, err=e.mean(), dz=np.linalg.norm(Z.loc[k] - Z.loc[l]), ds=np.linalg.norm(S.loc[k] - S.loc[l]), **{c: np.log(e[i] + 1e-6) for i, c in enumerate(coefs)}))
    X = pd.DataFrame(rows); X['log_err'] = np.log(X.err + 1e-6); return X

# ================= Chirkov: PLS-4 linear z (as scripts/11) =================
cfg = yaml.safe_load(open(CONFIGS / 'phase1.yaml')); coefs = cfg['target_coefs']
teacher, _ = load_teacher('runs/eb/tracking_all_ranef.csv', coefs)  # EB teacher (as scripts/11 --teacher ... --tag ridge_eb)
d = load_features(); meta = d.groupby('participant').agg(cond=('cond', 'first'), speed=('speed', 'first')).loc[teacher.index]
S = pd.read_csv(PROCESSED / 'search_summary_stats.csv', index_col=0).reindex(teacher.index).fillna(0.0)
Sstd = (S - S.mean()) / (S.std() + 1e-9)
sp = load_split(); fold = pd.Series({p: sp['fold_of_participant'][p] for p in teacher.index})
onehot = pd.get_dummies(meta.cond.astype(pd.CategoricalDtype(['A', 'NP', 'VP', 'FP']))).astype(float).join(pd.get_dummies(meta.speed.astype(pd.CategoricalDtype(['fast', 'slow']))).astype(float))
sd_all = teacher[coefs].std().to_numpy(); Xs = []
Zlin_check = pd.read_csv(R / 'phase1/z_linear_pls4_ridge_eb.csv', index_col=0)
for f in sorted(fold.unique()):
    tr, te = fold[fold != f].index, fold[fold == f].index
    mu, sd = S.loc[tr].mean(), S.loc[tr].std() + 1e-9
    Xtr, Xte = ((S.loc[tr] - mu) / sd).to_numpy(), ((S.loc[te] - mu) / sd).to_numpy()
    Ytr = ((teacher.loc[tr] - teacher.loc[tr].mean()) / (teacher.loc[tr].std() + 1e-9)).to_numpy()
    pls = PLSRegression(n_components=4, scale=False).fit(np.hstack([Xtr, onehot.loc[tr].to_numpy()]), Ytr)
    Bte = np.hstack([Xte, onehot.loc[te].to_numpy()]); Zte = pd.DataFrame(pls.transform(Bte), index=te)
    assert np.allclose(Zte.to_numpy(), Zlin_check.loc[te].to_numpy(), atol=1e-6), 'PLS z differs from the stored z_linear_pls4_ridge_eb.csv'
    pred = pd.DataFrame(pls.predict(Bte) * (teacher.loc[tr].std() + 1e-9).to_numpy() + teacher.loc[tr].mean().to_numpy(), index=te)
    Xs.append(pairs_from(list(te), Zte, lambda l: pred.loc[l].to_numpy(), teacher, sd_all, Sstd, coefs))
X = pd.concat(Xs, ignore_index=True); X.to_csv(R / 'phase1/ah2_linear_z_pairs.csv', index=False)
res_c = fit_models(X, 'Chirkov (PLS-4 linear z)', per_coef=coefs); res_c.to_csv(R / 'phase1/ah2_linear_z.csv', index=False)
pd.set_option('display.width', 250); print(res_c.round(4).to_string())

# ================= Wu: solo z -> group choice weights =================
D = P / 'data/processed/wu2025'; RW = R / 'wu2025'
feats = pd.read_parquet(D / 'round_features.parquet'); cw = pd.read_csv(RW / 'choice_weights.csv'); Z = pd.read_csv(RW / 'z_solo_pca4.csv', index_col=0)
GW5 = ['Locality', 'BlockVisibility', 'RewardPrediction', 'SuccessfulProximity', 'UnsuccessfulProximity']
T = cw[(cw.type == 'group') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value')[GW5].reindex(Z.index)
MOV = ['path_len', 'speed_mean', 'frac_moving', 'n_bouts', 'turn_move_mean', 'yaw_rate_mean', 'yaw_rate_sd', 'pitch_mean', 'n_cells', 'rgyr', 'straightness']
FOR = ['n_blocks', 'n_rewards', 'reward_rate', 'ibd_mean', 'ibd_median', 'ibd_cv', 'ibi_mean', 'ars_dist', 'ars_time', 't_first_block']
solo = feats[feats.type == 'solo'].copy()
for c in MOV + FOR: solo[c] = solo.groupby('env')[c].transform(lambda v: (v - v.mean()) / v.std())
Sw = solo.groupby('pid')[MOV + FOR].mean().reindex(Z.index); Sw = (Sw - Sw.mean()) / (Sw.std() + 1e-9)
folds = json.load(open(D / 'split_v1.json')); sess = pd.Series({p: int(p.split('session')[1].split('.')[0]) for p in Z.index}); fw = sess.map(lambda s: folds[str(s)])
sd_w = T.std().to_numpy(); Xs = []
for f in sorted(fw.unique()):
    tr, te = fw[fw != f].index, fw[fw == f].index
    m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Z.loc[tr].to_numpy(), T.loc[tr].to_numpy())
    pred = pd.DataFrame(m.predict(Z.loc[te].to_numpy()), index=te)
    Xs.append(pairs_from(list(te), Z, lambda l: pred.loc[l].to_numpy(), T, sd_w, Sw, GW5))
Xw = pd.concat(Xs, ignore_index=True); res_w = fit_models(Xw, 'Wu (solo PCA-4 z -> group choice weights)', per_coef=GW5); res_w.to_csv(R / 'pooled/ah2_wu.csv', index=False); print(res_w.round(4).to_string())

# ================= Hawkins Exp 1: first-half search-state z -> second-half targets =================
DH = P / 'data/processed/hawkins2023'; RH = R / 'hawkins2023'
F = pd.read_parquet(DH / 'features.parquet'); games = pd.read_parquet(DH / 'games.parquet'); zf = pd.read_csv(RH / 'z_first_half.csv')
F = F.merge(games[['experiment', 'gameid', 'field', 'noise']], on=['experiment', 'gameid'])
PRIV = ['frac_stop', 'frac_slow', 'frac_fast', 'path_len', 'turn_rate', 'turn_sd', 'frac_turning', 'n_cells', 'click_rate', 'click_dist', 'click_turn', 'click_dist_cv']
TGT = ['frac_stop_inspot', 'stop_latency', 'n_exit', 'n_enter', 'out_face', 'out_face_stop', 'out_face_mov', 'face_select', 'in_face', 'in_face_n', 'approach', 'nn_dist', 'cen_dist', 'frac_near50', 'score']
def cell_resid(df, cols):
    out = df.copy(); key = out['experiment'].astype(str) + '_' + out['n_players'].astype(str) + '_' + out['field'].astype(str)
    for c in cols:
        out[c] = out[c] - out.groupby(key)[c].transform('mean'); out[c] = out[c] / out.groupby('experiment')[c].transform('std')
    return out
zf1 = zf[zf.experiment == 1].set_index('pid'); Zh = zf1[['z0', 'z1', 'z2', 'z3']]
A2 = cell_resid(F[(F.half == 'second') & (F.state == 'all')].reset_index(drop=True), TGT).set_index('pid')[TGT].reindex(Zh.index)
S1 = cell_resid(F[(F.half == 'first') & (F.state == 'search')].reset_index(drop=True), PRIV).set_index('pid')[PRIV].reindex(Zh.index); S1 = S1.fillna(S1.mean())
ok = A2.notna().all(1) & Zh.notna().all(1); Zh, A2, S1 = Zh[ok], A2[ok], S1[ok]
fold_of = json.load(open(DH / 'split_v1.json')); fh = zf1.loc[Zh.index, 'gameid'].map(lambda g: fold_of[str(g)] if str(g) in fold_of else fold_of[g])
sd_h = A2.std().to_numpy(); Xs = []
for f in sorted(fh.unique()):
    tr, te = fh[fh != f].index, fh[fh == f].index
    m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Zh.loc[tr].to_numpy(), A2.loc[tr].to_numpy())
    pred = pd.DataFrame(m.predict(Zh.loc[te].to_numpy()), index=te)
    Xs.append(pairs_from(list(te), Zh, lambda l: pred.loc[l].to_numpy(), A2, sd_h, S1, TGT))
Xh = pd.concat(Xs, ignore_index=True); res_h = fit_models(Xh, 'Hawkins Exp 1 (first-half search z -> second-half targets)', per_coef=TGT); res_h.to_csv(R / 'pooled/ah2_hawkins.csv', index=False); print(res_h.round(4).to_string())
