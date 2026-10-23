"""Pooled equivalence test for the composition effect across the three datasets (HANDOVER §12 item 2).

Per dataset, the composition contrast is Delta r = r(model with within-group heterogeneity) - r(model with member means only),
re-computed here from the stored per-unit predictions / tables so that all units can be bootstrapped jointly:
  Chirkov D-1  ABM: members' fitted policies (main) vs group-mean agents (ctrl2b_gmean); 103 groups; score z within cond x speed
  Chirkov D-2  neural policy, same contrast (secondary)
  Wu ABM       authors' grid ABM main vs gmean; 64 session x env cells (secondary)
  Wu stat      LOSO ridge: member means (z4 + solo reward) + SDs vs means; 256 group rounds; cluster = session (32)
  Hawkins stat LOGO ridge: member means (z4 + first-half score) + SDs vs means; 169 games with >= 3 players
Pooling: (a) 'pooled r' = correlations computed on the concatenated within-cell-standardised units;
         (b) 'n-weighted mean' of the per-dataset Delta r. Stratified cluster bootstrap (5,000) -> two-sided 95 % CI,
one-sided 95 % upper bound, and the TOST equivalence bound (upper end of the 90 % CI). Primary set = Chirkov D-1, Wu stat,
Hawkins stat (one contrast per dataset); 'all' = the five contrasts.
Outputs: results/pooled/composition_units.csv, composition_pooled.csv, fig5_composition_pooled.{pdf,png} (paper_figures)
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
P = Path(__file__).resolve().parents[1]; R = P / 'results'; OUT = R / 'pooled'; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

def z(s): return (s - s.mean()) / s.std(ddof=0)
def zby(s, key): return s.groupby(key).transform(z)
units = []
# ---------- Chirkov D-1 / D-2 ----------
O = pd.read_csv(R / 'phase3/observed_group_metrics.csv').set_index('group')
for name, f in [('Chirkov D-1 ABM', 'sim_runs.parquet'), ('Chirkov D-2 neural', 'sim_runs_d2.parquet')]:
    S = pd.read_parquet(R / 'phase3' / f)
    pm = S[S.variant == 'main'].groupby('group')['score'].mean(); pc = S[S.variant == 'ctrl2b_gmean'].groupby('group')['score'].mean()
    g = sorted(set(O.index) & set(pm.index) & set(pc.index)); d = pd.DataFrame({'obs': O.loc[g, 'score'], 'p_het': pm.loc[g], 'p_mean': pc.loc[g], 'cell': O.loc[g, 'cell']})
    for c in ['obs', 'p_het', 'p_mean']: d[c] = zby(d[c], d['cell'])
    units.append(pd.DataFrame({'dataset': name, 'cluster': [f'C_{x}' for x in g], 'obs': d.obs.to_numpy(), 'p_het': d.p_het.to_numpy(), 'p_mean': d.p_mean.to_numpy()}))
# ---------- Wu ABM ----------
gr = pd.read_csv(R / 'wu2025/group_rounds.csv'); sim = pd.read_parquet(R / 'wu2025/sim_runs.parquet')
obs = gr.groupby(['session', 'env'])['total_reward'].mean()
agg = sim[sim.variant.isin(['main', 'gmean'])].groupby(['session', 'env', 'variant'])['total_reward'].mean().unstack('variant').reindex(obs.index)
d = pd.DataFrame({'obs': obs, 'p_het': agg['main'], 'p_mean': agg['gmean']}).reset_index()
for c in ['obs', 'p_het', 'p_mean']: d[c] = zby(d[c], d['env'])
units.append(pd.DataFrame({'dataset': 'Wu grid ABM', 'cluster': 'W_' + d.session.astype(str), 'obs': d.obs, 'p_het': d.p_het, 'p_mean': d.p_mean}))
# ---------- Wu statistical (LOSO ridge, as scripts/65) ----------
zc = ['z0', 'z1', 'z2', 'z3']; sess = gr.session.to_numpy(); us = np.unique(sess)
def loso(cols, y):
    X = gr[cols].to_numpy(); pred = np.full(len(y), np.nan)
    for s in us:
        tr, te = sess != s, sess == s
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred
y = gr['total_reward_z'].to_numpy()
means = [f'mean_{c}' for c in zc] + ['mean_solo_reward']; sds = [f'sd_{c}' for c in zc] + ['sd_solo_reward']
units.append(pd.DataFrame({'dataset': 'Wu statistical', 'cluster': 'W_' + gr.session.astype(str), 'obs': y, 'p_het': loso(means + sds, y), 'p_mean': loso(means, y)}))
# ---------- Hawkins statistical (LOGO ridge, as scripts/73) ----------
gt = pd.read_csv(R / 'hawkins2023/group_table.csv'); g3 = gt[gt.n_players >= 3].reset_index(drop=True)
y = g3.score2.to_numpy(); gid = g3.gameid.to_numpy()
def logo(cols):
    X = g3[cols].to_numpy(); pred = np.full(len(y), np.nan)
    for s in np.unique(gid):
        tr, te = gid != s, gid == s
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred
means = [f'mean_z{i}' for i in range(4)] + ['mean_score1']; sds = [f'sd_z{i}' for i in range(4)] + ['sd_score1']
units.append(pd.DataFrame({'dataset': 'Hawkins statistical', 'cluster': 'H_' + g3.gameid.astype(str), 'obs': y, 'p_het': logo(means + sds), 'p_mean': logo(means)}))
U = pd.concat(units, ignore_index=True); U.to_csv(OUT / 'composition_units.csv', index=False)

# ---------- per-dataset and pooled Delta r with stratified cluster bootstrap ----------
def r(a, b): return float(np.corrcoef(a, b)[0, 1])
def delta_pooled(df):
    return r(df.obs, df.p_het) - r(df.obs, df.p_mean)
def resample(df):
    parts = []
    for ds, d in df.groupby('dataset', sort=False):
        cl = d.cluster.to_numpy(); uc = np.unique(cl); take = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([np.where(cl == c)[0] for c in take]); parts.append(d.iloc[idx])
    return pd.concat(parts)
PRIMARY = ['Chirkov D-1 ABM', 'Wu statistical', 'Hawkins statistical']; ALL = list(U.dataset.unique())
B = 5000; rows = []
def summarise(label, df, fn, n, ncl):
    obs = fn(df); bs = np.array([fn(resample(df)) for _ in range(B)])
    rows.append(dict(contrast=label, delta_r=obs, ci95_lo=np.quantile(bs, .025), ci95_hi=np.quantile(bs, .975), ci90_lo=np.quantile(bs, .05), ci90_hi=np.quantile(bs, .95),
                     upper95_one_sided=np.quantile(bs, .95), p_boot_two_sided=2 * min((bs <= 0).mean(), (bs >= 0).mean()), n_units=n, n_clusters=ncl,
                     r_het=r(df.obs, df.p_het) if 'obs' in df else np.nan, r_mean=r(df.obs, df.p_mean) if 'obs' in df else np.nan))
for ds in ALL:
    d = U[U.dataset == ds]; summarise(ds, d, delta_pooled, len(d), d.cluster.nunique())
per_n = U.groupby('dataset').size()
def delta_weighted(df, sets):
    vals = {ds: delta_pooled(df[df.dataset == ds]) for ds in sets}
    w = np.array([per_n[ds] for ds in sets], float); return float(np.sum(w * np.array([vals[ds] for ds in sets])) / w.sum())
for label, sets in [('Pooled primary (3 datasets), pooled r', PRIMARY), ('Pooled all (5 contrasts), pooled r', ALL)]:
    d = U[U.dataset.isin(sets)]; summarise(label, d, delta_pooled, len(d), d.cluster.nunique())
for label, sets in [('Pooled primary (3 datasets), n-weighted mean', PRIMARY), ('Pooled all (5 contrasts), n-weighted mean', ALL)]:
    d = U[U.dataset.isin(sets)]; summarise(label, d, lambda df: delta_weighted(df, sets), len(d), d.cluster.nunique())
res = pd.DataFrame(rows); res['tost_equivalence_bound'] = res[['ci90_lo', 'ci90_hi']].abs().max(1)
res.to_csv(OUT / 'composition_pooled.csv', index=False)
pd.set_option('display.width', 250); print(res.round(3).to_string())

# ---------- figure: forest plot with pooled rows (new file; fig5_composition.pdf untouched) ----------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
                     'text.latex.preamble': r'\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}',
                     'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'savefig.transparent': True, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = plt.get_cmap('tab10').colors; DS = {'Chirkov': C[0], 'Wu': C[1], 'Hawkins': C[2], 'Pooled': 'k'}
rr = res.set_index('contrast')
order = [('Chirkov D-1 ABM', 'Chirkov, ABM policy', 'Chirkov'), ('Chirkov D-2 neural', 'Chirkov, neural policy', 'Chirkov'), ('Wu grid ABM', 'Wu, grid ABM', 'Wu'),
         ('Wu statistical', 'Wu, statistical', 'Wu'), ('Hawkins statistical', 'Hawkins, statistical', 'Hawkins'),
         ('Pooled primary (3 datasets), pooled r', 'Pooled, one per dataset', 'Pooled'), ('Pooled all (5 contrasts), pooled r', 'Pooled, all five', 'Pooled')]
fig, ax = plt.subplots(figsize=(2.58, 2.3)); yy = np.arange(len(order))[::-1]
for yi, (k, lab, ds) in zip(yy, order):
    row = rr.loc[k]; ax.plot([row.ci95_lo, row.ci95_hi], [yi, yi], color=DS[ds], lw=1.2)
    ax.plot(row.delta_r, yi, 'D' if ds == 'Pooled' else 'o', color=DS[ds], ms=4)
ax.axhline(1.5, color='0.8', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_yticks(yy); ax.set_yticklabels([o[1] for o in order], fontsize=8); ax.set_xlabel('$\\Delta r$ from within-group heterogeneity (95\\% CI)'); ax.set_xlim(-0.25, 0.45)
PF = R / 'paper_figures'; fig.savefig(PF / 'fig5_composition_pooled.pdf'); fig.savefig(PF / 'fig5_composition_pooled.png', dpi=220, transparent=False, facecolor='white'); plt.close(fig)
