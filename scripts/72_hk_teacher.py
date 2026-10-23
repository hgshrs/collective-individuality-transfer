"""Per-participant social-weight 'teacher' for Hawkins et al. (2023), analogous to Chirkov's rotation model.

Decision = change of movement direction. Exp 1: at each goal click, y = signed turn from the previous heading to the
direction of the new goal. Exp 3: every 8 ticks, y = signed heading change over the next second (rad).
Predictors (signed angles from the previous heading, in [-pi, pi], 0 when absent, with presence indicators):
  a_cen  angle to the centroid of the other players
  a_stop angle to the nearest 'stopped' other (exp 1: v = 0; exp 3: slow, v = 2.125)  -> payoff inference (others who stop are likely scoring)
  a_nn   angle to the nearest other
  own state: inspot (in the scoring region) and its interactions with a_cen and a_stop
Model: y ~ b0 + b_cen a_cen + b_stop a_stop + b_nn a_nn + b_cen_spot a_cen*inspot + b_stop_spot a_stop*inspot + b_spot inspot
Fits: population OLS per experiment; per participant x half (first/second/all) empirical-Bayes shrinkage of OLS towards the
population (tau^2 per coefficient by method of moments). Only participants with >= 15 decisions with others present.
Output: results/hawkins2023/teacher_weights.csv (long), teacher_reliability.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[1]; D = P / 'data/processed/hawkins2023'; R = P / 'results/hawkins2023'
t = pd.read_parquet(D / 'ticks.parquet'); games = pd.read_parquet(D / 'games.parquet')
t = t.merge(games[['experiment', 'gameid', 'complete']], on=['experiment', 'gameid']); t = t[t.complete & ~t.dropped_first_half]
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
COEFS = ['b0', 'b_cen', 'b_stop', 'b_nn', 'b_cen_spot', 'b_stop_spot', 'b_spot']
dec = []
for (exp, gid), g in t.groupby(['experiment', 'gameid']):
    pids = sorted(g.pid.unique())
    if len(pids) < 2: continue
    T = int(g.tick.max()) + 1
    A = {p: g[g.pid == p].set_index('tick').reindex(range(T))[['x', 'y', 'v', 'angle', 'bg', 'goal_x', 'goal_y']].to_numpy(float) for p in pids}
    for p in pids:
        a = A[p]; x, y, v, ang, bg = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        head = np.radians((ang - 90) % 360); inspot = (bg >= 0.5) if exp == 3 else (bg >= 1)
        others = [q for q in pids if q != p]
        ox = np.stack([A[q][:, 0] for q in others], 1); oy = np.stack([A[q][:, 1] for q in others], 1); ov = np.stack([A[q][:, 2] for q in others], 1)
        dist = np.hypot(ox - x[:, None], oy - y[:, None]); present = ~np.isnan(dist)
        ostop = ((ov == 0) if exp == 1 else (ov == 2.125)) & present
        if exp == 1:
            gx, gy = a[:, 5], a[:, 6]; idx = np.where((np.diff(gx, prepend=gx[0]) != 0) | (np.diff(gy, prepend=gy[0]) != 0))[0]
            idx = idx[idx > 0]
        else:
            idx = np.arange(8, T - 8, 8)
        for i in idx:
            if np.isnan(x[i]) or not present[i].any(): continue
            h0 = head[i - 1] if exp == 1 else head[i]
            if np.isnan(h0): continue
            if exp == 1:
                yv = wrap(np.arctan2(gy[i] - y[i], gx[i] - x[i]) - h0)
            else:
                if np.isnan(head[i + 8]): continue
                yv = wrap(head[i + 8] - h0)
            cx, cy = np.nanmean(ox[i]), np.nanmean(oy[i]); a_cen = wrap(np.arctan2(cy - y[i], cx - x[i]) - h0)
            dd = np.where(present[i], dist[i], np.inf); j = int(np.argmin(dd)); a_nn = wrap(np.arctan2(oy[i, j] - y[i], ox[i, j] - x[i]) - h0)
            if ostop[i].any():
                ds = np.where(ostop[i], dist[i], np.inf); k = int(np.argmin(ds)); a_stop = wrap(np.arctan2(oy[i, k] - y[i], ox[i, k] - x[i]) - h0); has_stop = 1
            else:
                a_stop = 0.0; has_stop = 0
            dec.append((exp, gid, p, int(i), 'first' if i < 1200 else 'second', yv, a_cen, a_stop, has_stop, a_nn, float(inspot[i])))
dec = pd.DataFrame(dec, columns=['experiment', 'gameid', 'pid', 'tick', 'half', 'y', 'a_cen', 'a_stop', 'has_stop', 'a_nn', 'inspot']); n0 = len(dec); dec = dec.dropna(); print('dropped NaN decisions', n0 - len(dec))
dec.to_parquet(D / 'decisions.parquet', index=False); print('decisions', dec.shape, dec.groupby('experiment').size().to_dict())
def X_of(d):
    return np.column_stack([np.ones(len(d)), d.a_cen, d.a_stop, d.a_nn, d.a_cen * d.inspot, d.a_stop * d.inspot, d.inspot])
rows = []; relrows = []
for exp, de in dec.groupby('experiment'):
    X = X_of(de); yv = de.y.to_numpy()
    beta_pop = np.linalg.lstsq(X, yv, rcond=None)[0]; resid = yv - X @ beta_pop; s2 = resid.var()
    # per-participant OLS (ridge-stabilised) for tau^2 estimation on 'all'
    ols = {}; se2 = {}
    for p, d in de.groupby('pid'):
        if len(d) < 15: continue
        Xi = X_of(d); yi = d.y.to_numpy()
        XtX = Xi.T @ Xi + 1e-3 * np.eye(7); b = np.linalg.solve(XtX, Xi.T @ yi); ols[p] = b
        se2[p] = s2 * np.diag(np.linalg.inv(XtX))
    B = np.array(list(ols.values())); S = np.array(list(se2.values()))
    # between-participant variance from the covariance of independent first- and second-half OLS estimates (unbiased, no floor artefact)
    Bh = {}
    for half in ['first', 'second']:
        for p, d in de[de.half == half].groupby('pid'):
            if len(d) < 10: continue
            Xi = X_of(d); yi = d.y.to_numpy(); Bh[(p, half)] = np.linalg.solve(Xi.T @ Xi + 1e-3 * np.eye(7), Xi.T @ yi)
    common = [p for p in ols if (p, 'first') in Bh and (p, 'second') in Bh]
    B1 = np.array([Bh[(p, 'first')] for p in common]); B2 = np.array([Bh[(p, 'second')] for p in common])
    cov12 = np.array([np.cov(B1[:, j], B2[:, j])[0, 1] for j in range(7)])
    tau2 = np.maximum(cov12, 1e-4)
    print(exp, 'tau2 from half-covariance', cov12.round(4), 'moment estimate', (B.var(0) - S.mean(0)).round(4))
    print(exp, 'n fit', len(B), 'pop', beta_pop.round(3), 'tau', np.sqrt(tau2).round(3), 'sd_ols', B.std(0).round(3))
    for half in ['all', 'first', 'second']:
        sub = de if half == 'all' else de[de.half == half]
        for p, d in sub.groupby('pid'):
            if len(d) < 15: continue
            Xi = X_of(d); yi = d.y.to_numpy()
            # MAP with prior N(beta_pop, diag(tau2)) and noise s2
            A_ = Xi.T @ Xi / s2 + np.diag(1 / tau2); bvec = Xi.T @ yi / s2 + beta_pop / tau2; b = np.linalg.solve(A_, bvec)
            for c, val in zip(COEFS, b): rows.append(dict(experiment=exp, pid=p, gameid=d.gameid.iloc[0], half=half, n_dec=len(d), coef=c, value=val, dev=val - beta_pop[COEFS.index(c)]))
    W = pd.DataFrame([r for r in rows if r['experiment'] == exp]).pivot_table(index=['pid', 'coef'], columns='half', values='value').reset_index()
    for c in COEFS:
        w = W[W.coef == c].dropna(subset=['first', 'second']); r = np.corrcoef(w['first'], w['second'])[0, 1]
        relrows.append(dict(experiment=exp, coef=c, r_split=r, spearman_brown=2 * r / (1 + r), n=len(w), pop=beta_pop[COEFS.index(c)], tau=np.sqrt(tau2[COEFS.index(c)])))
pd.DataFrame(rows).to_csv(R / 'teacher_weights.csv', index=False)
rel = pd.DataFrame(relrows); rel.to_csv(R / 'teacher_reliability.csv', index=False); print(rel.round(3).to_string())
