"""Per-participant MAP fits of the Wu et al. (2025) block-choice model (ARS + conditional social learning).

Model (as in modeling/STAN/Minecraft_parallel_conditionalSL_ARS.stan): softmax over available blocks of
  X @ (w_pop[env] + v) + (X[:, adapt] * t_last_reward) @ (bt_pop[env] + v_adapt)
with X standardised within each choice over available blocks (features: distance, block visibility,
GP reward prediction, successful-player proximity, unsuccessful-player proximity; solo: first three),
first decision of each round skipped, social features zeroed when no such player was visible.
Population weights / adaptive terms / prior SDs are the posterior means of the authors' fit
(results/wu2025/stan_population.json); individual deviations v ~ N(0, sigma_ID) (MAP, L-BFGS).
Fits: type in {group, solo} x split in {all, A (rounds 1-2 of each 4-round block), B (rounds 3-4), odd, even}.
Outputs: results/wu2025/choice_weights.csv (long), choice_reliability.csv, choice_vs_stan.csv
"""
import json, sys
import numpy as np, pandas as pd, torch
from pathlib import Path
torch.set_default_dtype(torch.float64)
P = Path('/home/higashi/Research/20260924_collective'); R = P / 'results/wu2025'; D = P / 'data/processed/wu2025'
MF = P / 'external/wu2025/modeling/data/modelFeatures'
pop = json.load(open(R / 'stan_population.json'))
rounds = pd.read_parquet(D / 'rounds.parquet')
import pyreadr
rt = list(pyreadr.read_r(str(P / 'external/wu2025/modeling/data/rewardTimes.RDS')).values())[0]
rt['session'] = rt['session'].str.extract(r'session(\d+)\.json').astype(int)
rt = rt.sort_values(['id', 'round', 'time'])
rt['decision_number'] = rt.groupby(['id', 'round']).cumcount()
FEATS = {'group': ['fruit_distance', 'fruit_pov', 'asocial_GP_pred', 'successful_player_distance', 'visible_player_distance'],
         'solo': ['fruit_distance', 'fruit_pov', 'asocial_GP_pred']}
ADAPT = {'group': [0, 3, 4], 'solo': [0]}
NAMES = {'group': ['Locality', 'BlockVisibility', 'RewardPrediction', 'SuccessfulProximity', 'UnsuccessfulProximity', 'AdaptLocality', 'AdaptSuccessful', 'AdaptUnsuccessful'],
         'solo': ['Locality', 'BlockVisibility', 'RewardPrediction', 'AdaptLocality']}
POPKEY = {'group': 'conditional_ARS', 'solo': 'solo_ARS'}
ENV = {'random': 0, 'smooth': 1}

def load_participant(i):
    f = pd.read_feather(MF / f'{i:03d}.feather')
    f['session'] = f['session'].str.extract(r'session(\d+)\.json').astype(int)
    return f

def build(f, rtype):
    """Return list of decisions: (X [B x F] standardised, chosen index, env, t_last, flags)"""
    feats = FEATS[rtype]
    sub = f[(f.type == rtype) & (f.decision_number > 0)]
    pid = f['id'].iloc[0]
    out = []
    for (rnd, dn), d in sub.groupby(['round', 'decision_number'], sort=True):
        avail = d['fruit_distance'].notna().to_numpy()
        d = d[avail]
        if d['chosen'].sum() != 1:
            continue
        X = np.array(d[feats].to_numpy(dtype=float), copy=True)
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.isnan(col).all():
                X[:, j] = 0.0
            else:
                sd = col.std(ddof=1)
                X[:, j] = (col - col.mean()) / sd if sd > 0 else 0.0
        tl = rt[(rt['id'] == pid) & (rt['round'] == rnd) & (rt['decision_number'] == dn)]
        t_last = float(tl['lastIndReward'].iloc[0]) if len(tl) else np.nan
        out.append(dict(round=int(rnd), X=X, y=int(np.argmax(d['chosen'].to_numpy())), env=ENV[d['env'].iloc[0]], t_last=t_last))
    return out

def fit(decs, rtype, split_rounds):
    feats = FEATS[rtype]; F = len(feats); A = ADAPT[rtype]
    pp = pop[POPKEY[rtype]]
    w_pop = torch.tensor(np.array(pp['weights']))            # F x 2
    bt_pop = torch.tensor(np.array(pp['beta_t_ind']))        # len(A) x 2
    sig = torch.tensor(np.array(pp['sigma_ID']))             # F + len(A)
    ds = [d for d in decs if d['round'] in split_rounds]
    if len(ds) < 10:
        return None, len(ds)
    N = len(ds); Bmax = max(d['X'].shape[0] for d in ds)
    X = np.zeros((N, Bmax, F)); mask = np.full((N, Bmax), -np.inf)
    for i, d in enumerate(ds):
        b = d['X'].shape[0]; X[i, :b] = d['X']; mask[i, :b] = 0.0
    X = torch.tensor(X); mask = torch.tensor(mask)
    y = torch.tensor([d['y'] for d in ds]); env = torch.tensor([d['env'] for d in ds])
    tl = torch.tensor([d['t_last'] for d in ds]); tl = torch.where(torch.isnan(tl), torch.zeros_like(tl), tl)
    Aidx = torch.tensor(A)
    v = torch.zeros(F + len(A), requires_grad=True)
    opt = torch.optim.LBFGS([v], max_iter=500, line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
    def closure():
        opt.zero_grad()
        w = w_pop[:, env].T + v[:F][None, :]                       # N x F
        wa = torch.zeros(N, F); wa[:, Aidx] = (bt_pop[:, env].T + v[F:][None, :]) * tl[:, None]
        logits = torch.einsum('nbf,nf->nb', X, w + wa) + mask
        nll = -(logits[torch.arange(N), y] - torch.logsumexp(logits, 1)).sum()
        loss = nll + 0.5 * ((v / sig) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return v.detach().numpy(), N

def main():
    # standardise t_last within type across all decisions (as in the authors' prep)
    allrows = []
    cache = {}
    for i in range(1, 129):
        f = load_participant(i); pid = f['id'].iloc[0]
        for rtype in ['group', 'solo']:
            cache[(pid, rtype)] = build(f, rtype)
        if i % 16 == 0: print('loaded', i, flush=True)
    for rtype in ['group', 'solo']:
        tl = np.array([d['t_last'] for (p, t), ds in cache.items() if t == rtype for d in ds])
        mu, sd = np.nanmean(tl), np.nanstd(tl)
        for (p, t), ds in cache.items():
            if t == rtype:
                for d in ds: d['t_last'] = (d['t_last'] - mu) / sd
    splits = {}
    for rtype in ['group', 'solo']:
        rr = rounds[rounds.type == rtype]
        splits[rtype] = {'all': set(rr['round']), 'A': set(rr[rr.pos_in_block < 2]['round']), 'B': set(rr[rr.pos_in_block >= 2]['round']),
                         'odd': set(rr[rr.pos_in_block % 2 == 1]['round']), 'even': set(rr[rr.pos_in_block % 2 == 0]['round'])}
    # note: round indices of a given type are identical across sessions (same counterbalancing structure -> check)
    chk = rounds.groupby('round')['type'].nunique().max()
    print('round->type unique across sessions:', chk == 1)
    for (pid, rtype), ds in cache.items():
        session = int(pid.split('session')[1].split('.')[0])
        rr = rounds[(rounds.session == session) & (rounds.type == rtype)]
        sp = {'all': set(rr['round']), 'A': set(rr[rr.pos_in_block < 2]['round']), 'B': set(rr[rr.pos_in_block >= 2]['round']),
              'odd': set(rr[rr.pos_in_block % 2 == 1]['round']), 'even': set(rr[rr.pos_in_block % 2 == 0]['round'])}
        for sname, srounds in sp.items():
            v, n = fit(ds, rtype, srounds)
            if v is None: continue
            for name, val in zip(NAMES[rtype], v):
                allrows.append(dict(pid=pid, session=session, type=rtype, split=sname, n_decisions=n, param=name, value=float(val)))
    out = pd.DataFrame(allrows); out.to_csv(R / 'choice_weights.csv', index=False)
    # reliability
    rel = []
    for rtype in ['group', 'solo']:
        for a, b in [('A', 'B'), ('odd', 'even')]:
            w = out[(out.type == rtype)].pivot_table(index=['pid', 'param'], columns='split', values='value').reset_index()
            for name in NAMES[rtype]:
                ww = w[w.param == name].dropna(subset=[a, b])
                r = np.corrcoef(ww[a], ww[b])[0, 1]
                rel.append(dict(type=rtype, split=f'{a}_vs_{b}', param=name, r=r, spearman_brown=2 * r / (1 + r), n=len(ww), sd_all=out[(out.type == rtype) & (out.split == 'all') & (out.param == name)]['value'].std()))
    rel = pd.DataFrame(rel); rel.to_csv(R / 'choice_reliability.csv', index=False); print(rel.round(3).to_string())
    # validation vs the authors' Stan posterior means
    val = []
    for rtype, tag in POPKEY.items():
        st = pd.read_csv(R / f'stan_vID_{tag}.csv').set_index('id')
        w = out[(out.type == rtype) & (out.split == 'all')].pivot(index='pid', columns='param', values='value')
        for name in NAMES[rtype]:
            j = w.index.intersection(st.index)
            val.append(dict(type=rtype, param=name, r_vs_stan=np.corrcoef(w.loc[j, name], st.loc[j, name])[0, 1], n=len(j)))
    val = pd.DataFrame(val); val.to_csv(R / 'choice_vs_stan.csv', index=False); print(val.round(3).to_string())

if __name__ == '__main__':
    main()
