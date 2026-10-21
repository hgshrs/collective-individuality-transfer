"""Exploratory search over social rules that might reproduce the visibility-network structure (HANDOVER §12 item 5).

Setting: vectorised Chirkov environment (src/collective_id/env.py), every analysed group (103) simulated with its members'
individually fitted ABM parameters (I, S, k, x0, r_max from results/phase3/policy_params_all.csv; sigma 10; persistence =
group mean), 20 seeds per group x rule. Nothing is fitted to group outcomes. The baseline rule (R0) is the paper's ABM
decision rule; each alternative changes ONE element of the social part of the rule:
  R0 baseline        turn toward the mean bearing of visible others with p = sigmoid(k (q_mean - x0))
  R1 nearest         turn toward the NEAREST visible other (not the mean bearing)
  R2 best-informed   social quality from the best (closest-to-resource) visible other (q_best) instead of the mean
  R3 scan            while searching privately, with p 0.2 make a wide scanning turn (uniform 0-180 deg)
  R4 align           when using social information, align heading with the others' mean heading instead of their position
  R5 tracker-only    social information = visible others that are currently ON the resource (<= 40 m); quality fixed at 1
  R6 tracker-calm    while tracking privately, scale the rotation magnitude by 0.8 (trackers hold their heading)
  R7 R5 + R3
Evaluation: per rule, seed-mean group metrics vs observed (results/phase3/observed_group_metrics.csv): r pooled over groups
after z-scoring within condition x speed (group bootstrap 2,000) and raw levels (mean, SD over groups).
Outputs: results/phase3/social_rules_search.csv (r), social_rules_levels.csv, social_rules_runs.parquet
Usage: OMP_NUM_THREADS=1 uv run python scripts/84_social_rules_search.py [--seeds 20] [--procs 8]
"""
import argparse, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collective_id.paths import RESULTS
from collective_id.env import run, group_metrics, wrap, FOV_HALF, RADIUS
OUT = RESULTS / 'phase3'
RULES = ['R0_baseline', 'R1_nearest', 'R2_best', 'R3_scan', 'R4_align', 'R5_tracker_only', 'R6_tracker_calm', 'R7_tracker_scan']


def make_policy(rule, I, S, k, x0, rmax, pers, sigma=10.0, p_s=0.15, p_u=0.35, p_scan=0.2):
    """Per-agent parameter arrays of shape (N,) broadcast over simulations. Returns policy(env, o, t) -> rotation (S, N)."""
    I, S, k, x0, rmax = [np.asarray(v, float)[None, :] for v in (I, S, k, x0, rmax)]
    def policy(env, o, t):
        rng = env.rng; sh = (env.S, env.N)
        has, alpha, q = o['has'], o['alpha'], o['q_abm']
        if rule in ('R1_nearest', 'R5_tracker_only', 'R7_tracker_scan'):
            dx = env.x[:, None, :] - env.x[:, :, None]; dy = env.y[:, None, :] - env.y[:, :, None]
            pa = wrap(np.arctan2(dy, dx) - env.h[:, :, None]); vis = (np.abs(pa) < FOV_HALF) & ~np.eye(env.N, dtype=bool)[None]
            if rule == 'R1_nearest':
                dist = np.where(vis, np.sqrt(dx ** 2 + dy ** 2), np.inf); j = dist.argmin(2)
                alpha = np.where(has, np.take_along_axis(pa, j[..., None], 2)[..., 0], 0.0)
            else:
                on = (o['d'] <= RADIUS)[:, None, :]; vt = vis & on; n = vt.sum(2); has = n > 0
                alpha = np.where(has, (pa * vt).sum(2) / np.maximum(n, 1), 0.0); q = np.where(has, 1.0, 0.0)
        elif rule == 'R2_best':
            q = o['q_best']
        elif rule == 'R4_align':
            alpha = -o['mov']
        p_soc = np.where(has, 1 / (1 + np.exp(-k * (q - x0))), 0.0)
        use_soc = rng.random(sh) < p_soc
        mag = np.where(use_soc, np.abs(alpha), 0.0); dire = np.where(use_soc, np.sign(alpha), env.dir)
        priv_t = ~use_soc & o['tracking']
        mag = np.where(priv_t, np.deg2rad(I + S * o['det']), mag)
        if rule == 'R6_tracker_calm': mag = np.where(priv_t, 0.8 * mag, mag)
        thr = np.where(o['det'] > o['det_lag'], p_u, p_s); flip = priv_t & (rng.random(sh) < thr); dire = np.where(flip, -dire, dire)
        priv_s = ~use_soc & ~o['tracking']
        mag = np.where(priv_s, rng.uniform(0, 1, sh) * np.deg2rad(rmax), mag); dire = np.where(priv_s, rng.choice([-1.0, 1.0], size=sh), dire)
        if rule in ('R3_scan', 'R7_tracker_scan'):
            scan = priv_s & (rng.random(sh) < p_scan); mag = np.where(scan, rng.uniform(0, np.pi, sh), mag)
        mag = np.where(rng.random(sh) < pers, 0.0, mag)
        mag = mag + rng.normal(0, np.deg2rad(sigma), sh)
        env.dir = dire
        return mag * dire
    return policy


def run_group(job):
    rule, g, speed, P, pers, seeds, seed = job
    pol = make_policy(rule, P[:, 0], P[:, 1] * (1 + speed), P[:, 2], P[:, 3], P[:, 4], pers)
    res = run(seeds, P.shape[0], speed, pol, T=900, seed=seed, record=True)
    m = group_metrics(res); rows = []
    for s in range(seeds):
        rows.append(dict(rule=rule, group=g, seed=s, **{k: float(v[s]) for k, v in m.items()}))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--procs', type=int, default=8); a = ap.parse_args()
    par = pd.read_csv(OUT / 'policy_params_all.csv'); grp = par[par.cond != 'A']
    O = pd.read_csv(OUT / 'observed_group_metrics.csv').set_index('group')
    jobs = []
    for gi, (g, mem) in enumerate(grp.groupby('group')):
        speed = 0.4 if mem.speed.iloc[0] == 'slow' else 0.8
        P = mem[['I', 'S', 'k', 'x0', 'r_max']].to_numpy(float); pers = float(mem.persistence.mean())
        for ri, rule in enumerate(RULES): jobs.append((rule, g, speed, P, pers, a.seeds, 40_000_000 + 1000 * gi + ri))
    print(len(jobs), 'group x rule jobs', flush=True); t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(run_group, jobs, chunksize=4)):
            rows.extend(r)
            if i % 100 == 0: print(i, f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df.to_parquet(OUT / 'social_rules_runs.parquet', index=False)
    # ---- evaluation ----
    METRICS = ['score', 'tracking_time', 'pcr', 'med_dist', 'in_deg', 'out_deg', 'in_out_corr', 'disc_gain']
    pred = df.groupby(['rule', 'group'])[METRICS].mean()
    groups = sorted(set(O.index) & set(pred.index.get_level_values(1))); O = O.loc[groups]; cell = O.cell
    rng = np.random.default_rng(0)
    def zcell(s):
        s = pd.Series(s, index=groups, dtype=float); return s.groupby(cell).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))
    def r_of(p, o):
        m = p.notna() & o.notna(); return float(np.corrcoef(p[m], o[m])[0, 1]) if m.sum() > 3 and p[m].std() > 0 else np.nan
    rres, lev = [], []
    for rule in RULES:
        for met in METRICS:
            p = pred.loc[rule][met].reindex(groups); o = O[met]; pz, oz = zcell(p), zcell(o)
            bs = [r_of(pz.iloc[i].reset_index(drop=True), oz.iloc[i].reset_index(drop=True)) for i in (rng.integers(0, len(groups), len(groups)) for _ in range(2000))]
            rres.append(dict(rule=rule, metric=met, r_within_cell=r_of(pz, oz), ci_lo=float(np.nanquantile(bs, .025)), ci_hi=float(np.nanquantile(bs, .975)), r_raw=r_of(p, o), n=int((p.notna() & o.notna()).sum())))
            lev.append(dict(rule=rule, metric=met, sim_mean=float(p.mean()), sim_sd=float(p.std()), obs_mean=float(o.mean()), obs_sd=float(o.std())))
    RR = pd.DataFrame(rres); RR.to_csv(OUT / 'social_rules_search.csv', index=False); L = pd.DataFrame(lev); L.to_csv(OUT / 'social_rules_levels.csv', index=False)
    pd.set_option('display.width', 250); print(RR.pivot(index='rule', columns='metric', values='r_within_cell').reindex(RULES).round(3).to_string())
    print(L[L.metric.isin(['in_deg', 'in_out_corr', 'disc_gain', 'tracking_time'])].pivot(index='rule', columns='metric', values='sim_mean').reindex(RULES).round(3).to_string())
    print('observed means:', L[L.rule == 'R0_baseline'].set_index('metric').obs_mean.round(3).to_dict())
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
