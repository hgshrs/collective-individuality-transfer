"""Phase 3 (reduced D-1): leave-one-group-out simulation of every analysed group with the paper's ABM (external/,
unmodified) and per-member policy parameters fitted from individual rotation data only (scripts/40).

Per group (n members = 4 or 5, speed 0.4 / 0.8):
  main            members' own MAP parameters (I, S, k, x0, r_max; sigma = group mean, a model-level parameter in the ABM)
  ctrl1_paper     paper ABM defaults (I 50, S 35, r_max 90, x0 0) with k = pooled fit of the cell on the training folds
  ctrl1b_pooled   all parameters = pooled fit of the cell on the training folds (homogeneous)
  ctrl2_avg       mean MAP parameters of the TRAINING participants of the cell (identical agents; spec control 2)
  ctrl2b_gmean    mean MAP parameters of the group's own members (isolates the composition effect)
  ctrl3_nominal   each member simulated ALONE (N=2) with their own parameters; group prediction = mean over members
  ctrl4_shuffle   members replaced by random same-cell participants from OTHER groups (n_shuffle draws)
  ctrl4b_mps      mean-preserving shuffle: among 20x candidate draws, keep the n_shuffle whose standardised parameter
                  means are closest to the real group's means
Detector slope conversion: data detector ~ dd/7, ABM det = dd/(7(1+v)) -> S_abm = S_data * (1 + v).
Collective metrics per run (computed from the trajectories): mean score, tracking time, pcr, in/out-degree (90 deg FOV)
means and member-level in-out correlation, median distance from the resource, discovery events and discoverer in-degree
gain (10 s after - 10 s before). Nothing is fitted to group outcomes.
Usage: python scripts/41_phase3_simulate.py [--seeds 100] [--seeds_shuffle 10] [--n_shuffle 50] [--procs 20] [--groups all]
Output: results/phase3/sim_runs.parquet (one row per run), sim_group_params.csv
"""
import argparse, sys, time, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from collective_id.paths import RESULTS, EXTERNAL, PROCESSED
from collective_id.split import load_split
sys.path.insert(0, str(EXTERNAL))
from agent_based_model.abm import SearcherResourceModel

OUT = RESULTS / "phase3"; OUT.mkdir(parents=True, exist_ok=True)
BASE = dict(width=400, height=400, resource_movement='random', p_s=0.15, p_u=0.35, p_perceive_social_info_quality=1,
            social_alignment='position', social_info_quality_func='sigmoid')
PCOLS = ["I", "S", "k", "x0", "r_max", "sigma"]
FOV_HALF = np.pi / 4


def metrics_from_run(m, n):
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    T = int(df.Step.max()); steps = np.arange(1, T + 1)
    res = df[df.AgentID == 0].set_index("Step").loc[steps]
    ag = df[df.AgentID != 0]
    ids = sorted(ag.AgentID.unique())
    X = np.stack([ag[ag.AgentID == i].set_index("Step").loc[steps].x.to_numpy(float) for i in ids], 1)
    Y = np.stack([ag[ag.AgentID == i].set_index("Step").loc[steps].y.to_numpy(float) for i in ids], 1)
    R = np.stack([ag[ag.AgentID == i].set_index("Step").loc[steps].rotation.to_numpy(float) for i in ids], 1)
    sc = np.array([ag[ag.AgentID == i].score.iloc[-1] for i in ids], float); ton = np.array([ag[ag.AgentID == i].n_times_on_res.iloc[-1] for i in ids], float)
    D = np.sqrt((X - res.x.to_numpy()[:, None]) ** 2 + (Y - res.y.to_numpy()[:, None]) ** 2)   # (T, n) distance to resource
    out = dict(score=float(sc.mean()), tracking_time=float(ton.mean() / T), pcr=float(np.nanmean(sc / np.where(ton > 0, ton, np.nan))), med_dist=float(np.median(np.median(D, 0))))
    if n > 1:
        # visibility: i sees j if |angle(i->j) - heading_i| < 45 deg
        dx = X[:, None, :] - X[:, :, None]; dy = Y[:, None, :] - Y[:, :, None]       # [t, i, j] = pos_j - pos_i
        ang = np.arctan2(dy, dx) - R[:, :, None]; ang = (ang + np.pi) % (2 * np.pi) - np.pi
        see = np.abs(ang) < FOV_HALF; np.einsum("tii->ti", see)[:] = False
        outd = see.sum(2).astype(float); ind = see.sum(1).astype(float)           # (T, n)
        mi, mo = ind.mean(0), outd.mean(0)
        out.update(in_deg=float(mi.mean()), out_deg=float(mo.mean()), in_out_corr=float(np.corrcoef(mi, mo)[0, 1]) if mi.std() > 0 and mo.std() > 0 else np.nan)
        # discovery events: all searching (>40 m) for >= 2 s, then exactly one enters and stays >= 2 s
        on = D <= 40; gains = []
        allsearch = ~on.any(1)
        for t in range(2, T - 11):
            if allsearch[t - 2] and allsearch[t - 1] and on[t].sum() == 1:
                i = int(np.where(on[t])[0][0])
                if on[t + 1, i]:
                    pre = ind[max(0, t - 10):t, i].mean(); post = ind[t + 1:t + 11, i].mean(); gains.append(post - pre)
        out.update(disc_gain=float(np.mean(gains)) if gains else np.nan, n_disc=len(gains))
    return out


def run_one(job):
    tag, group, variant, draw, speed, P, (sigma, pers), seed = job     # P: (n, 5) array I, S_data, k, x0, r_max; sigma & persistence are model-level in the ABM
    np.random.seed(seed)
    n = P.shape[0]
    m = SearcherResourceModel(N=n + 1, resource_speed=speed, noize_degree=float(sigma), persistence=float(pers),
                              rotation_tracking_intercept=[float(v) for v in P[:, 0]], detector_sensitivity_slope=[float(v * (1 + speed)) for v in P[:, 1]],
                              social_info_quality_sensitivity_slope=[float(v) for v in P[:, 2]], social_info_quality_sensitivity_intercept=[float(v) for v in P[:, 3]],
                              search_max_rotation=[float(v) for v in P[:, 4]], **BASE)
    for _ in range(900): m.step()
    out = metrics_from_run(m, n); out.update(tag=tag, group=group, variant=variant, draw=draw, speed=speed, n=n, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=100); ap.add_argument("--seeds_shuffle", type=int, default=10); ap.add_argument("--n_shuffle", type=int, default=50)
    ap.add_argument("--procs", type=int, default=20); ap.add_argument("--groups", default="all"); ap.add_argument("--sigma", default="fixed10", help="fixed10 = paper noise 10 deg for all variants; fitted = group-mean fitted sigma"); ap.add_argument("--tag", default=""); ap.add_argument("--only", default="", help="comma-separated variants to run (default all)"); a = ap.parse_args()
    par = pd.read_csv(OUT / "policy_params_all.csv").set_index("participant"); pooled = pd.read_csv(OUT / "policy_pooled_by_cell_fold.csv").set_index(["cell", "fold"])
    sp = load_split(); fold_g = sp["fold_of_group"]
    grp = par[par.cond != "A"]
    groups = sorted(grp.group.unique()) if a.groups == "all" else a.groups.split(",")
    rng = np.random.default_rng(0)
    P5 = ["I", "S", "k", "x0", "r_max"]
    jobs, gpar = [], []
    for gi, g in enumerate(groups):
        mem = grp[grp.group == g]; cell = mem.cell.iloc[0]; speed = 0.4 if mem.speed.iloc[0] == "slow" else 0.8; fo = fold_g[g]; n = len(mem)
        Pm = mem[P5].to_numpy(float); sig = (float(mem.sigma.mean()) if a.sigma == "fitted" else 10.0, float(mem.persistence.mean()))
        train = grp[(grp.cell == cell) & (grp.fold != fo)]                 # training participants of the cell (other folds)
        po = pooled.loc[(cell, fo)]
        sig_pool = (float(po.sigma) if a.sigma == "fitted" else 10.0, float(po.persistence))
        variants = {
            "main": (Pm, sig),
            "ctrl1_paper": (np.tile([50.0, 35.0 / (1 + speed), po.k, 0.0, 90.0], (n, 1)), (10.0, 0.0)),
            "ctrl1b_pooled": (np.tile(po[P5].to_numpy(float), (n, 1)), sig_pool),
            "ctrl2_avg": (np.tile(train[P5].mean().to_numpy(float), (n, 1)), (float(train.sigma.mean()) if a.sigma == "fitted" else 10.0, float(train.persistence.mean()))),
            "ctrl2b_gmean": (np.tile(Pm.mean(0), (n, 1)), sig),
        }
        gpar.append(dict(group=g, cell=cell, n=n, fold=fo, **{f"{c}_mean": Pm[:, i].mean() for i, c in enumerate(P5)}, **{f"{c}_sd": Pm[:, i].std() for i, c in enumerate(P5)}, sigma_mean=sig[0], persistence_mean=sig[1]))
        for v, (P, s) in variants.items():
            for sd in range(a.seeds): jobs.append((f"{g}|{v}", g, v, 0, speed, P, s, 10_000_000 + 10_000 * gi + sd))
        # nominal: each member alone
        for mi in range(n):
            for sd in range(a.seeds): jobs.append((f"{g}|ctrl3_nominal", g, "ctrl3_nominal", mi, speed, Pm[mi:mi + 1], sig, 20_000_000 + 10_000 * gi + 100 * mi + sd))
        # shuffles: same-cell participants from other groups (any fold; the replaced members' own fits do not involve this group)
        pool = train if len(train) >= 3 * n else grp[(grp.cell == cell) & (grp.group != g)]
        cand = [pool.sample(n, random_state=int(rng.integers(1e9))) for _ in range(20 * a.n_shuffle)]
        PM = P5 + ["persistence"]                                           # match on all transferred parameters incl. persistence (model-level, group mean)
        mu, sd_ = grp[PM].mean(), grp[PM].std() + 1e-9
        target = ((mem[PM].mean() - mu) / sd_).to_numpy()
        dist = np.array([np.linalg.norm(((c[PM].mean() - mu) / sd_).to_numpy() - target) for c in cand])
        order = np.argsort(dist); mps = [cand[i] for i in order[:a.n_shuffle]]; rnd = cand[:a.n_shuffle]
        for v, lst in (("ctrl4_shuffle", rnd), ("ctrl4b_mps", mps)):
            for di, c in enumerate(lst):
                for sd in range(a.seeds_shuffle): jobs.append((f"{g}|{v}", g, v, di, speed, c[P5].to_numpy(float), (float(c.sigma.mean()) if a.sigma == "fitted" else 10.0, float(c.persistence.mean())), 30_000_000 + 100_000 * gi + 100 * di + sd))
    if a.only: jobs = [j for j in jobs if j[2] in a.only.split(",")]
    pd.DataFrame(gpar).to_csv(OUT / f"sim_group_params{a.tag}.csv", index=False)
    print(f"{len(groups)} groups, {len(jobs)} runs", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=16)):
            rows.append(r)
            if i % 5000 == 0: print(f"  {i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(OUT / f"sim_runs{a.tag}.parquet", index=False)
    json.dump(dict(n_runs=len(df), wall_s=time.time() - t0, sigma=a.sigma, seeds=a.seeds, seeds_shuffle=a.seeds_shuffle, n_shuffle=a.n_shuffle, procs=a.procs), open(OUT / f"sim_budget{a.tag}.json", "w"), indent=1)
    print(f"done {len(df)} runs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
