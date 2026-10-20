"""Phase 3 (reduced D-1): per-participant maximum-a-posteriori fit of the paper's ABM policy to the participant's own
1 s rotation data. The policy (abm.py, SearcherAgent): tracking  -> with prob p_soc turn by |theta_pos|, else
|I + S*det| ; searching -> with prob p_soc turn by |theta_pos|, else Uniform(0, r_max); p_soc = sigmoid(k (q - x0)) if
somebody is in the FOV else 0; Gaussian rotation noise sigma (ABM default 10 deg). Likelihood = folded-normal mixture per
second (degrees per second). Parameters (I, S, log k, x0, log r_max, log sigma); Gaussian prior centred on the pooled
fit of the participant's condition x speed cell estimated on the OTHER folds (leave-one-group-out respected).
Nothing here uses group outcomes (AGENT_SPEC §6.2).

Also: pooled per-cell fits (controls), split-half (first/second 7.5 min) fits for reliability, and the §6.3-2 check:
second-half predictive NLL of the ABM policy fitted on the first half vs the paper's linear Exponential model with
first-half individual random effects (EB) on the same tracking seconds.
Usage: python scripts/40_phase3_fit_policy.py [--procs 20]
Outputs: results/phase3/policy_params_{all,first,second}.csv, policy_pooled_by_cell_fold.csv, policy_reliability.csv,
         policy_vs_linear_predictive.csv
"""
import argparse, json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr
from scipy.stats import norm
from collective_id.paths import PROCESSED, RESULTS, RUNS
from collective_id.split import load_split

OUT = RESULTS / "phase3"; OUT.mkdir(parents=True, exist_ok=True)
FOV_HALF = 45.746; LOG150 = np.log(150.0)
PRIOR_SD = np.array([20.0, 20.0, 1.0, 0.5, 0.5, 0.3, 1.0])      # I, S, log k, x0, log r_max, log sigma, logit persistence
POOL_PRIOR_SD = PRIOR_SD * 3                                     # weak prior toward the paper defaults for the pooled fits
BOUNDS = [(0, 150), (-100, 200), (np.log(0.05), np.log(20)), (-2, 2), (np.log(5), np.log(180)), (np.log(5), np.log(80)), (-5, 3)]   # I >= 0: the folded likelihood is invariant to (I, S) -> (-I, -S)
DEFAULT = np.array([50.0, 35.0, 0.0, 0.0, np.log(90.0), np.log(10.0), -1.4])   # persistence 0.2 (ABM parameter: P(rotation magnitude = 0))


def prep(d):
    """Arrays for the likelihood from a participant's 1 Hz rows (state tracking with no_move<3, or searching)."""
    d = d[((d.state == "tracking") & (d.no_move < 3)) | (d.state == "searching")]
    y = np.clip((d.rot_abs.to_numpy() - 0.01) * 150.0, 0, 150)
    trk = (d.state == "tracking").to_numpy()
    vis = (d.n_vis.to_numpy() > 0)
    theta = np.where(vis, (d.soc_pos.to_numpy() / 2 + 0.5) * FOV_HALF, 0.0)
    q = np.where(trk, -np.clip(d.soc_qual_raw.to_numpy() / 40.0, -1, 1), d.soc_qual_search.to_numpy()); q = np.where(vis, q, 0.0)
    det = np.nan_to_num(d.detector.to_numpy())
    return dict(y=y, trk=trk, vis=vis, theta=theta, q=q, det=det)


def folded_norm_logpdf(y, mu, s):
    return np.logaddexp(norm.logpdf(y, mu, s), norm.logpdf(-y, mu, s))


def unif_conv_norm_logpdf(y, rmax, s):
    """log density of |U(0, rmax) + N(0, s^2)| at y >= 0."""
    def one(v):
        return np.log(np.clip(norm.cdf(v / s) - norm.cdf((v - rmax) / s), 1e-300, None)) - np.log(rmax)
    return np.logaddexp(one(y), one(-y))


def _ll_elementwise(th, A):
    I, S, lk, x0, lr, ls, lpers = th; k, rmax, s, pers = np.exp(lk), np.exp(lr), np.exp(ls), expit(lpers)
    y, trk, vis, theta, q, det = A["y"], A["trk"], A["vis"], A["theta"], A["q"], A["det"]
    p = np.where(vis, expit(k * (q - x0)), 0.0); p = np.clip(p, 1e-9, 1 - 1e-9)
    lsoc = folded_norm_logpdf(y, theta, s)
    lpriv = np.where(trk, folded_norm_logpdf(y, I + S * det, s), unif_conv_norm_logpdf(y, rmax, s))
    lmove = np.logaddexp(np.log(p) + lsoc, np.log1p(-p) + lpriv)
    # Zero-rotation seconds (|rot| < 1 deg/s; exact zeros in the data) are a point mass: persistence (ABM: magnitude set
    # to 0) plus the probability that a movement rotation falls below 1 deg/s. Treating them as a density would drive
    # sigma -> 0 for participants with many zeros.
    zero = y < 1.0
    m_soc = norm.cdf((1 - theta) / s) - norm.cdf((-1 - theta) / s); m_priv_t = norm.cdf((1 - (I + S * det)) / s) - norm.cdf((-1 - (I + S * det)) / s)
    m_priv_s = np.exp(unif_conv_norm_logpdf(np.full_like(y, 0.5), rmax, s))          # folded density at 0.5 x width 1
    m_move = p * m_soc + (1 - p) * np.where(trk, m_priv_t, m_priv_s)
    ll_zero = np.log(pers + (1 - pers) * np.clip(m_move, 1e-12, 1.0))
    ll = np.where(zero, ll_zero, np.log1p(-pers) + lmove)
    return ll


def nll(th, A, prior_mu=None, prior_sd=None):
    out = -_ll_elementwise(th, A).sum()
    if prior_mu is not None: out += 0.5 * (((th - prior_mu) / prior_sd) ** 2).sum()
    return out


ALT = DEFAULT.copy(); ALT[2] = np.log(5.0); ALT[3] = 0.5; ALT[0] = 25.0

def fit(A, prior_mu=None, prior_sd=None, x0=None):
    best = None
    for start in ([DEFAULT, ALT] if x0 is None else [x0, DEFAULT, ALT]):
        r = minimize(nll, start, args=(A, prior_mu, prior_sd), method="L-BFGS-B", bounds=BOUNDS)
        if best is None or r.fun < best.fun: best = r
    return best.x, float(best.fun), int(len(A["y"]))


def fit_participant(args):
    pid, A, prior_mu, tag = args
    th, f, n = fit(A, prior_mu, PRIOR_SD, x0=prior_mu)
    return dict(participant=pid, tag=tag, I=th[0], S=th[1], k=np.exp(th[2]), x0=th[3], r_max=np.exp(th[4]), sigma=np.exp(th[5]), persistence=float(expit(th[6])), nll=f, n_sec=n, n_track=int(A["trk"].sum()), n_vis_frac=float(A["vis"].mean()))


def fit_pooled(args):
    cell, fold, A = args
    th, f, n = fit(A, DEFAULT, POOL_PRIOR_SD)
    return dict(cell=cell, fold=fold, I=th[0], S=th[1], k=np.exp(th[2]), x0=th[3], r_max=np.exp(th[4]), sigma=np.exp(th[5]), persistence=float(expit(th[6])), nll=f, n_sec=n)


YMIN = 0.01 + 1.0 / 150.0   # rot_abs of 1 deg/s

def linear_eb_nll(d, fixef, ranef, pid, cell_prefix):
    """Second-half tracking NLL (nats/s on the rot_abs scale) of the paper's linear truncated-Exponential model,
    on moving seconds (|rot| >= 1 deg/s) and renormalised to that region."""
    d = d[(d.state == "tracking") & (d.no_move < 3) & (d.rot_abs >= YMIN)]
    with_soc = (d.n_vis > 0).astype(float)
    X = pd.DataFrame({"Intercept": 1.0, "Detector": d.detector.fillna(0), "DistFromRes": d.dist_track, "withSocInfo1:SocPosInFOV": with_soc * d.soc_pos, "withSocInfo1:SocMovDir": with_soc * d.soc_mov,
                      "withSocInfo1:SocPosInFOV:SocQual": with_soc * d.soc_qual_track * d.soc_pos, "withSocInfo1:SocMovDir:SocQual": with_soc * d.soc_qual_track * d.soc_mov})
    beta = np.array([fixef.get(f"{cell_prefix}:{c}", 0.0) for c in X.columns]); u = np.array([ranef.get((pid, c), 0.0) for c in X.columns])
    eta = X.to_numpy() @ (beta + u); rate = np.clip(np.exp(-eta), 1e-4, 1e4); y = d.rot_abs.to_numpy()
    lp = np.log(rate) - rate * y - np.log1p(-np.exp(-rate * 1.02))
    lp -= np.log(np.clip((np.exp(-rate * YMIN) - np.exp(-rate * 1.02)) / (1 - np.exp(-rate * 1.02)), 1e-12, 1.0))   # P(rot_abs >= YMIN)
    return -lp.mean(), len(y)


def policy_nll_rotabs(th, A_trk):
    """ABM-policy NLL (nats/s) on the rot_abs scale, moving seconds only (|rot| >= 1 deg/s), renormalised by
    P(moving) so that it is comparable with the linear model evaluated on the same seconds."""
    keep = A_trk["y"] >= 1.0; A = {k: v[keep] for k, v in A_trk.items()}
    Az = dict(A); Az["y"] = np.zeros_like(A["y"])
    pz = np.exp(_ll_elementwise(th, Az))                                   # P(zero_t) per second
    lp = _ll_elementwise(th, A) - np.log(np.clip(1 - pz, 1e-12, 1))
    return float(-lp.mean() - LOG150)                                       # p_rotabs = p_deg * 150


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=20); a = ap.parse_args()
    f = pd.read_parquet(PROCESSED / "features_1hz.parquet")
    meta = f.groupby("participant").agg(cond=("cond", "first"), speed=("speed", "first"), group=("group", "first"))
    meta["cell"] = meta.cond + "_" + meta.speed
    sp = load_split(); fold = pd.Series(sp["fold_of_participant"]); meta["fold"] = fold.reindex(meta.index)
    parts = list(meta.index)
    byp = {p: g for p, g in f.groupby("participant")}
    halves = {"all": (None, None), "first": (None, 450), "second": (450, None)}
    A = {}
    for tag, (t0, t1) in halves.items():
        for p in parts:
            g = byp[p]
            if t0 is not None: g = g[g.t >= t0]
            if t1 is not None: g = g[g.t < t1]
            A[(tag, p)] = prep(g)
    # ---- pooled per cell x fold (training folds only), on the full session ----
    jobs = []
    for cell, m in meta.groupby("cell"):
        for fo in range(sp["n_folds"]):
            tr = [p for p in m.index if m.loc[p, "fold"] != fo]
            jobs.append((cell, fo, {k: np.concatenate([A[("all", p)][k] for p in tr]) for k in A[("all", tr[0])]}))
    print(f"pooled fits: {len(jobs)}", flush=True)
    with ProcessPoolExecutor(a.procs) as ex: pooled = pd.DataFrame(list(ex.map(fit_pooled, jobs)))
    pooled.to_csv(OUT / "policy_pooled_by_cell_fold.csv", index=False)
    pooled_i = pooled.set_index(["cell", "fold"])
    def prior_for(p):
        r = pooled_i.loc[(meta.loc[p, "cell"], int(meta.loc[p, "fold"]))]
        return np.array([r.I, r.S, np.log(r.k), r.x0, np.log(r.r_max), np.log(r.sigma), np.log(r.persistence / (1 - r.persistence))])
    # ---- per-participant MAP fits (all / first / second) ----
    jobs = [(p, A[(tag, p)], prior_for(p), tag) for tag in halves for p in parts if len(A[(tag, p)]["y"]) >= 20]
    print(f"participant fits: {len(jobs)}", flush=True)
    with ProcessPoolExecutor(a.procs) as ex: res = pd.DataFrame(list(ex.map(fit_participant, jobs, chunksize=16)))
    res = res.merge(meta.reset_index()[["participant", "cond", "speed", "group", "cell", "fold"]], on="participant")
    for tag in halves: res[res.tag == tag].drop(columns="tag").to_csv(OUT / f"policy_params_{tag}.csv", index=False)
    # ---- split-half reliability ----
    P1 = res[res.tag == "first"].set_index("participant"); P2 = res[res.tag == "second"].set_index("participant"); common = P1.index.intersection(P2.index)
    rel = []
    for c in ["I", "S", "k", "x0", "r_max", "sigma", "persistence"]:
        a1, a2 = P1.loc[common, c], P2.loc[common, c]
        if c in ("k", "r_max", "sigma"): a1, a2 = np.log(a1), np.log(a2)
        rel.append(dict(param=c, r_first_second=float(np.corrcoef(a1, a2)[0, 1]), spearman=float(pd.Series(a1.values).corr(pd.Series(a2.values), method="spearman")), n=len(common), sd_all=float(res[res.tag == "all"][c].pipe(lambda s: np.log(s) if c in ("k", "r_max", "sigma") else s).std())))
    rel = pd.DataFrame(rel); rel.to_csv(OUT / "policy_reliability.csv", index=False)
    # ---- §6.3-2: second-half predictive NLL, ABM policy (first-half fit) vs linear EB model (first-half fit) ----
    fx = pd.read_csv(RUNS / "eb" / "tracking_first_fixef.csv"); fixef = dict(zip(fx.coef, fx.Estimate))
    rf = pd.read_csv(RUNS / "eb" / "tracking_first_ranef.csv"); ranef = {(r.participant, r.coef): r.estimate for r in rf.itertuples()}
    rows = []
    for p in common:
        g2 = byp[p]; g2 = g2[g2.t >= 450]
        A2 = prep(g2[g2.state == "tracking"])
        if len(A2["y"]) < 20: continue
        th = P1.loc[p]; thv = np.array([th.I, th.S, np.log(th.k), th.x0, np.log(th.r_max), np.log(th.sigma), np.log(th.persistence / (1 - th.persistence))])
        n_pol = policy_nll_rotabs(thv, A2)
        n_pol_pooled = policy_nll_rotabs(prior_for(p), A2)
        n_lin, n = linear_eb_nll(g2, fixef, ranef, p, f"SignalingType{meta.loc[p, 'cond']}:ResourceSpeed{meta.loc[p, 'speed']}")
        n_lin_pop, _ = linear_eb_nll(g2, fixef, {}, p, f"SignalingType{meta.loc[p, 'cond']}:ResourceSpeed{meta.loc[p, 'speed']}")
        rows.append(dict(participant=p, cell=meta.loc[p, "cell"], n_sec=n, nll_policy_indiv=n_pol, nll_policy_pooled=n_pol_pooled, nll_linear_indiv=n_lin, nll_linear_pop=n_lin_pop))
    pv = pd.DataFrame(rows); pv.to_csv(OUT / "policy_vs_linear_predictive.csv", index=False)
    summ = dict(n=len(pv), **{c: float(pv[c].mean()) for c in ["nll_policy_indiv", "nll_policy_pooled", "nll_linear_indiv", "nll_linear_pop"]},
                gain_policy_indiv_vs_pooled=float((pv.nll_policy_pooled - pv.nll_policy_indiv).mean()), gain_linear_indiv_vs_pop=float((pv.nll_linear_pop - pv.nll_linear_indiv).mean()),
                frac_participants_policy_better_than_linear=float((pv.nll_policy_indiv < pv.nll_linear_indiv).mean()))
    json.dump(summ, open(OUT / "policy_fit_summary.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(res[res.tag == "all"].groupby("cell")[["I", "S", "k", "x0", "r_max", "sigma", "persistence"]].median().round(2)); print(pooled.groupby("cell")[["I", "S", "k", "x0", "r_max", "sigma", "persistence"]].mean().round(2)); print(rel.round(3)); print(summ)


if __name__ == "__main__":
    main()
