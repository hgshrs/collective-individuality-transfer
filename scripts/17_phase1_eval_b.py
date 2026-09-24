"""Decoder (b) evaluation: held-out tracking-rotation NLL of the z-conditioned hypernetwork vs. baselines:
(i) condition x speed population weights (fitted on training participants' tracking data by max likelihood),
(ii) oracle: each participant's own teacher random effects (fixed + random from brms), (iii) participant's own ML fit.
Usage: --exp phase1_b_z4"""
import argparse, glob, json, numpy as np, pandas as pd, torch
from collective_id.paths import RUNS, RESULTS
from collective_id.phase1 import Phase1Data
from collective_id.data import TRK_FEATS, pad_batch
from collective_id.models import exp_nll
ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="phase1_b_z4"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args()
cfg = json.load(open(RUNS / a.exp / "config.json")); D = Phase1Data(cfg); dev = a.device
def fit_weights(parts, iters=300):
    """ML fit of a single weight vector (or per-participant vectors) for tracking rotation."""
    F, M = pad_batch([D.trk[p][0] for p in parts], len(TRK_FEATS)); Y, _ = pad_batch([D.trk[p][1][:, None] for p in parts], 1)
    F, M, Y = F.to(dev), M.to(dev), Y[..., 0].to(dev)
    w = torch.zeros(1, len(TRK_FEATS) + 1, device=dev, requires_grad=True); w.data[0, 0] = np.log(0.25)
    opt = torch.optim.Adam([w], 0.05)
    for _ in range(iters):
        loss = exp_nll(w.expand(len(parts), -1), F, Y, M); opt.zero_grad(); loss.backward(); opt.step()
    return w.detach()
def nll_of(w, parts):
    F, M = pad_batch([D.trk[p][0] for p in parts], len(TRK_FEATS)); Y, _ = pad_batch([D.trk[p][1][:, None] for p in parts], 1)
    F, M, Y = F.to(dev), M.to(dev), Y[..., 0].to(dev)
    eta = w[:, :1] + torch.einsum("bk,btk->bt", w[:, 1:], F); rate = torch.exp(-eta).clamp(1e-4, 1e4)
    lp = torch.log(rate) - rate * Y - torch.log1p(-torch.exp(-rate * 1.02)); return (-(lp * M).sum(1) / M.sum(1).clamp(min=1)).cpu().numpy()
rows = []
for f in sorted(glob.glob(str(RUNS / a.exp / "seed*" / "fold*" / "pred_test.csv"))):
    seed = int(f.split("seed")[1].split("/")[0]); fold = int(f.split("fold")[1].split("/")[0])
    pred = pd.read_csv(f, index_col=0); test = [p for p in pred.index if p in D.trk]; train = [p for p in D.participants if D.fold[p] != fold and p in D.trk]
    # model
    w_model = torch.tensor(pred.loc[test].to_numpy(np.float32), device=dev); nll_model = nll_of(w_model, test)
    # cond x speed population baseline (from training participants)
    nll_pop = np.zeros(len(test)); groups = D.meta.loc[test].groupby(["cond", "speed"]).groups
    for (c, s), idx in groups.items():
        tr = [p for p in train if D.meta.loc[p, "cond"] == c and D.meta.loc[p, "speed"] == s]; w = fit_weights(tr)
        ii = [test.index(p) for p in idx]; nll_pop[ii] = nll_of(w.expand(len(ii), -1), [test[i] for i in ii])
    # oracle: own ML fit (in-sample) — lower bound on achievable NLL
    nll_own = np.array([nll_of(fit_weights([p], 200), [p])[0] for p in test])
    for i, p in enumerate(test): rows.append(dict(seed=seed, fold=fold, participant=p, nll_model=nll_model[i], nll_pop=nll_pop[i], nll_own=nll_own[i]))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "phase1" / f"nll_b_{a.exp}.csv", index=False)
R["gain_vs_pop"] = R.nll_pop - R.nll_model; R["frac_of_oracle_gain"] = (R.nll_pop - R.nll_model) / (R.nll_pop - R.nll_own).clip(lower=1e-6)
summ = R.groupby("seed").agg(nll_model=("nll_model", "mean"), nll_pop=("nll_pop", "mean"), nll_own=("nll_own", "mean"), gain_vs_pop=("gain_vs_pop", "mean"), frac_participants_improved=("gain_vs_pop", lambda x: (x > 0).mean()))
import statsmodels.formula.api as smf
mm = smf.mixedlm("gain_vs_pop ~ 1", R, groups=R["participant"]).fit(reml=False)
summ.to_csv(RESULTS / "phase1" / f"nll_b_summary_{a.exp}.csv"); json.dump(dict(gain_vs_pop=mm.params["Intercept"], p=mm.pvalues["Intercept"], median_frac_of_oracle_gain=float(R.frac_of_oracle_gain.median())), open(RESULTS / "phase1" / f"nll_b_test_{a.exp}.json", "w"), indent=1)
print(summ.round(4)); print("mixed model gain vs population:", mm.params["Intercept"], "p=", mm.pvalues["Intercept"])
