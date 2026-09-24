"""Fast provisional teacher: empirical-Bayes / penalised-ML fit (GPU, PyTorch) of the paper's rotation-magnitude model
(same features, same fixed-effect structure incl. the constant(0) terms, same participant random-effect structure, same
truncated Exponential likelihood, same N(0,0.5)/N(0,1)/half-N(0,0.1) priors). Random-effect covariance is diagonal
(brms: full LKJ) — documented deviation. Posterior SD of random effects via Laplace (expected information + prior).
Outputs runs/eb/<tag>_{fixef,ranef,diag}.csv in the same format as scripts/r/fit_rotation_models.R.
Usage: python scripts/05_teacher_eb.py --model tracking|search --half all|first|second [--tag ...] [--iters 3000]"""
import argparse, time, numpy as np, pandas as pd, torch
from collective_id.paths import RUNS
from collective_id.data import load_features
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="tracking"); ap.add_argument("--half", default="all"); ap.add_argument("--tag", default=None)
ap.add_argument("--iters", type=int, default=4000); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--participants", default=None, help="CSV with a participant column: restrict the fit to these participants")
a = ap.parse_args(); tag = a.tag or f"{a.model}_{a.half}"; out = RUNS / "eb"; out.mkdir(parents=True, exist_ok=True); dev = a.device
d = load_features()
if a.participants: keep = set(pd.read_csv(a.participants).participant); d = d[d.participant.isin(keep)]
if a.half == "first": d = d[d.t < 450]
if a.half == "second": d = d[d.t >= 450]
if a.model == "tracking": d = d[(d.state == "tracking") & (d.no_move < 3)].copy(); q = d.soc_qual_track
else: d = d[d.state == "searching"].copy(); q = d.soc_qual_search
ws = d.with_soc.to_numpy().astype(np.float32)
# features exactly as the Rmd (SocPos/SocMov already 0 when no social info)
priv = {"Detector": d.detector.to_numpy(np.float32), "DistFromRes": (d.dist_track if a.model == "tracking" else d.dist_search).to_numpy(np.float32)} if a.model == "tracking" else {}
soc = {"withSocInfo1:SocPosInFOV": (ws * d.soc_pos).astype(np.float32), "withSocInfo1:SocMovDir": (ws * d.soc_mov).astype(np.float32),
       "withSocInfo1:SocPosInFOV:SocQual": (ws * q * d.soc_pos).astype(np.float32), "withSocInfo1:SocMovDir:SocQual": (ws * q * d.soc_mov).astype(np.float32)}
feat_names = list(priv) + list(soc); Fm = np.stack([priv[k] if k in priv else soc[k] for k in feat_names], 1)
re_names = ["Intercept"] + ([ "Detector"] if a.model == "tracking" else []) + list(soc)
re_idx = [feat_names.index(k) for k in re_names[1:]]
conds = ["A", "NP", "VP", "FP"]; speeds = ["fast", "slow"]
cell = (d.cond.map({c: i for i, c in enumerate(conds)}) * 2 + d.speed.map({s: i for i, s in enumerate(speeds)})).to_numpy()
parts = d.participant.astype("category"); pid = parts.cat.codes.to_numpy(); P = len(parts.cat.categories)
y = torch.tensor(d.rot_abs.to_numpy(np.float32), device=dev); F = torch.tensor(Fm, device=dev); cell_t = torch.tensor(cell.astype(np.int64), device=dev); pid_t = torch.tensor(pid.astype(np.int64), device=dev)
K = len(feat_names); R = len(re_names)
# fixed effects: intercept per cell (8) + slope per cell per feature (8 x K); mask of structurally-zero terms (paper's constant(0) priors)
mask = torch.ones(8, K, device=dev)
for ci, c in enumerate(conds):
    for si, s in enumerate(speeds):
        if c == "A":
            for k in feat_names:
                if "SocQual" in k: mask[ci * 2 + si, feat_names.index(k)] = 0
            if a.model == "tracking":
                if s == "fast": mask[ci * 2 + si, feat_names.index("withSocInfo1:SocPosInFOV")] = 0
                if s == "slow": mask[ci * 2 + si, feat_names.index("withSocInfo1:SocMovDir")] = 0
b0 = torch.full((8,), float(np.log(0.25)), device=dev, requires_grad=True); B = torch.zeros(8, K, device=dev, requires_grad=True)
U = torch.zeros(P, R, device=dev, requires_grad=True); log_tau = torch.full((R,), np.log(0.1), device=dev, requires_grad=True)
opt = torch.optim.Adam([b0, B, U, log_tau], a.lr); n = len(y); t0 = time.time()
# expected Fisher information per participant for the RE block (Exponential, log link: sum_t f f^T) — used for the
# Laplace approximation of the marginal likelihood (prevents the joint-MAP collapse of tau to 0) and for posterior SDs
Fr0 = torch.cat([torch.ones(n, 1, device=dev), F[:, re_idx]], 1)
INFO = torch.zeros(P, R, R, device=dev); INFO.index_add_(0, pid_t, Fr0[:, :, None] * Fr0[:, None, :])
def objective():
    Bm = B * mask; tau = torch.exp(log_tau)
    eta = b0[cell_t] + (F * Bm[cell_t]).sum(1) + U[pid_t, 0] + (F[:, re_idx] * U[pid_t, 1:]).sum(1)
    rate = torch.exp(-eta).clamp(1e-4, 1e4); ll = (torch.log(rate) - rate * y - torch.log1p(-torch.exp(-rate * 1.02))).sum()
    lp = -0.5 * ((U / tau) ** 2).sum() - P * log_tau.sum()                       # RE Gaussian with log-det
    lp -= 0.5 * torch.logdet(INFO + torch.diag(1 / tau ** 2)[None]).sum()       # Laplace correction (marginal likelihood)
    lp += -0.5 * (b0 ** 2).sum() / 1.0 - 0.5 * (Bm ** 2).sum() / 0.25 - 0.5 * (tau ** 2).sum() / 0.01   # priors N(0,1), N(0,0.5), half-N(0,0.1)
    return -(ll + lp), ll
for it in range(a.iters):
    loss, ll = objective(); opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0 or it == a.iters - 1: print(f"it {it} negpost {loss.item()/n:.5f} nll/row {-ll.item()/n:.5f} tau {torch.exp(log_tau).detach().cpu().numpy().round(3)}", flush=True)
with torch.no_grad():
    tau = torch.exp(log_tau); Bm = B * mask
    # Laplace posterior sd for RE: (sum_t f f^T + diag(1/tau^2))^-1  (expected information of Exponential/log-link = f f^T)
    Fr = torch.cat([torch.ones(n, 1, device=dev), F[:, re_idx]], 1)
    info = torch.zeros(P, R, R, device=dev); info.index_add_(0, pid_t, Fr[:, :, None] * Fr[:, None, :])
    cov = torch.linalg.inv(info + torch.diag(1 / tau ** 2)[None]); sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
    U_np, sd_np = U.cpu().numpy(), sd.cpu().numpy()
    Ucent = U_np - U_np.mean(0, keepdims=True)   # brms ranef are deviations; centre (fixed intercepts absorb the mean)
    re_rows = [dict(participant=p, coef=re_names[k], estimate=Ucent[i, k], sd=sd_np[i, k], q05=Ucent[i, k] - 1.645 * sd_np[i, k], q95=Ucent[i, k] + 1.645 * sd_np[i, k]) for i, p in enumerate(parts.cat.categories) for k in range(R)]
    pd.DataFrame(re_rows).to_csv(out / f"{tag}_ranef.csv", index=False)
    fe_rows = []
    for ci, c in enumerate(conds):
        for si, s in enumerate(speeds):
            j = ci * 2 + si; fe_rows.append(dict(Estimate=b0[j].item(), coef=f"SignalingType{c}:ResourceSpeed{s}:Intercept"))
            for k, kn in enumerate(feat_names): fe_rows.append(dict(Estimate=Bm[j, k].item(), coef=f"SignalingType{c}:ResourceSpeed{s}:{kn}"))
    pd.DataFrame(fe_rows).to_csv(out / f"{tag}_fixef.csv", index=False)
    pd.DataFrame([dict(model=a.model, half=a.half, n_rows=n, n_participants=P, iters=a.iters, fit_min=(time.time() - t0) / 60, nll_per_row=-ll.item() / n, **{f"tau_{k}": tau[i].item() for i, k in enumerate(re_names)})]).to_csv(out / f"{tag}_diag.csv", index=False)
print(pd.DataFrame(fe_rows).assign(cond=lambda x: x.coef.str.extract(r"SignalingType(\w+?):")[0], speed=lambda x: x.coef.str.extract(r"ResourceSpeed(\w+?):")[0], feature=lambda x: x.coef.str.replace(r"SignalingType\w+?:ResourceSpeed\w+?:", "", regex=True)).pivot_table(index=["feature", "speed"], columns="cond", values="Estimate").reindex(columns=conds).round(3))
print("RE sd across participants:", pd.DataFrame(Ucent, columns=re_names).std().round(3).to_dict()); print("mean posterior sd:", pd.DataFrame(sd_np, columns=re_names).mean().round(3).to_dict()); print(f"{(time.time()-t0)/60:.1f} min")
