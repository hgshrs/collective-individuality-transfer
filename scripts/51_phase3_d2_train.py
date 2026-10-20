"""Phase 3 D-2: z-conditioned neural policy (AGENT_SPEC §6.1 D-2, §6.3-6). Per fold of the fixed group-wise split:
a shared MLP policy  pi(x_t, z_i) -> (P(no rotation), Exponential mean of |Δθ| in deg/s (truncated at 150), P(direction +))
is trained on the 1 s data of the TRAINING participants only, with a learned 4-d embedding z_i per training participant
(prior N(0, 1)). Held-out participants (the evaluated groups) get their z by maximising their own likelihood with the
network frozen. Individual rotation data only; no group outcome (§6.2).
Inputs x_t (same information as the paper's features, plus the previous second's rotation state so that the policy can
be simulated): tracking flag, detector (Δd/7, data scale), detector change, normalised distance, with-social flag,
#visible/4, signed social angle (positive = toward the positive rotation direction), social movement-direction
difference, social quality (tracking: continuous; searching: ±1 by best visible other), previous direction, previous zero flag.
Usage: python scripts/51_phase3_d2_train.py [--epochs 40] [--z_dim 4] [--device cuda] [--seed 0]
Outputs: runs/phase3_d2/fold*/model.pt, results/phase3/d2_z.csv (participant z, fold), d2_fit_summary.json, d2_predictive.csv
"""
import argparse, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collective_id.paths import PROCESSED, RESULTS, RUNS, RAW_DATA
from collective_id.split import load_split

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--z_dim", type=int, default=4); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--folds", default="0-9"); a = ap.parse_args()
dev = a.device; torch.manual_seed(a.seed); np.random.seed(a.seed)
OUT = RESULTS / "phase3"; EXP = RUNS / "phase3_d2"; EXP.mkdir(parents=True, exist_ok=True)
FEATS = ["trk", "det", "ddet", "dist_n", "has", "nvis", "alpha", "mov", "q", "prev_dir", "prev_zero"]
MAXROT = 150.0


def build_table():
    f = pd.read_parquet(PROCESSED / "features_1hz.parquet")
    ts = pd.read_csv(RAW_DATA / "processed/time_series_data.csv", usecols=["participant", "step", "mean_degree_visible_players_first"]).rename(columns={"step": "t"})
    f = f.merge(ts, on=["participant", "t"], how="left").sort_values(["participant", "t"])
    f = f[f.state.isin(["tracking", "searching"])].copy()
    f["trk"] = (f.state == "tracking").astype(np.float32)
    f["det"] = f.detector.fillna(0).astype(np.float32)
    f["ddet"] = (f.det - f.groupby("participant").det.shift(1)).fillna(0).astype(np.float32)
    f["dist_n"] = (f.dist_raw / 200.0 - 1).astype(np.float32)
    f["has"] = (f.n_vis > 0).astype(np.float32); f["nvis"] = (f.n_vis / 4.0).astype(np.float32)
    f["alpha"] = np.where(f.has > 0, -f.mean_degree_visible_players_first.fillna(0) / 45.746, 0.0).astype(np.float32)     # sign flipped: positive = toward
    f["mov"] = f.soc_mov.astype(np.float32)
    f["q"] = np.where(f.trk > 0, -np.clip(f.soc_qual_raw / 40.0, -1, 1), f.soc_qual_search); f["q"] = np.where(f.has > 0, f.q, 0.0).astype(np.float32)
    rs = f.rot_signed.to_numpy(); zero = f.rot_abs.to_numpy() <= 0.011
    f["zero"] = zero.astype(np.float32); f["mag"] = np.clip((f.rot_abs.to_numpy() - 0.01) * MAXROT, 0, MAXROT).astype(np.float32); f["dirp"] = (rs > 0).astype(np.float32)
    # previous direction (last non-zero sign) and previous-zero flag, per participant
    sgn = pd.Series(np.where(zero, np.nan, np.sign(rs)), index=f.index)
    f["prev_dir"] = sgn.groupby(f.participant).shift(1).groupby(f.participant).ffill().fillna(0).astype(np.float32)
    f["prev_zero"] = f.groupby("participant").zero.shift(1).fillna(0).astype(np.float32)
    f["half"] = np.where(f.t < 450, 0, 1)
    keep = (f.state == "searching") | (f.no_move < 3)
    return f[keep][["participant", "cond", "speed", "group", "t", "half"] + FEATS + ["zero", "mag", "dirp"]].reset_index(drop=True)


class Policy(nn.Module):
    def __init__(self, nf, z_dim, h):
        super().__init__(); self.net = nn.Sequential(nn.Linear(nf + z_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
    def forward(self, x, z):
        o = self.net(torch.cat([x, z], -1)); return o[:, 0], o[:, 1].clamp(-2, 6), o[:, 2]     # logit p_zero, log mean magnitude (deg/s), logit p_dir+


def nll(out, zero, mag, dirp):
    lz, lmu, ld = out
    l_zero = F.binary_cross_entropy_with_logits(lz, zero, reduction="none")
    rate = torch.exp(-lmu); lmag = torch.log(rate) - rate * mag - torch.log1p(-torch.exp(-rate * MAXROT))     # truncated Exponential on (0, 150]
    l_dir = F.binary_cross_entropy_with_logits(ld, dirp, reduction="none")
    return l_zero - (1 - zero) * (lmag - l_dir)                                                          # per-second NLL (zero mass + magnitude density + direction)


def fit_z(model, X, zero, mag, dirp, pid_idx, n_p, steps=300, lr=0.05, prior=1.0):
    """Optimise embeddings only (network frozen) for a set of participants; returns (n_p, z_dim)."""
    Z = torch.zeros(n_p, a.z_dim, device=dev, requires_grad=True); opt = torch.optim.Adam([Z], lr)
    for _ in range(steps):
        loss = nll(model(X, Z[pid_idx]), zero, mag, dirp).sum() + 0.5 * prior * (Z ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return Z.detach()


def tens(df):
    X = torch.tensor(df[FEATS].to_numpy(np.float32), device=dev); return X, torch.tensor(df.zero.to_numpy(), device=dev), torch.tensor(df.mag.to_numpy(), device=dev), torch.tensor(df.dirp.to_numpy(), device=dev)


tab = build_table(); tab.to_parquet(PROCESSED / "phase3_d2_table.parquet", index=False)
sp = load_split(); fold = pd.Series(sp["fold_of_participant"]); tab["fold"] = tab.participant.map(fold)
mu, sd = tab[FEATS].mean(), tab[FEATS].std() + 1e-6          # input scaling constants (feature definitions only; no targets) — recomputed per fold below
f0, f1 = (int(x) for x in a.folds.split("-"))
Zrows, pred_rows, summ = [], [], {}
t0 = time.time()
for fo in range(f0, f1 + 1):
    tr = tab[tab.fold != fo]; te = tab[tab.fold == fo]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-6
    trn = tr.copy(); trn[FEATS] = (trn[FEATS] - mu) / sd; ten = te.copy(); ten[FEATS] = (ten[FEATS] - mu) / sd
    pids = sorted(trn.participant.unique()); pidx = {p: i for i, p in enumerate(pids)}
    X, Zr, Mg, Dp = tens(trn); PI = torch.tensor(trn.participant.map(pidx).to_numpy(), device=dev)
    # validation = 10 % random seconds of the training participants
    g = torch.Generator(device="cpu").manual_seed(a.seed + fo); perm = torch.randperm(len(X), generator=g).to(dev); nv = len(X) // 10; vi, ti = perm[:nv], perm[nv:]
    model = Policy(len(FEATS), a.z_dim, a.hidden).to(dev); emb = nn.Embedding(len(pids), a.z_dim).to(dev); nn.init.normal_(emb.weight, 0, 0.1)
    opt = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()), a.lr)
    best, state, bad = np.inf, None, 0
    for ep in range(a.epochs):
        model.train(); p2 = ti[torch.randperm(len(ti), device=dev)]
        for i in range(0, len(p2), 4096):
            b = p2[i:i + 4096]; z = emb(PI[b])
            loss = nll(model(X[b], z), Zr[b], Mg[b], Dp[b]).mean() + 0.5 * (emb.weight ** 2).sum() / len(X)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nll(model(X[vi], emb(PI[vi])), Zr[vi], Mg[vi], Dp[vi]).mean().item()
        if vl < best - 1e-4: best, bad, state = vl, 0, ({k: v.clone() for k, v in model.state_dict().items()}, emb.weight.detach().clone())
        else: bad += 1
        if bad >= 5: break
    model.load_state_dict(state[0]); model.eval(); Ztr = state[1]
    for p in model.parameters(): p.requires_grad_(False)
    # population NLL (z = 0) and individual NLL on the validation seconds
    with torch.no_grad(): v0 = nll(model(X[vi], torch.zeros(len(vi), a.z_dim, device=dev)), Zr[vi], Mg[vi], Dp[vi]).mean().item()
    # held-out participants: z from their own full-session data (frozen network); plus first-half z -> second-half NLL (predictive check)
    tp = sorted(ten.participant.unique()); tidx = {p: i for i, p in enumerate(tp)}
    Xt, Zt, Mt, Dt = tens(ten); PIt = torch.tensor(ten.participant.map(tidx).to_numpy(), device=dev)
    Zte = fit_z(model, Xt, Zt, Mt, Dt, PIt, len(tp))
    h1 = ten.half.to_numpy() == 0; H1 = torch.tensor(h1, device=dev)
    Z1 = fit_z(model, Xt[H1], Zt[H1], Mt[H1], Dt[H1], PIt[H1], len(tp))
    with torch.no_grad():
        n_ind = nll(model(Xt[~H1], Z1[PIt[~H1]]), Zt[~H1], Mt[~H1], Dt[~H1]); n_pop = nll(model(Xt[~H1], torch.zeros((~H1).sum(), a.z_dim, device=dev)), Zt[~H1], Mt[~H1], Dt[~H1])
    df2 = pd.DataFrame({"participant": ten.participant.to_numpy()[~h1], "trk": te.trk.to_numpy()[~h1], "n_ind": n_ind.cpu().numpy(), "n_pop": n_pop.cpu().numpy()})   # te: unstandardised flag
    pr = df2.groupby("participant").agg(nll_d2_indiv=("n_ind", "mean"), nll_d2_pop=("n_pop", "mean"), n_sec=("n_ind", "size")).reset_index()
    prt = df2[df2.trk == 1].groupby("participant").agg(nll_d2_indiv_track=("n_ind", "mean"), nll_d2_pop_track=("n_pop", "mean")).reset_index()
    pr = pr.merge(prt, on="participant", how="left"); pr["fold"] = fo; pred_rows.append(pr)
    for p in tp: Zrows.append(dict(participant=p, fold=fo, **{f"z{i}": float(Zte[tidx[p], i]) for i in range(a.z_dim)}))
    torch.save(dict(model=model.state_dict(), mu=mu.to_numpy(), sd=sd.to_numpy(), feats=FEATS, z_dim=a.z_dim, hidden=a.hidden, train_pids=pids, Ztr=Ztr.cpu().numpy()), EXP / f"fold{fo}.pt")
    summ[fo] = dict(epochs=ep + 1, val_nll_indiv=best, val_nll_pop=v0, n_train_sec=int(len(ti)), n_test_participants=len(tp), min=(time.time() - t0) / 60)
    print(fo, summ[fo], flush=True)
Z = pd.DataFrame(Zrows); Z.to_csv(OUT / "d2_z.csv", index=False)
P = pd.concat(pred_rows); P.to_csv(OUT / "d2_predictive.csv", index=False)
json.dump(dict(folds=summ, mean_val_gain_indiv_vs_pop=float(np.mean([s["val_nll_pop"] - s["val_nll_indiv"] for s in summ.values()])),
               heldout_gain_indiv_vs_pop=float((P.nll_d2_pop - P.nll_d2_indiv).mean()), heldout_gain_track=float((P.nll_d2_pop_track - P.nll_d2_indiv_track).mean()), frac_indiv_better=float((P.nll_d2_indiv < P.nll_d2_pop).mean())),
          open(OUT / "d2_fit_summary.json", "w"), indent=1)
print(json.load(open(OUT / "d2_fit_summary.json")))
