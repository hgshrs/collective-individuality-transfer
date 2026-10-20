"""Phase 2 Plan B, non-linear variant (spec §5.1 'Phase 1 encoder structure re-learned'): 1D-CNN encoder on the
first-3-min own-movement 1 s sequence -> z (M=4) -> MLP(z + cell one-hot) -> standardised role targets (MSE, masked for
missing discoverer gain). 10 folds x 3 seeds, fold-internal standardisation, early stopping on a 10 % validation split.
Usage: python scripts/33_phase2_cnn.py [--seeds 0,1,2] [--device cuda]
Outputs: runs/phase2_b_cnn/seed*/fold*/, results/phase2/planB_cnn_oof_predictions.csv, planB_cnn_r.csv
"""
import argparse, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from collective_id.paths import RESULTS, RUNS, PROCESSED
from collective_id.models import Encoder, kl_normal, reparam
from collective_id.phase2 import load_phase2, onehot, boot_r_ci, paired_delta

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--max_epochs", type=int, default=500); ap.add_argument("--patience", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args(); dev = a.device
OUT = RESULTS / "phase2"; EXP = RUNS / "phase2_b_cnn"
F3, T, fold = load_phase2(); meta = T[T.cond != "A"].copy(); parts = [p for p in meta.index if p in fold.index]; meta = meta.loc[parts]
TARGETS = ["in_deg", "out_deg", "in_minus_out", "n_disc_rest", "disc_gain_rest"]
seq = pd.read_parquet(PROCESSED / "phase2_first3min_seq.parquet"); FEATS = ["rot_abs", "rot_signed", "no_move", "spd", "mov", "dist_n", "trk"]
S = {p: g.sort_values("t")[FEATS].to_numpy(np.float32) for p, g in seq[seq.participant.isin(parts)].groupby("participant")}
L = min(len(v) for v in S.values()); X_all = np.stack([S[p][:L] for p in parts]); Y_all = meta[TARGETS].to_numpy(np.float32); C_all = onehot(meta).to_numpy(np.float32)
oh_cols = onehot(meta).shape[1]

class Net(nn.Module):
    def __init__(self, nf, nc, k, z=4, h=64):
        super().__init__(); self.enc = Encoder(nf, z, h, 3, 5, 0.1); self.dec = nn.Sequential(nn.Linear(z + nc, h), nn.GELU(), nn.Linear(h, k))
    def forward(self, x, c, sample):
        mu, lv = self.enc(x); return self.dec(torch.cat([reparam(mu, lv, sample), c], -1)), mu, lv

def run(fold_id, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx_tr_all = np.array([i for i, p in enumerate(parts) if fold[p] != fold_id]); idx_te = np.array([i for i, p in enumerate(parts) if fold[p] == fold_id])
    rng.shuffle(idx_tr_all); nv = max(1, int(0.1 * len(idx_tr_all))); idx_val, idx_tr = idx_tr_all[:nv], idx_tr_all[nv:]
    mu_x, sd_x = X_all[idx_tr].reshape(-1, len(FEATS)).mean(0), X_all[idx_tr].reshape(-1, len(FEATS)).std(0) + 1e-6
    mu_y, sd_y = np.nanmean(Y_all[idx_tr], 0), np.nanstd(Y_all[idx_tr], 0) + 1e-6
    tX = lambda i: torch.tensor((X_all[i] - mu_x) / sd_x, device=dev); tC = lambda i: torch.tensor(C_all[i], device=dev)
    Yn = (Y_all - mu_y) / sd_y; M = ~np.isnan(Yn); Yn = np.nan_to_num(Yn)
    tY = lambda i: torch.tensor(Yn[i], device=dev); tM = lambda i: torch.tensor(M[i].astype(np.float32), device=dev)
    net = Net(len(FEATS), oh_cols, len(TARGETS)).to(dev); opt = torch.optim.Adam(net.parameters(), a.lr)
    def loss(i, sample):
        out, mu, lv = net(tX(i), tC(i), sample); m = tM(i)
        return (((out - tY(i)) ** 2) * m).sum() / m.sum().clamp(min=1) + 1e-3 * kl_normal(mu, lv)
    best, bad, state = np.inf, 0, None
    for ep in range(a.max_epochs):
        net.train(); perm = rng.permutation(idx_tr)
        for j in range(0, len(perm), 32):
            l = loss(perm[j:j + 32], True); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = loss(idx_val, False).item()
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else: bad += 1
        if bad >= a.patience: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): out, mu, _ = net(tX(idx_te), tC(idx_te), False)
    pred = pd.DataFrame(out.cpu().numpy() * sd_y + mu_y, index=[parts[i] for i in idx_te], columns=TARGETS)
    Z = pd.DataFrame(mu.cpu().numpy(), index=pred.index, columns=[f"z{i}" for i in range(4)])
    return pred, Z, dict(fold=fold_id, seed=seed, epochs=ep + 1, best_val=best, n_train=len(idx_tr), n_test=len(idx_te))

t0 = time.time(); allp = {}
for seed in [int(s) for s in a.seeds.split(",")]:
    P, Zs = [], []
    for f in range(10):
        od = EXP / f"seed{seed}" / f"fold{f}"; od.mkdir(parents=True, exist_ok=True)
        pred, Z, res = run(f, seed); pred.to_csv(od / "pred_test.csv"); Z.to_csv(od / "z_test.csv"); json.dump(res, open(od / "metrics.json", "w")); P.append(pred); Zs.append(Z)
        print(res, f"{time.time()-t0:.0f}s", flush=True)
    allp[seed] = pd.concat(P).loc[parts]; pd.concat(Zs).loc[parts].to_csv(OUT / f"z_planB_cnn_seed{seed}.csv")
rows = []
for seed, P in allp.items():
    for y in TARGETS:
        r, lo, hi, n = boot_r_ci(P[y], meta[y].astype(float), n_boot=500); rows.append(dict(seed=seed, target=y, r=r, ci_lo=lo, ci_hi=hi, n=n))
R = pd.DataFrame(rows); R.to_csv(OUT / "planB_cnn_r.csv", index=False)
Pm = sum(allp.values()) / len(allp); Pm.to_csv(OUT / "planB_cnn_oof_predictions.csv")
# compare with the linear z and the score baseline (seed-averaged prediction)
lin = pd.read_csv(OUT / "planB_oof_predictions.csv", index_col=0); cmp = []
for y in TARGETS:
    yy = meta[y].astype(float)
    for b in ["z_pca4", "score3_only"]:
        d = paired_delta(Pm[y], lin[f"{y}|{b}"].reindex(parts), yy, meta.group, n_boot=1000); d.update(target=y, model="cnn_seedmean", baseline=b); cmp.append(d)
pd.DataFrame(cmp).to_csv(OUT / "planB_cnn_delta.csv", index=False)
pd.set_option("display.width", 200); print(R.groupby("target").r.agg(["mean", "std"]).round(3)); print(pd.DataFrame(cmp).round(3).to_string())
