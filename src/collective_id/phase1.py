"""Phase 1 training/evaluation utilities: search-phase encoder -> z -> tracking-phase teacher weights (decoder a)
or tracking rotation likelihood (decoder b). Fold-internal standardisation; groups never split across folds."""
import json, time, numpy as np, pandas as pd, torch
from pathlib import Path
from .models import Encoder, WeightDecoder, HyperRotationDecoder, exp_nll, kl_normal, reparam
from .data import load_features, participant_sequences, cond_onehot, Standardizer, pad_batch, SRC_FEATS, TRK_FEATS
from .split import load_split
from .paths import ROOT

def load_teacher(path, coefs):
    re = pd.read_csv(ROOT / path)
    est = re.pivot(index="participant", columns="coef", values="estimate")[coefs]
    sd = re.pivot(index="participant", columns="coef", values="sd")[coefs]
    return est, sd

class Phase1Data:
    """Holds per-participant source sequences, condition one-hots, teacher targets and tracking sequences."""
    def __init__(self, cfg, t_min=None, t_max=None, source_state=None):
        d = load_features()
        self.meta = d.groupby("participant").agg(cond=("cond", "first"), speed=("speed", "first"), group=("group", "first")).reset_index().set_index("participant")
        st = source_state or cfg["source_state"]
        self.src = participant_sequences(d, st, t_min, t_max)
        self.trk = participant_sequences(d, "tracking")
        self.cond = {p: cond_onehot(r.cond, r.speed) for p, r in self.meta.iterrows()}
        self.teacher, self.teacher_sd = load_teacher(cfg["teacher"], cfg["target_coefs"])
        self.participants = [p for p in self.meta.index if p in self.src and p in self.teacher.index and len(self.src[p][0]) >= 10]
        self.split = load_split()
        self.fold = {p: self.split["fold_of_participant"][p] for p in self.participants}

def stratified_val(parts, meta, frac, rng):
    val = []
    for _, g in meta.loc[parts].groupby(["cond", "speed"]):
        idx = list(g.index); rng.shuffle(idx); val += idx[:max(1, int(round(len(idx) * frac)))]
    return set(val)

def make_batch(D, parts, std, device, need_trk=False, crop=None, rng=None):
    seqs = [std(D.src[p][0]) for p in parts]
    if crop:   # random contiguous window of length <= crop (training-time augmentation)
        seqs = [s[(o := rng.integers(0, max(1, len(s) - crop + 1))):o + crop] for s in seqs]
    X, M = pad_batch(seqs, len(SRC_FEATS))
    C = torch.from_numpy(np.stack([D.cond[p] for p in parts]))
    out = dict(X=X.to(device), L=M.sum(1).to(device), C=C.to(device))
    if need_trk:
        F, MM = pad_batch([D.trk[p][0] if p in D.trk else np.zeros((1, len(TRK_FEATS)), np.float32) for p in parts], len(TRK_FEATS))
        Y, _ = pad_batch([D.trk[p][1][:, None] if p in D.trk else np.full((1, 1), 0.5, np.float32) for p in parts], 1)
        MM = MM * torch.tensor([1.0 if p in D.trk else 0.0 for p in parts])[:, None]
        out.update(F=F.to(device), Y=Y[..., 0].to(device), MM=MM.to(device))
    return out

def train_fold(D, cfg, fold, seed, decoder="a", device="cuda", out_dir: Path = None, verbose=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train_all = [p for p in D.participants if D.fold[p] != fold]; test = [p for p in D.participants if D.fold[p] == fold]
    val = stratified_val(train_all, D.meta, cfg["val_frac"], rng); train = [p for p in train_all if p not in val]; val = sorted(val)
    std = Standardizer().fit([D.src[p][0] for p in train])                       # fold-internal standardisation
    tmu = D.teacher.loc[train].mean(); tsd = D.teacher.loc[train].std() + 1e-6     # target scaling from training participants only
    T = lambda ps: torch.tensor(((D.teacher.loc[ps] - tmu) / tsd).to_numpy(np.float32), device=device)
    W = lambda ps: torch.tensor((1.0 / (D.teacher_sd.loc[ps] ** 2 + 1e-6)).to_numpy(np.float32), device=device)
    K = len(cfg["target_coefs"]); n_cond = 6
    enc = Encoder(len(SRC_FEATS), cfg["z_dim"], cfg["hidden"], cfg["n_layers"], cfg["kernel"], cfg["dropout"], cfg.get("arch", "cnn")).to(device)
    dec = (WeightDecoder(cfg["z_dim"], n_cond, K, cfg["hidden"]) if decoder == "a" else HyperRotationDecoder(cfg["z_dim"], n_cond, len(TRK_FEATS), cfg["hidden"])).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    in_drop = cfg.get("input_dropout", 0.0); crop = cfg.get("crop_len", None)
    def loss_fn(b, ps, sample):
        X = b["X"]
        if sample and in_drop > 0: X = X * (torch.rand_like(X[:, :1, :]) > in_drop).float() / (1 - in_drop)   # feature-channel dropout
        mu, lv = enc(X, b["L"]); z = reparam(mu, lv, sample); w = dec(z, b["C"])
        if decoder == "a":
            wt = W(ps); wt = wt / wt.mean(0, keepdim=True)                         # posterior-precision weights, normalised per coef
            fit = ((w - T(ps)) ** 2 * wt).mean()
        else:
            fit = exp_nll(w, b["F"], b["Y"], b["MM"])
        return fit + cfg["beta_kl"] * kl_normal(mu, lv), fit
    best, best_state, bad, hist = np.inf, None, 0, []
    bval = make_batch(D, val, std, device, decoder == "b")
    # fast path (no crop augmentation): pre-build padded GPU tensors for all training participants once per fold
    fast = crop is None
    if fast:
        ball = make_batch(D, train, std, device, decoder == "b"); Tall = T(train); Wall = W(train)
        def sub(idx):
            b = {k: v[idx] for k, v in ball.items()}
            if decoder == "b":
                L = int(b["MM"].sum(1).max().item()) if b["MM"].numel() else 1; b["F"], b["Y"], b["MM"] = b["F"][:, :L], b["Y"][:, :L], b["MM"][:, :L]
            L = int(b["L"].max().item()); b["X"] = b["X"][:, :L]
            return b
    def loss_fast(b, idx, sample):
        X = b["X"]
        if sample and in_drop > 0: X = X * (torch.rand_like(X[:, :1, :]) > in_drop).float() / (1 - in_drop)
        mu, lv = enc(X, b["L"]); z = reparam(mu, lv, sample); w = dec(z, b["C"])
        if decoder == "a":
            wt = Wall[idx]; wt = wt / wt.mean(0, keepdim=True); fit = ((w - Tall[idx]) ** 2 * wt).mean()
        else: fit = exp_nll(w, b["F"], b["Y"], b["MM"])
        return fit + cfg["beta_kl"] * kl_normal(mu, lv), fit
    t0 = time.time(); log_every = max(1, cfg["max_epochs"] // 1000)
    for ep in range(cfg["max_epochs"]):
        enc.train(); dec.train(); perm = rng.permutation(len(train)); tl = 0
        for i in range(0, len(train), cfg["batch_size"]):
            if fast:
                idx = torch.as_tensor(perm[i:i + cfg["batch_size"]], device=device); b = sub(idx)
                loss, _ = loss_fast(b, idx, True); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx); continue
            ps = [train[j] for j in perm[i:i + cfg["batch_size"]]]
            b = make_batch(D, ps, std, device, decoder == "b", crop=crop, rng=rng)
            loss, _ = loss_fn(b, ps, True); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(ps)
        enc.eval(); dec.eval()
        with torch.no_grad(): vl, vfit = loss_fn(bval, val, False)
        if ep % log_every == 0 or ep == cfg["max_epochs"] - 1: hist.append((ep, tl / len(train), vfit.item()))
        if vfit.item() < best - 1e-5: best, bad = vfit.item(), 0; best_state = ({k: v.clone() for k, v in enc.state_dict().items()}, {k: v.clone() for k, v in dec.state_dict().items()})
        else: bad += 1
        if bad >= cfg["patience"] and not cfg.get("no_early_stop", False): break
    if not cfg.get("no_early_stop", False):   # early stopping: restore best-validation weights; otherwise keep the final-epoch model
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.eval(); dec.eval()
    # ---- predictions for test participants ----
    with torch.no_grad():
        bt = make_batch(D, test, std, device, True); mu, lv = enc(bt["X"], bt["L"]); w = dec(mu, bt["C"])
        if decoder == "a": pred = pd.DataFrame((w.cpu().numpy() * tsd.to_numpy() + tmu.to_numpy()), index=test, columns=cfg["target_coefs"])
        else: pred = pd.DataFrame(w.cpu().numpy(), index=test, columns=["b0"] + TRK_FEATS)
        nll_test = exp_nll(w, bt["F"], bt["Y"], bt["MM"]).item() if decoder == "b" else None
        # cross-individual (A-H2): weights for k predicted from z of l (all test pairs), + z-space distances
        Z = mu.cpu().numpy(); C = bt["C"]
        n = len(test); cross = np.zeros((n, n, w.shape[1]), np.float32)
        for l in range(n):
            wl = dec(mu[l:l + 1].repeat(n, 1), C); cross[:, l] = wl.cpu().numpy()   # row k uses k's condition, l's z
        if decoder == "a": cross = cross * tsd.to_numpy() + tmu.to_numpy()
        # per-pair tracking NLL for decoder b
        cross_nll = None
        if decoder == "b":
            cross_nll = np.zeros((n, n), np.float32)
            for l in range(n):
                wl = torch.from_numpy(cross[:, l]).to(device)
                eta = wl[:, :1] + torch.einsum("bk,btk->bt", wl[:, 1:], bt["F"]); rate = torch.exp(-eta).clamp(1e-4, 1e4)
                lp = torch.log(rate) - rate * bt["Y"] - torch.log1p(-torch.exp(-rate * 1.02))
                cross_nll[:, l] = (-(lp * bt["MM"]).sum(1) / bt["MM"].sum(1).clamp(min=1)).cpu().numpy()
    res = dict(fold=fold, seed=seed, decoder=decoder, n_train=len(train), n_val=len(val), n_test=len(test), epochs=len(hist), best_val=best, train_min=(time.time() - t0) / 60, nll_test=nll_test)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_dir / "pred_test.csv"); pd.DataFrame(Z, index=test, columns=[f"z{i}" for i in range(Z.shape[1])]).to_csv(out_dir / "z_test.csv")
        np.savez_compressed(out_dir / "cross.npz", cross=cross, cross_nll=cross_nll if cross_nll is not None else np.zeros(0), test=np.array(test), Z=Z)
        json.dump(res, open(out_dir / "metrics.json", "w"), indent=1); pd.DataFrame(hist, columns=["epoch", "train_loss", "val_fit"]).to_csv(out_dir / "history.csv", index=False)
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "std_mu": std.mu, "std_sd": std.sd, "tmu": tmu.to_numpy(), "tsd": tsd.to_numpy(), "cfg": cfg}, out_dir / "model.pt")
    if verbose: print(res)
    return res, pred

def encode_participants(model_path, D, parts, device="cuda"):
    """Apply a saved fold model's encoder to arbitrary participants' source sequences (for ICC, simulated agents...)."""
    ck = torch.load(model_path, map_location=device, weights_only=False); cfg = ck["cfg"]
    enc = Encoder(len(SRC_FEATS), cfg["z_dim"], cfg["hidden"], cfg["n_layers"], cfg["kernel"], cfg["dropout"], cfg.get("arch", "cnn")).to(device); enc.load_state_dict(ck["enc"]); enc.eval()
    std = Standardizer(); std.mu, std.sd = ck["std_mu"], ck["std_sd"]
    with torch.no_grad():
        b = make_batch(D, parts, std, device); mu, _ = enc(b["X"], b["L"])
    return pd.DataFrame(mu.cpu().numpy(), index=parts, columns=[f"z{i}" for i in range(mu.shape[1])])
