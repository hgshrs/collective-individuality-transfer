"""Phase 1 datasets: per-participant sequences (source = search phase, target = tracking phase) from features_1hz.parquet.
All standardisation statistics are computed from training participants of the fold only (leak prevention)."""
import numpy as np, pandas as pd, torch
from .paths import PROCESSED
from .features import COND, SPEED

SRC_FEATS = ["rot_abs", "rot_signed", "soc_pos", "soc_mov", "soc_qual_search", "n_vis", "dist_search"]  # search-phase inputs
TRK_FEATS = ["detector", "dist_track", "soc_pos", "soc_mov", "qxpos", "qxmov"]                          # tracking predictors (paper model)
RE_COEFS = ["Intercept", "Detector", "withSocInfo1:SocPosInFOV", "withSocInfo1:SocMovDir", "withSocInfo1:SocPosInFOV:SocQual", "withSocInfo1:SocMovDir:SocQual"]

def cond_onehot(cond: str, speed: str) -> np.ndarray:
    v = np.zeros(len(COND) + len(SPEED), np.float32); v[COND.index(cond)] = 1; v[len(COND) + SPEED.index(speed)] = 1
    return v

def load_features() -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED / "features_1hz.parquet")
    d["qxpos"] = d.soc_qual_track * d.soc_pos; d["qxmov"] = d.soc_qual_track * d.soc_mov
    return d

def participant_sequences(d: pd.DataFrame, state: str, t_min=None, t_max=None, max_len: int = 900):
    """Return dict participant -> (X (T,F) float32, y (T,) rot_abs) for rows of the given state (search or tracking)."""
    m = d.state == state
    if t_min is not None: m &= d.t >= t_min
    if t_max is not None: m &= d.t < t_max
    if state == "tracking": m &= d.no_move < 3
    dd = d[m]
    feats = SRC_FEATS if state == "searching" else TRK_FEATS
    out = {}
    for p, g in dd.groupby("participant", sort=False):
        X = g[feats].to_numpy(np.float32)[:max_len]
        out[p] = (X, g.rot_abs.to_numpy(np.float32)[:max_len])
    return out

class Standardizer:
    def __init__(self): self.mu = None; self.sd = None
    def fit(self, arrays):
        X = np.concatenate(arrays, 0); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; return self
    def __call__(self, X): return (X - self.mu) / self.sd

def pad_batch(seqs, F):
    L = max(len(s) for s in seqs); B = len(seqs)
    X = np.zeros((B, L, F), np.float32); M = np.zeros((B, L), np.float32)
    for i, s in enumerate(seqs): X[i, :len(s)] = s; M[i, :len(s)] = 1
    return torch.from_numpy(X), torch.from_numpy(M)
