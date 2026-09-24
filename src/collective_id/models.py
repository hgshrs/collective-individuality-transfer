"""Phase 1 models (AGENT_SPEC §4.1): 1D-CNN encoder -> z (VAE-style), MLP decoder (a) z+cond -> teacher weights,
hypernetwork decoder (b) z+cond -> weights of an Exponential rotation predictor."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    """x: (B, T, F) 1 s sequence of features, returns mu, logvar of z (B, M). arch: 'cnn' (default) or 'gru'."""
    def __init__(self, n_feat: int, z_dim: int = 4, hidden: int = 64, n_layers: int = 3, kernel: int = 5, dropout: float = 0.1, arch: str = "cnn"):
        super().__init__()
        self.arch = arch
        if arch == "gru":
            self.rnn = nn.GRU(n_feat, hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
            self.proj = nn.Linear(2 * hidden, hidden)
        else:
            layers, c = [], n_feat
            for i in range(n_layers):
                layers += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.GELU(), nn.Dropout(dropout)]
                c = hidden
            self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 2 * z_dim))
        self.z_dim = z_dim

    def forward(self, x, lengths=None):
        if self.arch == "gru":
            h = self.proj(self.rnn(x)[0]).transpose(1, 2)   # (B, H, T)
        else:
            h = self.conv(x.transpose(1, 2))            # (B, H, T)
        if lengths is not None:                       # masked mean/max pooling over valid steps
            m = (torch.arange(h.shape[-1], device=h.device)[None, :] < lengths[:, None]).float()[:, None, :]
            mean = (h * m).sum(-1) / m.sum(-1).clamp(min=1)
            mx = (h + (m - 1) * 1e4).amax(-1)
        else:
            mean, mx = h.mean(-1), h.amax(-1)
        out = self.head(torch.cat([mean, mx], -1))
        mu, logvar = out[:, :self.z_dim], out[:, self.z_dim:].clamp(-8, 4)
        return mu, logvar

def reparam(mu, logvar, sample: bool = True):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu

class WeightDecoder(nn.Module):
    """(a) z + condition one-hot -> teacher weight vector (K)."""
    def __init__(self, z_dim: int, n_cond: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + n_cond, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))
    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], -1))

class HyperRotationDecoder(nn.Module):
    """(b) z + condition -> per-participant weight vector w (n_feat_t + 1) of a linear Exponential model of tracking
    rotation: |dtheta| ~ Exponential(rate = exp(-(w0 + w·f_t))) i.e. log-mean = w0 + w·f (brms exponential link)."""
    def __init__(self, z_dim: int, n_cond: int, n_feat_t: int, hidden: int = 64):
        super().__init__()
        self.n_feat_t = n_feat_t
        self.net = nn.Sequential(nn.Linear(z_dim + n_cond, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_feat_t + 1))
    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], -1))    # (B, n_feat_t+1)

def exp_nll(w, feats, y, mask, ub: float = 1.02):
    """Truncated-Exponential negative log-likelihood (as brms trunc(ub=1.02)) for rotation magnitude.
    w: (B, K+1); feats: (B, T, K); y: (B, T) in (0.01, 1.01]; mask: (B, T)."""
    eta = w[:, :1] + torch.einsum("bk,btk->bt", w[:, 1:], feats)      # log mean
    rate = torch.exp(-eta).clamp(1e-4, 1e4)
    logp = torch.log(rate) - rate * y - torch.log1p(-torch.exp(-rate * ub))
    return -(logp * mask).sum() / mask.sum().clamp(min=1)

def kl_normal(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
