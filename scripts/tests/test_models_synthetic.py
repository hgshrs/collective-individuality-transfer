"""Unit test on synthetic data: encoder+weight decoder recovers a latent trait from sequences; exp_nll trains."""
import torch, numpy as np
from collective_id.models import Encoder, WeightDecoder, HyperRotationDecoder, exp_nll, kl_normal, reparam
torch.manual_seed(0); np.random.seed(0)
B, T, F, M = 256, 100, 5, 2
z_true = torch.randn(B, M)
X = torch.randn(B, T, F) * 0.5 + z_true[:, :1, None] * torch.tensor([1., 0, 0, 0, 0]) + z_true[:, 1:2, None] * torch.tensor([0, 1., 0, 0, 0])
cond = torch.zeros(B, 6); cond[:, 0] = 1; cond[:, 4] = 1
W_true = torch.stack([z_true[:, 0], -z_true[:, 1], z_true[:, 0] * 0.5], 1)
enc, dec = Encoder(F, M), WeightDecoder(M, 6, 3)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
for it in range(400):
    mu, lv = enc(X); z = reparam(mu, lv); w = dec(z, cond)
    loss = ((w - W_true) ** 2).mean() + 1e-3 * kl_normal(mu, lv)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    mu, _ = enc(X); w = dec(mu, cond)
    r = [np.corrcoef(w[:, k], W_true[:, k])[0, 1] for k in range(3)]
print("weight recovery r:", np.round(r, 3)); assert min(r) > 0.8
# hyper decoder: fit synthetic exponential rotations
K = 3; feats = torch.randn(B, T, K); w_true = torch.cat([torch.full((B, 1), -1.0), z_true[:, :1].repeat(1, K) * torch.tensor([1., -1, 0.5])], 1)
eta = w_true[:, :1] + torch.einsum("bk,btk->bt", w_true[:, 1:], feats); y = torch.distributions.Exponential(torch.exp(-eta)).sample().clamp(0.01, 1.01)
hyp = HyperRotationDecoder(M, 6, K); opt = torch.optim.Adam(list(enc.parameters()) + list(hyp.parameters()), 1e-3)
mask = torch.ones(B, T)
for it in range(400):
    mu, lv = enc(X); w = hyp(reparam(mu, lv), cond); loss = exp_nll(w, feats, y, mask) + 1e-3 * kl_normal(mu, lv)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    w = hyp(enc(X)[0], cond); r2 = np.corrcoef(w[:, 1], w_true[:, 1])[0, 1]
print("hyper-decoder slope recovery r:", round(float(r2), 3)); assert r2 > 0.7
print("OK")
