"""§4.2-7: temporal stability of z. Encode first-half (t<450) and second-half (t>=450) search sequences of held-out
participants with their fold's encoder; ICC(2,1)-style agreement per z dimension (and Pearson r), averaged over seeds.
Usage: --exp phase1_a_z4"""
import argparse, glob, json, numpy as np, pandas as pd, torch
from collective_id.paths import RUNS, RESULTS
from collective_id.phase1 import Phase1Data, encode_participants
ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="phase1_a_z4"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args()
cfg = json.load(open(RUNS / a.exp / "config.json"))
D1 = Phase1Data(cfg, t_max=450); D2 = Phase1Data(cfg, t_min=450)
def icc21(x, y):
    """ICC(2,1) two-way random, single measures, absolute agreement."""
    n = len(x); k = 2; data = np.stack([x, y], 1); gm = data.mean(); ms_r = k * ((data.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((data.mean(0) - gm) ** 2).sum() / (k - 1); ss_e = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + gm) ** 2).sum(); ms_e = ss_e / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
rows, Z1s, Z2s = [], [], []
for mp in sorted(glob.glob(str(RUNS / a.exp / "seed*" / "fold*" / "model.pt"))):
    seed = int(mp.split("seed")[1].split("/")[0]); fold = int(mp.split("fold")[1].split("/")[0])
    test = [p for p in D1.participants if D1.fold[p] == fold and p in D2.src and len(D1.src[p][0]) >= 10 and len(D2.src[p][0]) >= 10]
    Z1 = encode_participants(mp, D1, test, a.device); Z2 = encode_participants(mp, D2, test, a.device); Z1["seed"] = seed; Z2["seed"] = seed; Z1s.append(Z1); Z2s.append(Z2)
Z1 = pd.concat(Z1s); Z2 = pd.concat(Z2s); zc = [c for c in Z1.columns if c.startswith("z")]
for s in sorted(Z1.seed.unique()):
    a1, a2 = Z1[Z1.seed == s], Z2[Z2.seed == s]
    for z in zc: rows.append(dict(seed=s, dim=z, icc21=icc21(a1[z].to_numpy(), a2[z].to_numpy()), r=np.corrcoef(a1[z], a2[z])[0, 1], n=len(a1)))
res = pd.DataFrame(rows); res.to_csv(RESULTS / "phase1" / f"icc_{a.exp}.csv", index=False)
print(res.groupby("dim").agg(icc_mean=("icc21", "mean"), icc_sd=("icc21", "std"), r_mean=("r", "mean")).round(3))
