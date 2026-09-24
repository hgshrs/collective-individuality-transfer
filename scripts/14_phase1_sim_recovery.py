"""§4.2-6: apply each fold/seed encoder to simulated agents' search-phase sequences and test whether z recovers the
known ABM parameters (β1, log k, r_max, x0) via 5-fold ridge regression (R²). Usage: --exp phase1_a_z4"""
import argparse, glob, json, numpy as np, pandas as pd, torch, yaml
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from collective_id.paths import RUNS, RESULTS, PROCESSED, CONFIGS
from collective_id.phase1 import Phase1Data, encode_participants
from collective_id.data import participant_sequences, cond_onehot
ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="phase1_a_z4"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args()
cfg = json.load(open(RUNS / a.exp / "config.json"))
F = pd.read_parquet(PROCESSED / "sim_agents_features.parquet"); F["qxpos"] = 0.0; F["qxmov"] = 0.0
Pp = pd.read_csv(PROCESSED / "sim_agents_params.csv").set_index("participant")
class SimData:  # duck-typed container for encode_participants
    pass
D = SimData(); D.src = participant_sequences(F, "searching"); D.cond = {p: cond_onehot("FP", "slow" if Pp.loc[p, "speed"] == 0.4 else "fast") for p in Pp.index}
parts = [p for p in Pp.index if p in D.src and len(D.src[p][0]) >= 10]
rows, Zs = [], []
for mp in sorted(glob.glob(str(RUNS / a.exp / "seed*" / "fold*" / "model.pt"))):
    seed = int(mp.split("seed")[1].split("/")[0]); fold = int(mp.split("fold")[1].split("/")[0])
    Z = encode_participants(mp, D, parts, a.device); Z["seed"] = seed; Z["fold"] = fold; Zs.append(Z)
    zc = [c for c in Z.columns if c.startswith("z")]
    for target in ["beta1", "log_k", "rmax", "x0"]:
        y = Pp.loc[parts, target].to_numpy(); pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), Z[zc].to_numpy(), y, cv=KFold(5, shuffle=True, random_state=0))
        rows.append(dict(seed=seed, fold=fold, target=target, r2=1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), r=np.corrcoef(y, pred)[0, 1]))
res = pd.DataFrame(rows); res.to_csv(RESULTS / "phase1" / f"sim_recovery_{a.exp}.csv", index=False)
summ = res.groupby("target").agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"), r_mean=("r", "mean")).reset_index(); summ.to_csv(RESULTS / "phase1" / f"sim_recovery_summary_{a.exp}.csv", index=False)
pd.concat(Zs).to_csv(RESULTS / "phase1" / f"sim_z_{a.exp}.csv"); print(summ.round(3).to_string(index=False)); print("n agents:", len(parts))
