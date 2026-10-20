"""Phase 3 D-2: leave-one-group-out simulation of every analysed group with the z-conditioned neural policy (scripts/51)
in the vectorised environment (src/collective_id/env.py). Variants (same names as D-1 where comparable):
  main           members' own z (inferred with the fold network that never saw their group)
  ctrl1_pop      population policy (z = 0 for every agent)
  ctrl2_avg      mean z of the fold's TRAINING participants in the same cell (identical agents, constant within cell)
  ctrl2b_gmean   mean z of the group's own members (composition removed)
  ctrl3_nominal  each member alone (N = 1) with their own z; group prediction = mean over members
Usage: python scripts/52_phase3_d2_simulate.py [--seeds 100] [--device cuda]
Output: results/phase3/sim_runs_d2.parquet (one row per simulated run, same columns as sim_runs.parquet)
"""
import argparse, time, json, numpy as np, pandas as pd, torch
from collective_id.paths import RESULTS, RUNS
from collective_id.split import load_split
from collective_id.env import Env, group_metrics, wrap, FOV_HALF
import importlib.util
spec = importlib.util.spec_from_file_location("d2", RESULTS.parent / "scripts" / "51_phase3_d2_train.py")

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=100); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--groups", default="all"); a = ap.parse_args()
dev = a.device; OUT = RESULTS / "phase3"; EXP = RUNS / "phase3_d2"; MAXROT = 150.0
import torch.nn as nn
class Policy(nn.Module):
    def __init__(self, nf, z_dim, h):
        super().__init__(); self.net = nn.Sequential(nn.Linear(nf + z_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
    def forward(self, x, z):
        o = self.net(torch.cat([x, z], -1)); return o[:, 0], o[:, 1].clamp(-2, 6), o[:, 2]

Z = pd.read_csv(OUT / "d2_z.csv"); zcols = [c for c in Z.columns if c.startswith("z")]; Z = Z.set_index("participant")
par = pd.read_csv(OUT / "policy_params_all.csv").set_index("participant"); grp = par[par.cond != "A"]
sp = load_split(); fold_g = sp["fold_of_group"]
groups = sorted(grp.group.unique()) if a.groups == "all" else a.groups.split(",")


def make_policy(model, mu, sd, Zarr, gen):
    """Zarr: (S, N, z_dim) numpy. Returns policy(env, obs, t) -> signed rotation (rad)."""
    S, N, _ = Zarr.shape; Zt = torch.tensor(Zarr.reshape(S * N, -1), dtype=torch.float32, device=dev)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=dev); sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)
    state = dict(prev_dir=np.zeros((S, N)), prev_zero=np.zeros((S, N)), det_prev=np.zeros((S, N)))
    def policy(env, o, t):
        det = np.clip(o["dd"] / 7.0, -1, 1) if t > 0 else np.zeros((S, N))
        x = np.stack([o["tracking"].astype(float), det, det - state["det_prev"], o["d"] / 200.0 - 1, o["has"].astype(float), o["n_vis"] / 4.0,
                      o["alpha"] / FOV_HALF, (np.abs(o["mov"]) / np.pi - 0.5) * 2 * o["has"], o["q_best"], state["prev_dir"], state["prev_zero"]], -1).reshape(S * N, -1)
        state["det_prev"] = det
        with torch.no_grad():
            xt = (torch.tensor(x, dtype=torch.float32, device=dev) - mu_t) / sd_t
            lz, lmu, ld = model(xt, Zt)
            pz = torch.sigmoid(lz); muv = torch.exp(lmu); pd_ = torch.sigmoid(ld)
            u = torch.rand(S * N, device=dev, generator=gen)
            mag = -muv * torch.log1p(-u * (1 - torch.exp(-MAXROT / muv)))                  # truncated Exponential sample on (0, 150]
            zero = torch.rand(S * N, device=dev, generator=gen) < pz
            dire = torch.where(torch.rand(S * N, device=dev, generator=gen) < pd_, 1.0, -1.0)
            rot = torch.where(zero, torch.zeros_like(mag), mag * dire).cpu().numpy().reshape(S, N)
        nz = rot != 0
        state["prev_dir"] = np.where(nz, np.sign(rot), state["prev_dir"]); state["prev_zero"] = (~nz).astype(float)
        return np.deg2rad(rot)
    return policy


def simulate(model, mu, sd, Zarr, speeds, seed):
    S, N, _ = Zarr.shape; gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed); env = Env(S, N, speeds, rng)
    T = 900; X = np.zeros((T, S, N)); Y = np.zeros((T, S, N)); Hh = np.zeros((T, S, N)); D = np.zeros((T, S, N))
    pol = make_policy(model, mu, sd, Zarr, gen)
    for t in range(T):
        env.step_resource(); o = env.observe(); rot = pol(env, o, t); d = env.apply(rot); X[t], Y[t], Hh[t], D[t] = env.x, env.y, env.h, d
    return group_metrics(dict(score=env.score, n_on=env.n_on, X=X, Y=Y, H=Hh, D=D))


rows = []; t0 = time.time()
for fo in range(sp["n_folds"]):
    ck = torch.load(EXP / f"fold{fo}.pt", map_location=dev, weights_only=False)
    model = Policy(len(ck["feats"]), ck["z_dim"], ck["hidden"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    Ztr = pd.DataFrame(ck["Ztr"], index=ck["train_pids"], columns=zcols).join(par[["cell"]])
    gs = [g for g in groups if fold_g[g] == fo]
    if not gs: continue
    for variant in ["main", "ctrl1_pop", "ctrl2_avg", "ctrl2b_gmean", "ctrl3_nominal"]:
        for n in (4, 5):
            gn = [g for g in gs if len(grp[grp.group == g]) == n]
            if not gn: continue
            if variant == "ctrl3_nominal":
                # each member alone: batch = seeds x (groups x members), N = 1
                mem = [(g, p) for g in gn for p in grp[grp.group == g].index]
                Zarr = np.stack([Z.loc[p, zcols].to_numpy(float) for g, p in mem for _ in range(a.seeds)])[:, None, :]
                speeds = np.array([0.4 if grp.loc[p, "speed"] == "slow" else 0.8 for g, p in mem for _ in range(a.seeds)])
                M = simulate(model, ck["mu"], ck["sd"], Zarr, speeds, seed=50_000 + 100 * fo + n)
                k = 0
                for g, p in mem:
                    for sd in range(a.seeds):
                        rows.append(dict(tag=f"{g}|ctrl3_nominal", group=g, variant="ctrl3_nominal", draw=int(np.where(grp[grp.group == g].index == p)[0][0]), speed=float(speeds[k]), n=1, seed=sd,
                                         **{m: float(M[m][k]) for m in ("score", "tracking_time", "pcr", "med_dist")})); k += 1
                continue
            Zl, speeds, meta = [], [], []
            for g in gn:
                mem = grp[grp.group == g]; cell = mem.cell.iloc[0]; spd = 0.4 if mem.speed.iloc[0] == "slow" else 0.8
                zm = Z.loc[mem.index, zcols].to_numpy(float)
                if variant == "main": zg = zm
                elif variant == "ctrl1_pop": zg = np.zeros_like(zm)
                elif variant == "ctrl2_avg": zg = np.tile(Ztr[Ztr.cell == cell][zcols].mean().to_numpy(float), (n, 1))
                elif variant == "ctrl2b_gmean": zg = np.tile(zm.mean(0), (n, 1))
                for sd in range(a.seeds): Zl.append(zg); speeds.append(spd); meta.append((g, sd))
            Zarr = np.stack(Zl); M = simulate(model, ck["mu"], ck["sd"], Zarr, np.array(speeds), seed=10_000 + 100 * fo + n)
            for k, (g, sd) in enumerate(meta):
                rows.append(dict(tag=f"{g}|{variant}", group=g, variant=variant, draw=0, speed=float(speeds[k]), n=n, seed=sd, **{m: float(M[m][k]) for m in M}))
        print(f"fold {fo} {variant} done {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT / "sim_runs_d2.parquet", index=False)
json.dump(dict(n_runs=len(df), wall_s=time.time() - t0, seeds=a.seeds), open(OUT / "sim_budget_d2.json", "w"), indent=1)
print("done", len(df), f"{time.time()-t0:.0f}s")
