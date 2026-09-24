"""§4.2-6: simulated individuals. Run the paper's ABM (unmodified, external/) with 100 groups of 5 heterogeneous agents
(known β1 = detector_sensitivity_slope, k = selectivity slope, r_max = search_max_rotation, x0 = quality intercept),
log 1 Hz trajectories, convert to the encoder's feature schema and save data/processed/sim_agents_features.parquet
+ sim_agents_params.csv. Encoding/recovery is done in 14_phase1_sim_recovery.py after training.
NOTE: the paper ABM has a single social alignment (position); 'position weight'/'movement weight' of the spec are not
separate parameters there, so the recoverable parameter set is (β1, k, r_max, x0)."""
import sys, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collective_id.paths import EXTERNAL, PROCESSED
from collective_id.abm_features import features_from_trajectories
sys.path.insert(0, str(EXTERNAL))
from agent_based_model.abm import SearcherResourceModel
BASE = dict(width=400, height=400, resource_movement='random', noize_degree=10, p_s=0.15, p_u=0.35,
            persistence=0, p_perceive_social_info_quality=1, social_alignment='position', social_info_quality_func='sigmoid')

def run_group(args):
    g, seed, speed = args
    rng = np.random.default_rng(seed); np.random.seed(seed)
    beta1 = rng.uniform(10, 80, 5); k = 10 ** rng.uniform(-1, 1, 5); rmax = rng.uniform(30, 150, 5); x0 = rng.uniform(-0.5, 0.5, 5)
    m = SearcherResourceModel(N=6, resource_speed=speed, detector_sensitivity_slope=list(beta1), social_info_quality_sensitivity_slope=list(k),
                              search_max_rotation=list(rmax), social_info_quality_sensitivity_intercept=list(x0),
                              rotation_tracking_intercept=[float(max(b, 50)) for b in beta1], **BASE)
    for _ in range(900): m.step()
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    res = df[df.AgentID == 0].sort_values("Step"); ag = df[df.AgentID != 0]
    T = int(df.Step.max()); ids = sorted(ag.AgentID.unique())
    X = np.stack([ag[ag.AgentID == i].sort_values("Step").x.to_numpy() for i in ids], 1); Y = np.stack([ag[ag.AgentID == i].sort_values("Step").y.to_numpy() for i in ids], 1)
    TH = np.stack([ag[ag.AgentID == i].sort_values("Step").rotation.to_numpy() for i in ids], 1)
    feats = features_from_trajectories(X, Y, TH, res.x.to_numpy(), res.y.to_numpy(), [f"sim{g:03d}_{i}" for i in ids], f"simG{g:03d}", cond="FP", speed="slow" if speed == 0.4 else "fast")
    params = pd.DataFrame(dict(participant=[f"sim{g:03d}_{i}" for i in ids], group=f"simG{g:03d}", beta1=beta1, k=k, log_k=np.log10(k), rmax=rmax, x0=x0, speed=speed))
    return feats, params

if __name__ == "__main__":
    jobs = [(g, 5000 + g, 0.4 if g % 2 == 0 else 0.8) for g in range(100)]
    t0 = time.time(); F, Pp = [], []
    with ProcessPoolExecutor(4) as ex:
        for f, p in ex.map(run_group, jobs): F.append(f); Pp.append(p)
    F = pd.concat(F, ignore_index=True); Pp = pd.concat(Pp, ignore_index=True)
    F.to_parquet(PROCESSED / "sim_agents_features.parquet", index=False); Pp.to_csv(PROCESSED / "sim_agents_params.csv", index=False)
    print(F.shape, Pp.shape, f"{time.time()-t0:.0f}s"); print(F.state.value_counts(normalize=True).round(3).to_dict()); print(F[["rot_abs", "soc_pos", "soc_mov", "n_vis", "dist_search"]].describe().T[["mean", "std", "min", "max"]])
