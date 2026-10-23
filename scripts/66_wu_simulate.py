"""Phase-3 analogue on Wu et al. (2025): zero-shot simulation of each real group with individually fitted policies.

Environment: the authors' simplified grid ABM (external/wu2025/simulations; 20 x 20 cells, 1 cell/step, 9 mining steps,
600 steps). Policy: softmax over unrevealed cells of standardised features (distance, in-FOV, GP reward prediction,
distance to successful visible players' centroid, distance to visible players' centroid) with weights
w = w_pop[env] + v_i, where v_i are the participant's MAP deviations fitted to their own group-round choices
(scripts/62; non-adaptive 5 weights only; adaptive terms dropped). Nothing is fitted to group outcomes.
Variants per session x env: main (members' own weights), gmean (all four = mean of members' weights),
pop (population weights), nominal (each member alone), shuffle (four random participants from other sessions).
Each session x env uses the four env files actually played in that session's group rounds x 25 seeds.
Outputs: results/wu2025/sim_runs.parquet
"""
import sys, json, itertools
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
P = Path(__file__).resolve().parents[1]; D = P / 'data/processed/wu2025'; R = P / 'results/wu2025'
sys.path.insert(0, str(P / 'external/wu2025/simulations'))
from task import State, AgentView, is_visible, distance, gaussian_process_logits
from simulation import Simulation
np.seterr(all='ignore')
ENV = {'random': 0, 'smooth': 1}
pop = json.load(open(R / 'stan_population.json'))['conditional_ARS']
W_POP = np.array(pop['weights'])  # 5 x 2
GW5 = ['Locality', 'BlockVisibility', 'RewardPrediction', 'SuccessfulProximity', 'UnsuccessfulProximity']
LS, SV = np.sqrt(48) / 3.0, 10.0  # GP lengthscale in cell units (blocks are 3 units apart), signal variance as in the authors' features

class StdPolicy:
    def __init__(self, w):
        self.w = np.asarray(w, float)
    def next_cell_logits(self, agent_view, state):
        ax, ay = agent_view.agent_locs[:, agent_view.agent_i]; dx, dy = agent_view.agent_dirs[:, agent_view.agent_i]
        cx, cy = np.where(state.revealed_map == False)
        X = np.zeros((len(cx), 5))
        X[:, 0] = distance(ax, ay, cx, cy)
        X[:, 1] = is_visible(ax, ay, dx, dy, cx, cy).astype(float)
        obs_cx, obs_cy = np.where(~np.isnan(agent_view.agent_rewards_map))
        if len(obs_cx) > 0:
            rewarded = agent_view.agent_rewards_map[(obs_cx, obs_cy)] == 1
            X[:, 2] = gaussian_process_logits(cx=obs_cx, cy=obs_cy, r=rewarded, ncx=cx, ncy=cy, lengthscale=LS, signalvariance=SV)
        others = np.arange(agent_view.agent_locs.shape[1]) != agent_view.agent_i
        locs = agent_view.agent_locs
        known = others & ~np.isnan(locs[0])
        if known.any():
            in_view = np.zeros(len(known), bool); in_view[known] = is_visible(ax, ay, dx, dy, locs[0, known], locs[1, known])
            succ = in_view & agent_view.rewarded_agents
            if succ.any():
                c = locs[:, succ].mean(1); X[:, 3] = distance(c[0], c[1], cx, cy)
            if in_view.any():
                c = locs[:, in_view].mean(1); X[:, 4] = distance(c[0], c[1], cx, cy)
        for j in range(5):
            sd = X[:, j].std(ddof=1) if len(cx) > 1 else 0
            X[:, j] = (X[:, j] - X[:, j].mean()) / sd if sd > 0 else 0.0
        return cx, cy, X @ self.w
    def sample_next_cell(self, agent_view, state, rng=None):
        cx, cy, logits = self.next_cell_logits(agent_view, state)
        p = np.exp(logits - logits.max()); p /= p.sum()
        i = rng.choice(len(p), p=p); return cx[i], cy[i]

def run_one(args):
    tag, session, env, variant, env_file, seed, weights = args
    n = len(weights)
    st = State.from_env_file(str(P / 'external/wu2025/environments' / env_file), seed=seed, n_agents=n)
    sim = Simulation(agent_policies=[StdPolicy(w) for w in weights], initial_state=st, max_steps=600, mining_steps=9)
    res = sim.run(seed=seed + 1000, sim_slice_rate=None)
    df = pd.DataFrame(res.stats)
    rew = df.groupby('agent')['reward'].sum().to_numpy()
    xy = df.pivot(index='step', columns='agent', values=['x', 'y'])
    if n > 1:
        xs, ys = xy['x'].to_numpy(), xy['y'].to_numpy()
        dmat = np.sqrt((xs[:, :, None] - xs[:, None, :]) ** 2 + (ys[:, :, None] - ys[:, None, :]) ** 2)
        iu = np.triu_indices(n, 1); pair = dmat[:, iu[0], iu[1]].mean()
        nn = np.where(np.eye(n, dtype=bool)[None], np.inf, dmat).min(2).mean()
    else:
        pair = np.nan; nn = np.nan
    return dict(tag=tag, session=session, env=env, variant=variant, env_file=env_file, seed=seed, n_agents=n,
                total_reward=float(rew.sum()), reward_sd=float(rew.std()) if n > 1 else np.nan, pair_dist=pair, nn_dist=nn,
                revealed=int(df['revealed'].max()))

def main():
    rounds = pd.read_parquet(D / 'rounds.parquet'); cw = pd.read_csv(R / 'choice_weights.csv')
    v = cw[(cw.type == 'group') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value')[GW5]
    pids = v.index.to_numpy(); sess_of = {p: int(p.split('session')[1].split('.')[0]) for p in pids}
    rng = np.random.default_rng(0)
    jobs = []
    for s in range(1, 33):
        members = [p for p in pids if sess_of[p] == s]
        assert len(members) == 4
        for env in ['random', 'smooth']:
            e = ENV[env]
            files = rounds[(rounds.session == s) & (rounds.type == 'group') & (rounds.env == env)]['env_file'].tolist()
            wm = {p: W_POP[:, e] + v.loc[p].to_numpy() for p in members}
            W_main = [wm[p] for p in members]; W_gmean = [np.mean(W_main, 0)] * 4; W_pop = [W_POP[:, e]] * 4
            others = [p for p in pids if sess_of[p] != s]
            for k, f in enumerate(files):
                for r in range(25):
                    seed = int(rng.integers(1e9))
                    jobs.append(('main', s, env, 'main', f, seed, W_main))
                    jobs.append(('gmean', s, env, 'gmean', f, seed, W_gmean))
                    jobs.append(('pop', s, env, 'pop', f, seed, W_pop))
                    for p in members:
                        jobs.append((p, s, env, 'nominal', f, seed, [wm[p]]))
                    if r < 12:
                        sh = rng.choice(others, 4, replace=False)
                        jobs.append((','.join(sh), s, env, 'shuffle', f, seed, [W_POP[:, e] + v.loc[p].to_numpy() for p in sh]))
    print('jobs', len(jobs), flush=True)
    with Pool(20) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=50)):
            out.append(r)
            if i % 5000 == 0: print(i, flush=True)
    df = pd.DataFrame(out); df.to_parquet(R / 'sim_runs.parquet', index=False)
    print(df.groupby(['variant', 'env'])[['total_reward', 'pair_dist', 'revealed']].mean().round(2))

if __name__ == '__main__':
    main()
