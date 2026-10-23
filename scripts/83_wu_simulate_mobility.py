"""Wu et al. (2025) grid ABM with a per-agent MOBILITY parameter (HANDOVER §12 item 4).

The authors' ABM moves every agent exactly one cell per step, so individual differences in movement rate (the strongest
solo predictor of group reward in the real data) cannot enter the simulation. Here each agent i moves on a step with
probability p_i and otherwise stays (navigation and mining are unchanged). p_i is calibrated zero-shot from the
participant's SOLO rounds: p_i = mean solo path length / max over participants (0.28-1.00); nothing is fitted to group play.
Weights as in scripts/66 (w_pop[env] + v_i, group-round MAP deviations, 5 non-adaptive weights).
Variants per session x env (same 4 env files x 25 seeds as scripts/66):
  main_mob   members' own weights + own p_i
  gmean_mob  all four agents = mean weights + mean p_i of the members (composition removed)
  pop_mob    population weights + members' own p_i (isolates mobility)
  nominal_mob each member alone with own weights + own p_i
Output: results/wu2025/sim_runs_mobility.parquet, sim_mobility_params.csv
Usage: OMP_NUM_THREADS=1 uv run python scripts/83_wu_simulate_mobility.py [--procs 16] [--seeds 25]
"""
import sys, json, argparse, copy, time
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
P = Path(__file__).resolve().parents[1]; D = P / 'data/processed/wu2025'; R = P / 'results/wu2025'
sys.path.insert(0, str(P / 'external/wu2025/simulations')); sys.path.insert(0, str(P / 'scripts'))
from task import State, AgentView
from simulation import Simulation, SimulationResult
import importlib.util
spec = importlib.util.spec_from_file_location('s66', P / 'scripts/66_wu_simulate.py'); s66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s66)
StdPolicy, W_POP, GW5, ENV = s66.StdPolicy, s66.W_POP, s66.GW5, s66.ENV
np.seterr(all='ignore')


class MobilitySimulation(Simulation):
    """Authors' Simulation.run with one change: agent i executes its move on a step with probability p_move[i]."""
    p_move: np.ndarray = None

    def run(self, seed=None, sim_slice_rate=None):
        rng = np.random.Generator(np.random.PCG64(seed))
        state = copy.deepcopy(self.initial_state)
        agent_views = [AgentView.from_state(state, i) for i in range(state.n_agents)]
        agent_states = [('choose_cell', None) for _ in range(state.n_agents)]
        agent_moves = np.zeros((2, state.n_agents), dtype=int)
        agent_mine_actions = np.zeros(state.n_agents, dtype=bool)
        stats = []
        for step in range(self.max_steps):
            agent_mine_actions[:] = False
            moving = rng.random(state.n_agents) < self.p_move
            for i in range(state.n_agents):
                agent_view = agent_views[i]; policy = self.agent_policies[i]; agent_state = agent_states[i]
                cur_cell = state.agent_locs[:, i]
                if agent_state[0] == 'choose_cell':
                    goal_cell = policy.sample_next_cell(agent_view, state, rng=rng)
                    agent_states[i] = ('navigate_to', goal_cell); agent_view.clear_agent_info()
                elif agent_state[0] == 'navigate_to':
                    goal_cell = agent_state[1]
                    if np.all(goal_cell == cur_cell):
                        agent_states[i] = ('mine_cell', self.mining_steps) if not state.revealed_map[tuple(cur_cell)] else ('choose_cell', None)
                        goal_cell = None
                elif agent_state[0] == 'mine_cell':
                    goal_cell = None
                    if agent_state[1] == 0:
                        agent_states[i] = ('choose_cell', None); agent_mine_actions[i] = True
                    else:
                        agent_states[i] = ('mine_cell', agent_state[1] - 1)
                if goal_cell is not None and moving[i]:
                    if goal_cell[0] > cur_cell[0]: mv = (1, 0)
                    elif goal_cell[0] < cur_cell[0]: mv = (-1, 0)
                    elif goal_cell[1] > cur_cell[1]: mv = (0, 1)
                    elif goal_cell[1] < cur_cell[1]: mv = (0, -1)
                    else: mv = (0, 0)
                    agent_moves[:, i] = mv
                else:
                    agent_moves[:, i] = 0
            state.update(agent_moves, agent_mine_actions)
            for i in range(state.n_agents): agent_views[i].update(state)
            stats.extend(self.step_stats(step, state, agent_views, agent_states))
            if np.all(state.revealed_map): break
        return SimulationResult(stats=stats, sim_slices=[], init_state=self.initial_state)


def run_one(args):
    tag, session, env, variant, env_file, seed, weights, pm = args
    n = len(weights)
    st = State.from_env_file(str(P / 'external/wu2025/environments' / env_file), seed=seed, n_agents=n)
    sim = MobilitySimulation(agent_policies=[StdPolicy(w) for w in weights], initial_state=st, max_steps=600, mining_steps=9)
    sim.p_move = np.asarray(pm, float)
    res = sim.run(seed=seed + 1000)
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
                total_reward=float(rew.sum()), reward_sd=float(rew.std()) if n > 1 else np.nan, pair_dist=pair, nn_dist=nn, revealed=int(df['revealed'].max()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--seeds', type=int, default=25); a = ap.parse_args()
    rounds = pd.read_parquet(D / 'rounds.parquet'); cw = pd.read_csv(R / 'choice_weights.csv'); feats = pd.read_parquet(D / 'round_features.parquet')
    v = cw[(cw.type == 'group') & (cw.split == 'all')].pivot(index='pid', columns='param', values='value')[GW5]
    solo_path = feats[feats.type == 'solo'].groupby('pid')['path_len'].mean()
    p_move = (solo_path / solo_path.max()).clip(0.05, 1.0)
    pd.DataFrame({'solo_path_len': solo_path, 'p_move': p_move}).to_csv(R / 'sim_mobility_params.csv')
    pids = v.index.to_numpy(); sess_of = {p: int(p.split('session')[1].split('.')[0]) for p in pids}
    rng = np.random.default_rng(0); jobs = []
    for s in range(1, 33):
        members = [p for p in pids if sess_of[p] == s]; assert len(members) == 4
        for env in ['random', 'smooth']:
            e = ENV[env]
            files = rounds[(rounds.session == s) & (rounds.type == 'group') & (rounds.env == env)]['env_file'].tolist()
            wm = {p: W_POP[:, e] + v.loc[p].to_numpy() for p in members}
            W_main = [wm[p] for p in members]; pm = [float(p_move[p]) for p in members]
            W_gmean = [np.mean(W_main, 0)] * 4; pm_g = [float(np.mean(pm))] * 4; W_pop = [W_POP[:, e]] * 4
            for f in files:
                for r in range(a.seeds):
                    seed = int(rng.integers(1e9))
                    jobs.append(('main_mob', s, env, 'main_mob', f, seed, W_main, pm))
                    jobs.append(('gmean_mob', s, env, 'gmean_mob', f, seed, W_gmean, pm_g))
                    jobs.append(('pop_mob', s, env, 'pop_mob', f, seed, W_pop, pm))
                    for p in members: jobs.append((p, s, env, 'nominal_mob', f, seed, [wm[p]], [float(p_move[p])]))
    print('jobs', len(jobs), flush=True); t0 = time.time()
    with Pool(a.procs) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=40)):
            out.append(r)
            if i % 2000 == 0: print(i, f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(out); df.to_parquet(R / 'sim_runs_mobility.parquet', index=False)
    json.dump(dict(n_runs=len(df), wall_s=time.time() - t0, seeds=a.seeds, procs=a.procs), open(R / 'sim_budget_mobility.json', 'w'), indent=1)
    print(df.groupby(['variant', 'env'])[['total_reward', 'pair_dist', 'revealed']].mean().round(2))


if __name__ == '__main__':
    main()
