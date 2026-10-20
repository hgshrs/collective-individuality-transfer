"""Vectorised re-implementation of the paper's resource-tracking environment (external/analysis/agent_based_model/abm.py)
for Phase 3 D-2 (neural policies cannot run inside the mesa ABM). Batch of S independent simulations x N agents.

Differences from the mesa ABM (validated statistically against it in scripts/50_env_validate.py):
  * agents observe all other agents' positions synchronously (mesa BaseScheduler updates them sequentially);
  * everything else (Lévy-walk resource, boundaries, detector, FOV 90°, social quality, ABM decision rule, noise,
    persistence, scoring) follows abm.py line by line.
Conventions: headings in radians, positive rotation = counter-clockwise; the social angle alpha is the bearing of the
visible others relative to the heading (positive = they are to the counter-clockwise side), so turning toward them means
a rotation with the sign of alpha.
"""
import numpy as np

W = H = 400.0; MARGIN = 10.0; AGENT_SPEED = 7.0; FOV_HALF = np.pi / 4; RADIUS = 40.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Env:
    def __init__(self, S, N, speed, rng, resource_movement="random"):
        self.S, self.N, self.v, self.rng = S, N, speed, rng
        self.rx = np.full(S, W / 2); self.ry = np.full(S, H / 2); self.rrot = np.zeros(S); self.rstep = np.zeros(S)   # resource
        self.x = np.full((S, N), W / 2 - 10.0); self.y = np.full((S, N), H / 2); self.h = np.zeros((S, N))           # agents
        self.d_last = np.sqrt((self.x - self.rx[:, None]) ** 2 + (self.y - self.ry[:, None]) ** 2)
        self.det = np.zeros((S, N)); self.det_lag = np.zeros((S, N)); self.first = True
        self.dir = rng.choice([-1.0, 1.0], size=(S, N)); self.prev_rot = np.zeros((S, N))
        self.score = np.zeros((S, N)); self.n_on = np.zeros((S, N))
        self.v = np.broadcast_to(np.asarray(speed, float), (S,)).copy()                 # per-simulation resource speed allowed
        self.max_dd = (AGENT_SPEED + AGENT_SPEED * self.v)[:, None]
        self.resource_movement = resource_movement

    # ---------------- resource (ResourceAgent.step) ----------------
    def step_resource(self):
        if self.resource_movement == "random":
            sel = self.rstep < 0
            if sel.any():
                self.rstep[sel] = np.minimum(self.rng.pareto(1.0, sel.sum()) + 1, 100); self.rrot[sel] = self.rng.uniform(-np.pi, np.pi, sel.sum())
            self.rstep[~sel] -= 1
        sp = self.v * AGENT_SPEED
        nx = self.rx + sp * np.cos(self.rrot); ny = self.ry + sp * np.sin(self.rrot)
        self.rrot, self.rx, self.ry = self._bound(nx, ny, self.rrot)

    @staticmethod
    def _bound(x, y, r):
        near = (x < MARGIN) | (x > W - MARGIN) | (y < MARGIN) | (y > H - MARGIN)
        r = np.where(near, np.arctan2(H / 2 - y, W / 2 - x), r)
        return wrap(r), np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)

    # ---------------- observation (read_detector_value + observe_social_environment) ----------------
    def observe(self):
        d_new = np.sqrt((self.x - self.rx[:, None]) ** 2 + (self.y - self.ry[:, None]) ** 2)
        self.det_lag = self.det.copy()
        self.det = np.clip((d_new - self.d_last) / self.max_dd, -1, 1) if not self.first else np.zeros_like(d_new)
        self.dd = d_new - self.d_last                                  # raw distance change (m/s) for data-scale detector
        others_d_last = self.d_last                                    # others' distance_from_resource_last (previous step)
        self.d_last = d_new; self.first = False
        dx = self.x[:, None, :] - self.x[:, :, None]; dy = self.y[:, None, :] - self.y[:, :, None]       # [s, i, j] = pos_j - pos_i
        alpha = wrap(np.arctan2(dy, dx) - self.h[:, :, None])
        vis = np.abs(alpha) < FOV_HALF
        eye = np.eye(self.N, dtype=bool)[None]; vis &= ~eye
        n = vis.sum(2); has = n > 0; nn = np.maximum(n, 1)
        mean_alpha = (alpha * vis).sum(2) / nn
        od = np.broadcast_to(others_d_last[:, None, :], vis.shape)
        mean_od = (od * vis).sum(2) / nn; min_od = np.where(vis, od, np.inf).min(2)
        oh = np.broadcast_to(self.h[:, None, :], vis.shape)
        mean_oh = np.arctan2((np.sin(oh) * vis).sum(2), (np.cos(oh) * vis).sum(2))       # circular mean heading of visible others
        mov = wrap(self.h - mean_oh)
        tracking = d_new <= RADIUS
        q_abm = np.where(tracking, -np.clip((mean_od - d_new) / 40.0, -1, 1), np.where(mean_od < 40, 1.0, -1.0))
        q_best = np.where(tracking, -np.clip((min_od - d_new) / 40.0, -1, 1), np.where(min_od < 40, 1.0, -1.0))
        return dict(d=d_new, tracking=tracking, det=self.det, det_lag=self.det_lag, dd=self.dd, has=has, n_vis=n, alpha=np.where(has, mean_alpha, 0.0),
                    mov=np.where(has, mov, 0.0), q_abm=np.where(has, q_abm, 0.0), q_best=np.where(has, q_best, 0.0))

    # ---------------- ABM decision rule (SearcherAgent.tracking / searching / update_rotation) ----------------
    def abm_policy(self, o, I, S, k, x0, rmax, sigma, pers, p_s=0.15, p_u=0.35):
        """Parameters: arrays broadcastable to (S, N) (per agent) or scalars. S here is the slope on the ABM detector scale."""
        rng = self.rng; sh = (self.S, self.N)
        p_soc = np.where(o["has"], 1 / (1 + np.exp(-k * (o["q_abm"] - x0))), 0.0)
        use_soc = rng.random(sh) < p_soc
        mag = np.where(use_soc, np.abs(o["alpha"]), 0.0); dire = np.where(use_soc, np.sign(o["alpha"]), self.dir)
        # private tracking
        priv_t = ~use_soc & o["tracking"]
        mag = np.where(priv_t, np.deg2rad(I + S * o["det"]), mag)
        thr = np.where(o["det"] > o["det_lag"], p_u, p_s); flip = priv_t & (rng.random(sh) < thr)
        dire = np.where(flip, -dire, dire)
        # private searching
        priv_s = ~use_soc & ~o["tracking"]
        mag = np.where(priv_s, rng.uniform(0, 1, sh) * np.deg2rad(rmax), mag); dire = np.where(priv_s, rng.choice([-1.0, 1.0], size=sh), dire)
        mag = np.where(rng.random(sh) < pers, 0.0, mag)
        mag = mag + rng.normal(0, np.deg2rad(sigma), sh)
        self.dir = dire
        return mag * dire                                               # signed rotation (rad)

    def apply(self, rot):
        self.h = wrap(self.h + rot); self.prev_rot = rot
        nx = self.x + AGENT_SPEED * np.cos(self.h); ny = self.y + AGENT_SPEED * np.sin(self.h)
        self.h, self.x, self.y = self._bound(nx, ny, self.h)
        d = np.sqrt((self.x - self.rx[:, None]) ** 2 + (self.y - self.ry[:, None]) ** 2)
        on = d <= RADIUS; self.score += np.where(on, RADIUS - d, 0.0); self.n_on += on
        return d


def run(S, N, speed, policy, T=900, seed=0, record=True, resource_movement="random"):
    """policy(env, obs, t) -> signed rotation (rad) array (S, N). Returns metrics dict (+ trajectories if record)."""
    rng = np.random.default_rng(seed); env = Env(S, N, speed, rng, resource_movement)
    X = np.zeros((T, S, N)); Y = np.zeros((T, S, N)); Hh = np.zeros((T, S, N)); D = np.zeros((T, S, N))
    for t in range(T):
        env.step_resource(); o = env.observe(); rot = policy(env, o, t); d = env.apply(rot)
        if record: X[t], Y[t], Hh[t], D[t] = env.x, env.y, env.h, d
    out = dict(score=env.score, n_on=env.n_on, tracking_time=env.n_on / T)
    if record: out.update(X=X, Y=Y, H=Hh, D=D)
    return out


def group_metrics(res, T=900):
    """Collective metrics per simulation from run() output (same definitions as scripts/41): returns dict of arrays (S,)."""
    sc, ton = res["score"], res["n_on"]; S, N = sc.shape
    out = dict(score=sc.mean(1), tracking_time=(ton / T).mean(1), pcr=np.nanmean(sc / np.where(ton > 0, ton, np.nan), 1), med_dist=np.median(np.median(res["D"], 0), 1))
    if N > 1:
        X, Y, Hh, D = res["X"], res["Y"], res["H"], res["D"]
        dx = X[..., None, :] - X[..., :, None]; dy = Y[..., None, :] - Y[..., :, None]
        ang = wrap(np.arctan2(dy, dx) - Hh[..., :, None]); see = np.abs(ang) < FOV_HALF
        see &= ~np.eye(N, dtype=bool)[None, None]
        outd = see.sum(-1).astype(float); ind = see.sum(-2).astype(float)       # (T, S, N)
        mi, mo = ind.mean(0), outd.mean(0)
        def corr_rows(a, b):
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True); den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
            return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)
        out.update(in_deg=mi.mean(1), out_deg=mo.mean(1), in_out_corr=corr_rows(mi, mo))
        on = D <= RADIUS; allsearch = ~on.any(2)                                  # (T, S)
        gains = [[] for _ in range(S)]
        one = on.sum(2) == 1
        for t in range(2, T - 11):
            cand = allsearch[t - 2] & allsearch[t - 1] & one[t]
            for s in np.where(cand)[0]:
                i = int(np.argmax(on[t, s]))
                if on[t + 1, s, i]:
                    gains[s].append(ind[t + 1:t + 11, s, i].mean() - ind[max(0, t - 10):t, s, i].mean())
        out["disc_gain"] = np.array([np.mean(g) if g else np.nan for g in gains]); out["n_disc"] = np.array([len(g) for g in gains])
    return out
