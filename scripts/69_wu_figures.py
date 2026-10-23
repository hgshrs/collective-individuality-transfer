"""Diagnostic figures for the Wu et al. (2025) replication (PNG + PDF, results/wu2025/fig_*)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
P = Path(__file__).resolve().parents[1]; R = P / 'results/wu2025'
# 1. transfer r vs noise ceiling
tr = pd.read_csv(R / 'transfer_r.csv', index_col=0)
sel = [t for t in tr.index if not t.startswith('stan_')]
tr = tr.loc[sel]
fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(tr)); w = 0.18
for i, (c, lab) in enumerate([('solo_reward', 'solo reward only'), ('solo_z4', 'solo z (PCA4)'), ('solo_z4+choice4', 'solo z + solo choice weights'), ('solo_stats_ridge', 'solo stats ridge')]):
    ax.bar(x + (i - 1.5) * w, tr[c], w, label=lab)
ax.plot(x, tr['noise_ceiling'], 'k_', ms=14, mew=2, label='split-half ceiling')
ax.axhline(0, color='k', lw=.5); ax.set_xticks(x); ax.set_xticklabels(tr.index, rotation=70, ha='right', fontsize=8); ax.set_ylabel('out-of-fold r (8 folds of 4 sessions)')
ax.set_title('Wu 2025: solo rounds -> group-round behaviour and choice weights'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(R / 'fig_transfer.png', dpi=150); fig.savefig(R / 'fig_transfer.pdf')
# 2. group prediction: nominal and ridge vs observed
gr = pd.read_csv(R / 'group_rounds.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (xc, lab) in zip(axes, [('nominal_reward_z', "members' mean solo reward (zero-shot)"), ('mean_z0', 'members mean solo z0')]):
    for e, col in [('random', 'tab:orange'), ('smooth', 'tab:green')]:
        d = gr[gr.env == e]; ax.scatter(d[xc], d['total_reward_z'], s=14, alpha=.6, color=col, label=e)
    r = np.corrcoef(gr[xc], gr['total_reward_z'])[0, 1]; ax.set_title(f'{lab}: r = {r:.2f}', fontsize=9); ax.set_xlabel(xc); ax.set_ylabel('group reward (z within env)')
axes[0].legend(); fig.tight_layout(); fig.savefig(R / 'fig_group_pred.png', dpi=150); fig.savefig(R / 'fig_group_pred.pdf')
# 3. simulation: main vs observed, and controls
try:
    se = pd.read_csv(R / 'sim_eval.csv'); sim = pd.read_parquet(R / 'sim_runs.parquet')
    obs = gr.groupby(['session', 'env'])[['total_reward', 'nn_dist_mean']].mean()
    agg = sim[sim.variant == 'main'].groupby(['session', 'env'])[['total_reward', 'nn_dist']].mean().reindex(obs.index)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (oc, sc) in zip(axes[:2], [('total_reward', 'total_reward'), ('nn_dist_mean', 'nn_dist')]):
        for e, col in [('random', 'tab:orange'), ('smooth', 'tab:green')]:
            m = obs.index.get_level_values('env') == e
            ax.scatter(agg[sc][m], obs[oc][m], s=18, alpha=.7, color=col, label=e)
        rr = se[(se.observed == oc) & (se.simulated == sc) & (se.variant == 'main')].iloc[0]
        ax.set_title(f'{oc}: r within env = {rr.r:.2f} [{rr.ci_lo:.2f}, {rr.ci_hi:.2f}]', fontsize=9); ax.set_xlabel('simulated (main, seed mean)'); ax.set_ylabel('observed (mean of 4 group rounds)')
    axes[0].legend()
    piv = se[se.simulated.isin(['total_reward', 'nn_dist'])].pivot(index='observed', columns='variant', values='r')[['main', 'gmean', 'pop', 'nominal', 'shuffle']]
    piv.plot.bar(ax=axes[2], width=.8); axes[2].axhline(0, color='k', lw=.5); axes[2].set_ylabel('r within env'); axes[2].set_title('main vs controls', fontsize=9); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(R / 'fig_sim_pred_vs_obs.png', dpi=150); fig.savefig(R / 'fig_sim_pred_vs_obs.pdf')
except FileNotFoundError as e:
    print('skip sim figure', e)
print('figures written')
