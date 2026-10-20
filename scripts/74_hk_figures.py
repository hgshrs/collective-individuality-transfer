"""Diagnostic figures for the Hawkins 2023 replication (results/hawkins2023/fig_*)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
P = Path('/home/higashi/Research/20260924_collective'); R = P / 'results/hawkins2023'
tr = pd.read_csv(R / 'transfer_r.csv')
fig, axes = plt.subplots(2, 1, figsize=(13, 9))
for ax, e in zip(axes, [1, 3]):
    d = tr[tr.experiment == e].pivot_table(index='target', columns='source', values='r'); ceil = tr[tr.experiment == e].groupby('target').noise_ceiling.first().reindex(d.index)
    x = np.arange(len(d)); srcs = [s for s in ['score1', 'z4', 'z4+teacher1', 'search_stats_ridge', 'waiting'] if s in d]; w = 0.8 / len(srcs)
    for i, s in enumerate(srcs): ax.bar(x + (i - (len(srcs) - 1) / 2) * w, d[s], w, label={'score1': 'first-half score only', 'z4': 'first-half search z (PCA4)', 'z4+teacher1': 'z + first-half teacher weights', 'search_stats_ridge': 'search stats ridge', 'waiting': 'waiting-room (solo) stats'}[s])
    ax.plot(x, ceil, 'k_', ms=12, mew=2, label='first-vs-second-half ceiling'); ax.axhline(0, color='k', lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=60, ha='right', fontsize=8); ax.set_ylabel('out-of-fold r (within exp x size x field)'); ax.set_title(f'Hawkins 2023 Exp {e}: first-half search phase -> second-half targets', fontsize=10); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(R / 'fig_transfer.png', dpi=150); fig.savefig(R / 'fig_transfer.pdf')
gr = pd.read_csv(R / 'group_table.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (xc, yc, lab) in zip(axes, [('score1_raw', 'score2_raw', 'raw: members mean first-half score vs second-half group score'), ('mean_score1', 'score2', 'within exp x size x field (residualised)')]):
    for e, col in [(1, 'tab:blue'), (3, 'tab:red')]:
        d = gr[gr.experiment == e]; ax.scatter(d[xc], d[yc], s=10 + 6 * d.n_players, alpha=.5, color=col, label=f'exp {e}')
    ax.set_title(f'{lab}: r = {np.corrcoef(gr[xc], gr[yc])[0, 1]:.2f}', fontsize=8); ax.set_xlabel(xc); ax.set_ylabel(yc)
axes[0].legend(); fig.tight_layout(); fig.savefig(R / 'fig_group_pred.png', dpi=150); fig.savefig(R / 'fig_group_pred.pdf'); print('figures written')
