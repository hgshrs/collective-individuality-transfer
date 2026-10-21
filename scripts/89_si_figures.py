"""Supporting-information figure panels (S1-S5 Fig) for PLOS Computational Biology, same conventions as scripts/80:
one panel = one PDF, usetex Helvetica (sfmath), tab10, no grid, no titles, transparent, bbox tight; fonts: ticks/legend/
annotations 8 pt, axis labels 9 pt. Everything is read from existing result files (no re-analysis).
Output: results/paper_figures/si/*.pdf (+ .png previews) and si/captions.txt (one line per file, appended to INDEX.md).
Usage: uv run python scripts/89_si_figures.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
P = Path('/home/higashi/Research/20260924_collective'); R = P / 'results'; OUT = R / 'paper_figures' / 'si'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
                     'text.latex.preamble': r'\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}',
                     'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'savefig.transparent': True, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = plt.get_cmap('tab10').colors
caps = []
def save(fig, name, caption):
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=220, transparent=False, facecolor='white'); plt.close(fig); caps.append(f'si/{name}.pdf\t{caption}')
def save_legend(handles, name, caption, ncol=1):
    fig = plt.figure(figsize=(3.4, 0.6)); leg = fig.legend(handles=handles, loc='center', frameon=False, ncol=ncol, handlelength=1.2, handletextpad=0.5, columnspacing=1.2)
    fig.canvas.draw(); bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT / f'{name}.pdf', bbox_inches=bb.expanded(1.04, 1.15)); fig.savefig(OUT / f'{name}.png', dpi=220, transparent=False, facecolor='white', bbox_inches=bb.expanded(1.04, 1.15)); plt.close(fig)
    caps.append(f'si/{name}.pdf\t{caption} (legend only; saved with ~2 % padding, no further trim needed)')
def zcell(s, key): return s.groupby(key).transform(lambda v: (v - v.mean()) / v.std(ddof=0))
METRICS = ['score', 'tracking_time', 'pcr', 'med_dist', 'in_deg', 'out_deg', 'in_out_corr', 'disc_gain']
MNAME2 = {'score': 'Score', 'tracking_time': 'Tracking\ntime', 'pcr': 'Efficiency', 'med_dist': 'Median\ndistance', 'in_deg': 'In-degree', 'out_deg': 'Out-degree', 'in_out_corr': 'In--out\ncorrelation', 'disc_gain': 'Discoverer\ngain'}
MNAME = {'score': 'Score', 'tracking_time': 'Tracking time', 'pcr': 'Efficiency', 'med_dist': 'Median distance', 'in_deg': 'In-degree', 'out_deg': 'Out-degree', 'in_out_corr': 'In--out correlation', 'disc_gain': 'Discoverer gain'}
VNAME = {'main': "members' own policies", 'ctrl2b_gmean': 'group-mean agents', 'ctrl3_nominal': 'each member alone', 'ctrl4b_mps': 'mean-preserving shuffle', 'ctrl4_shuffle': 'random shuffle',
         'ctrl1_paper': 'paper default parameters', 'ctrl2_avg': 'training-cell mean', 'ctrl1_pop': 'population policy ($z = 0$)'}
obs = pd.read_csv(R / 'phase3/observed_group_metrics.csv').set_index('group')

# ---------------- S1 / S2: Chirkov D-1 and D-2 ----------------
def chirkov_set(prefix, sim_file, eval_file, variants, label):
    sim = pd.read_parquet(R / 'phase3' / sim_file); ev = pd.read_csv(R / 'phase3' / eval_file).set_index(['variant', 'metric'])
    pm = sim[sim.variant == 'main'].groupby('group')[METRICS].mean()
    for i, met in enumerate(METRICS):
        d = obs[[met, 'cell', 'speed']].join(pm[met].rename('sim')).dropna(); d['oz'] = zcell(d[met], d['cell']); d['sz'] = zcell(d['sim'], d['cell'])
        fig, ax = plt.subplots(figsize=(1.55, 1.45))
        for sp, col in [('slow', C[0]), ('fast', C[1])]:
            m = d.speed == sp; ax.plot(d.sz[m], d.oz[m], 'o', ms=3, color=col, alpha=0.75, mew=0)
        e = ev.loc[('main', met)]; ax.text(0.03, 0.97, f'{MNAME[met]}\n$r = {e.r_within_cell:.2f}$ [${e.ci_lo:.2f}$, ${e.ci_hi:.2f}$]', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('Predicted ($z$)'); ax.set_ylabel('Observed ($z$)'); ax.set_aspect('equal'); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        save(fig, f'{prefix}_{"abcdefgh"[i]}_{met}', f'{label}: observed vs predicted (seed mean of the members\'-own-policies simulation, leave-one-group-out) {MNAME[met]}, z within condition x speed; blue slow, orange fast; r [95 % CI] from {eval_file}; n = {int(e.n)} groups. Sources: results/phase3/{sim_file}, observed_group_metrics.csv, {eval_file}.')
    ev2 = ev.reset_index(); piv = ev2[ev2.variant.isin(variants)].pivot(index='metric', columns='variant', values='r_within_cell').reindex(METRICS)[variants]
    fig, ax = plt.subplots(figsize=(6.6, 1.9)); x = np.arange(len(METRICS)); w = 0.8 / len(variants)
    for j, v in enumerate(variants): ax.bar(x + (j - (len(variants) - 1) / 2) * w, piv[v], width=w, color=C[j], label=VNAME[v])
    ax.axhline(0, color='k', lw=0.5); ax.set_xticks(x); ax.set_xticklabels([MNAME2[m] for m in METRICS]); ax.set_ylabel('$r$ within cell'); ax.set_ylim(-0.3, 0.7)
    save(fig, f'{prefix}_i_controls', f'{label}: r within cell between observed group metrics and the seed-mean prediction of each simulation variant ({", ".join(variants)}); out-degree r ~ 0.48 for every variant reflects group size (4 vs 5 members) and is not policy information. Source: results/phase3/{eval_file}.')
    save_legend([Patch(color=C[j], label=VNAME[v]) for j, v in enumerate(variants)], f'{prefix}_legend', f'Legend for {prefix}_i_controls: simulation variants', ncol=4 if len(variants) > 5 else 3)
chirkov_set('s1', 'sim_runs.parquet', 'eval_r.csv', ['main', 'ctrl2b_gmean', 'ctrl3_nominal', 'ctrl4b_mps', 'ctrl4_shuffle', 'ctrl1_paper', 'ctrl2_avg'], 'Chirkov D-1 (ABM policy)')
chirkov_set('s2', 'sim_runs_d2.parquet', 'eval_r_d2.csv', ['main', 'ctrl2b_gmean', 'ctrl3_nominal', 'ctrl1_pop', 'ctrl2_avg'], 'Chirkov D-2 (neural policy)')

# ---------------- S3: Wu grid ABM without / with mobility ----------------
gr = pd.read_csv(R / 'wu2025/group_rounds.csv'); ob = gr.groupby(['session', 'env'])['total_reward'].mean()
me = pd.read_csv(R / 'wu2025/sim_mobility_eval.csv'); me = me[me.observed == 'total_reward'].set_index('variant')
for name, f, v, lab in [('s3_a_no_mobility', 'sim_runs.parquet', 'main', 'without mobility term'), ('s3_b_mobility', 'sim_runs_mobility.parquet', 'main_mob', 'with mobility term')]:
    sim = pd.read_parquet(R / 'wu2025' / f); pr = sim[sim.variant == v].groupby(['session', 'env'])['total_reward'].mean()
    d = pd.DataFrame({'obs': ob, 'pred': pr}).reset_index(); d['oz'] = zcell(d['obs'], d['env']); d['pz'] = zcell(d['pred'], d['env'])
    fig, ax = plt.subplots(figsize=(2.15, 2.0))
    for e, col in [('random', C[0]), ('smooth', C[1])]:
        m = d.env == e; ax.plot(d.pz[m], d.oz[m], 'o', ms=3, color=col, alpha=0.75, mew=0)
    r = me.loc[v]; ax.text(0.04, 0.95, f'$r = {r.r:.2f}$ [${r.ci_lo:.2f}$, ${r.ci_hi:.2f}$]', transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel('Simulated group reward\n($z$ within environment)'); ax.set_ylabel('Observed group reward ($z$)'); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    save(fig, name, f'Wu: observed group reward (mean of the 4 group rounds per session x environment, 64 cells) vs the zero-shot grid-ABM simulation with the members\' fitted weights, {lab} (variant {v}); blue random, orange smooth environment (as fig4b); r [95 % CI] from sim_mobility_eval.csv. Sources: results/wu2025/{f}, group_rounds.csv.')
fig, ax = plt.subplots(figsize=(2.4, 2.0)); base = ['main', 'gmean', 'pop', 'nominal']; x = np.arange(4)
for k, (suf, mfc, lab) in enumerate([('', 'white', 'without mobility'), ('_mob', C[0], 'with mobility')]):
    rr = me.loc[[b + suf for b in base]]; off = -0.15 + 0.3 * k
    ax.errorbar(x + off, rr.r, yerr=[rr.r - rr.ci_lo, rr.ci_hi - rr.r], fmt='o', ms=4, color=C[0], mfc=mfc, mew=0.8, capsize=2, lw=0.8, label=lab)
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(x); ax.set_xticklabels(["members'\nown", 'group\nmean', 'popu-\nlation', 'each\nalone']); ax.set_ylabel('$r$ within environment'); ax.set_ylim(-0.3, 1.0); ax.legend(frameon=False, loc='upper right')
save(fig, 's3_c_variants', 'Wu: r within environment (64 cells; session-cluster bootstrap 95 % CI) for the four simulation variants without (open) and with (filled) the per-agent mobility term p_i = solo path length / max. Source: results/wu2025/sim_mobility_eval.csv.')
cal = pd.read_csv(R / 'wu2025/sim_mobility_calibration.csv').set_index('env'); cols = ['main', 'main_mob', 'gmean_mob', 'observed']; labs = ['without mobility', 'with mobility', 'group-mean agents, with mobility', 'observed']
fig, ax = plt.subplots(figsize=(3.4, 1.8)); x = np.arange(2); w = 0.2
for j, (c, l) in enumerate(zip(cols, labs)): ax.bar(x + (j - 1.5) * w, cal[c], width=w, color=C[j] if c != 'observed' else '0.3', label=l)
ax.set_xticks(x); ax.set_xticklabels(['random environment', 'smooth environment']); ax.set_ylabel('Group reward (blocks)'); ax.legend(frameon=False, loc='upper left', ncol=2); ax.set_ylim(0, 75)
save(fig, 's3_d_calibration', 'Wu: absolute group reward per environment (mean over cells and runs) for the simulation without mobility (main), with mobility (main_mob), group-mean agents with mobility (gmean_mob) and the observed value. Source: results/wu2025/sim_mobility_calibration.csv.')

# ---------------- S4: social-rule search ----------------
sr = pd.read_csv(R / 'phase3/social_rules_search.csv'); sl = pd.read_csv(R / 'phase3/social_rules_levels.csv')
RULES = ['R0_baseline', 'R1_nearest', 'R2_best', 'R3_scan', 'R4_align', 'R5_tracker_only', 'R6_tracker_calm', 'R7_tracker_scan']; RS = [r.split('_')[0] for r in RULES]
HM = ['score', 'tracking_time', 'in_deg', 'out_deg', 'in_out_corr', 'disc_gain']
piv = sr.pivot(index='rule', columns='metric', values='r_within_cell').reindex(RULES)[HM]
fig, ax = plt.subplots(figsize=(4.5, 2.7)); im = ax.imshow(piv.to_numpy(), cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
for i in range(len(RULES)):
    for j in range(len(HM)): ax.text(j, i, f'{piv.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8, color='k')
ax.set_xticks(range(len(HM))); ax.set_xticklabels([MNAME2[m] for m in HM]); ax.set_yticks(range(len(RULES))); ax.set_yticklabels(RS)
for s in ['top', 'right', 'left', 'bottom']: ax.spines[s].set_visible(False)
ax.tick_params(length=0)
save(fig, 's4_a_heatmap', 'Chirkov: social-rule search, r within cell between the seed-mean simulation (103 groups x 50 seeds) and the observed group metric for the baseline ABM rule (R0) and seven single-element modifications (R1 nearest other; R2 best-informed quality; R3 scanning while searching; R4 heading alignment; R5 tracker-only social information; R6 calmer tracking; R7 = R5 + R3); colour = r (diverging, symmetric +-0.6; colour bar in s4_colorbar). Source: results/phase3/social_rules_search.csv.')
fig = plt.figure(figsize=(0.8, 2.4)); cax = fig.add_axes([0.1, 0.05, 0.22, 0.9]); cb = matplotlib.colorbar.ColorbarBase(cax, cmap='RdBu_r', norm=matplotlib.colors.Normalize(-0.6, 0.6), orientation='vertical'); cb.set_label('$r$ within cell'); cb.outline.set_linewidth(0.6)
fig.savefig(OUT / 's4_colorbar.pdf'); fig.savefig(OUT / 's4_colorbar.png', dpi=220, transparent=False, facecolor='white'); plt.close(fig); caps.append('si/s4_colorbar.pdf\tColour bar for s4_a_heatmap (r within cell, -0.6 to +0.6) (colour bar only; bbox tight, ~0 pt padding)')
for i, met in enumerate(['disc_gain', 'in_out_corr', 'in_deg', 'tracking_time']):
    d = sl[sl.metric == met].set_index('rule').reindex(RULES)
    fig, ax = plt.subplots(figsize=(1.75, 1.5)); ax.plot(range(8), d.sim_mean, 'o', ms=3.5, color=C[0], mew=0)
    ax.axhline(d.obs_mean.iloc[0], color='k', lw=0.8, ls='--'); ax.text(0.3, d.obs_mean.iloc[0], 'observed', ha='left', va='bottom', fontsize=8)
    ax.margins(y=0.3); ax.set_xticks(range(8)); ax.set_xticklabels(RS, fontsize=8); ax.set_ylabel(MNAME[met])
    save(fig, f's4_b{i + 1}_{met}', f'Chirkov: level (mean over the 103 groups) of simulated {MNAME[met]} for rules R0-R7 (points) and the observed level (dashed line). Source: results/phase3/social_rules_levels.csv.')

# ---------------- S5: environment validation ----------------
V = pd.read_csv(R / 'phase3/env_validation.csv'); speeds = sorted(V.resource_speed.unique())
for name, m, lab in [('s5_a_tracking_time', 'tracking_time', 'Tracking time'), ('s5_b_pcr', 'pcr', 'Efficiency (points per tracking step)')]:
    fig, ax = plt.subplots(figsize=(2.3, 2.2))
    for si_, sp in enumerate(speeds):
        for N, mk in [(2, '^'), (6, 'o')]:
            d = V[(V.resource_speed == sp) & (V.N == N)]; ax.plot(d[f'{m}_mesa'], d[f'{m}_np'], mk, ms=4, color=C[si_], mew=0, alpha=0.85)
    lo, hi = V[[f'{m}_mesa', f'{m}_np']].min().min(), V[[f'{m}_mesa', f'{m}_np']].max().max(); pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.6); r = np.corrcoef(V[f'{m}_mesa'], V[f'{m}_np'])[0, 1]
    ax.text(0.04, 0.95, f'$r = {r:.3f}$', transform=ax.transAxes, va='top', fontsize=8); ax.set_xlabel(f'{lab.split(" (")[0]}, mesa ABM'); ax.set_ylabel(f'{lab.split(" (")[0]}, vectorised'); ax.set_aspect('equal')
    save(fig, name, f'Environment validation: {lab} of homogeneous paper-parameter agents on the Fig 5B grid (k in 0.1, 1, 10; resource speed 0.2-1.0; solo = triangles, 5 agents = circles; colour = resource speed), mesa ABM (100 runs) vs the vectorised re-implementation (200 runs); dashed = identity; r = {r:.3f}. Source: results/phase3/env_validation.csv.')
save_legend([Line2D([], [], marker='o', color=C[i], ls='none', ms=4, label=f'resource speed {sp:.1f}') for i, sp in enumerate(speeds)] + [Line2D([], [], marker='^', color='0.4', ls='none', ms=4, label='solo (1 agent)'), Line2D([], [], marker='o', color='0.4', ls='none', ms=4, label='group (5 agents)')],
            's5_legend', 'Legend for s5_a/b: colour = resource speed, marker = solo vs group', ncol=4)
(OUT / 'captions.txt').write_text('\n'.join(caps) + '\n')
import subprocess, re
for f in sorted(OUT.glob('*.pdf')):
    out = subprocess.run(['pdfinfo', str(f)], capture_output=True, text=True).stdout; print(f.name, re.search(r'Page size:\s+(.*)', out).group(1))
