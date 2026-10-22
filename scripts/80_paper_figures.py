"""Paper figures (one panel = one PDF) across the three datasets. Style per CLAUDE.md: usetex + Helvetica (sfmath),
tab10, no grid, no titles, transparent, bbox tight. Fonts: PLOS Computational Biology (8-12 pt at final size): labels 9 pt,
ticks/legend/annotations 8 pt. Panels are assembled into PLOS multi-panel figures by paper/figures/assemble.py. Also writes PNG previews and captions.txt (one line per file)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
P = Path('/home/higashi/Research/20260924_collective'); OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'results/paper_figures'; OUT.mkdir(exist_ok=True, parents=True)  # optional output dir (to regenerate selected files only)
plt.rcParams.update({'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
                     'text.latex.preamble': r'\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}',
                     'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,  # PLOS Comput Biol: 8-12 pt (2026-10-21)
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'savefig.transparent': True, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = plt.get_cmap('tab10').colors
DS = {'Chirkov': C[0], 'Wu': C[1], 'Hawkins': C[2]}
caps = []
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def save_legend(handles, name, caption, ncol=1):
    fig = plt.figure(figsize=(3.4, 0.6)); leg = fig.legend(handles=handles, loc='center', frameon=False, ncol=ncol, handlelength=1.2, handletextpad=0.5, columnspacing=1.2)
    fig.canvas.draw(); bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT / f'{name}.pdf', bbox_inches=bb.expanded(1.04, 1.15)); fig.savefig(OUT / f'{name}.png', dpi=220, transparent=False, facecolor='white', bbox_inches=bb.expanded(1.04, 1.15)); plt.close(fig)
    caps.append(f'{name}.pdf\t{caption} (legend only; saved with ~2 % padding, no further trim needed)')
def save(fig, name, caption):
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=220, transparent=False, facecolor='white'); plt.close(fig)
    caps.append(f'{name}.pdf\t{caption}')

# ---------------- Fig 1: split-half reliability by trait class ----------------
rel_rows = [  # (class, dataset, label, r)
    ('Private processing', 'Chirkov', 'Intercept (turning rate)', 0.763), ('Private processing', 'Chirkov', 'Detector weight', 0.605),
    ('Private processing', 'Wu', 'Locality weight', 0.831), ('Private processing', 'Wu', 'Block-visibility weight', 0.760),
    ('Private processing', 'Hawkins', 'Stop latency in region', 0.689), ('Private processing', 'Hawkins', 'Stopping in region', 0.636),
    ('Social attraction', 'Chirkov', 'Social position weight', 0.298), ('Social attraction', 'Chirkov', 'Social movement weight', 0.167),
    ('Social attraction', 'Wu', 'Looking at others', 0.586), ('Social attraction', 'Hawkins', 'Approach events', 0.610), ('Social attraction', 'Hawkins', 'Attraction to centroid', 0.433),
    ('Selective social learning', 'Chirkov', 'Position x payoff', 0.116), ('Selective social learning', 'Chirkov', 'Movement x payoff', 0.090),
    ('Selective social learning', 'Wu', 'Successful-player proximity', 0.445), ('Selective social learning', 'Wu', 'Unsuccessful-player proximity', 0.271),
    ('Selective social learning', 'Hawkins', 'Facing stopped others', 0.549), ('Selective social learning', 'Hawkins', 'Selectivity index', 0.246), ('Selective social learning', 'Hawkins', 'Attraction to stopped others', 0.161),
    ('Being attended to', 'Wu', 'In-degree (seen by others)', 0.236), ('Being attended to', 'Hawkins', 'Faced by others', 0.216),
]
rel = pd.DataFrame(rel_rows, columns=['cls', 'ds', 'label', 'r'])
classes = ['Private processing', 'Social attraction', 'Selective social learning', 'Being attended to']
fig, ax = plt.subplots(figsize=(3.4, 3.5))
y = 0; yt = []; ytl = []; sep = []
for cls in classes:
    sub = rel[rel.cls == cls].sort_values('r', ascending=True)
    y0 = y
    for _, r in sub.iterrows():
        ax.plot([0, r.r], [y, y], color=DS[r.ds], lw=0.8, alpha=0.5); ax.plot(r.r, y, 'o', color=DS[r.ds], ms=4)
        ax.text(r.r + 0.02, y, r.label, va='center', ha='left', fontsize=8, color='0.25'); y += 1  # 8 pt (PLOS; was 5.5)
    yt.append((y0 + y - 1) / 2); ytl.append(cls.replace(' ', '\n') if cls != 'Being attended to' else 'Being\nattended to'); sep.append(y - 0.5); y += 0.8
ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=8)
for s in sep[:-1]: ax.axhline(s + 0.4, color='0.8', lw=0.5)
ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel('Split-half reliability $r$')
ax.invert_yaxis()
save_legend([Line2D([], [], marker='o', color=DS[d], ls='none', ms=4, label=l) for d, l in [('Chirkov', 'Chirkov et al. (3-D tracking)'), ('Wu', 'Wu et al. (Minecraft foraging)'), ('Hawkins', 'Hawkins et al. (2-D hidden spotlight)')]], 'fig1_legend', 'Legend for fig1_reliability and fig5_composition: dataset colours (tab10 C0 Chirkov, C1 Wu, C2 Hawkins)', ncol=1)
save(fig, 'fig1_reliability', 'Split-half reliability (first vs second half of the session) of individual parameters, grouped by trait class, for the three datasets (colour = dataset; each point is one parameter, named beside it). Private processing 0.60-0.83; social attraction 0.17-0.61; selective (payoff-dependent) social learning 0.09-0.55 but <= 0.27 for the Chirkov quality interactions and the Hawkins selectivity measures; being attended to 0.22-0.24. Sources: results/phase0/split_half_reliability.csv, results/wu2025/{choice_reliability,ind_reliability}.csv, results/hawkins2023/{reliability,teacher_reliability}.csv.')

# ---------------- Fig 2: transfer (z -> targets) per dataset ----------------
def transfer_panel(rows, name, caption, xlim=(-0.3, 0.9)):
    fig, ax = plt.subplots(figsize=(2.1, 2.3))  # 2026-10-21: 2.1 in so that the PDF (with tick labels) is <= 3.4 in wide
    labels = [r[0] for r in rows]; vals = np.array([r[1] for r in rows]); ceil = np.array([r[2] for r in rows], float); cls = [r[3] for r in rows]
    yy = np.arange(len(rows))[::-1]
    cols = {'private': C[0], 'social': C[1], 'selective': C[3], 'role': C[2], 'performance': C[4]}
    ax.barh(yy, vals, height=0.6, color=[cols[c] for c in cls])
    for yi, ce in zip(yy, ceil):
        if not np.isnan(ce): ax.plot([ce, ce], [yi - 0.38, yi + 0.38], color='k', lw=1.2)
    ax.axvline(0, color='k', lw=0.5); ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8); ax.set_xlim(*xlim); ax.set_xlabel('Out-of-fold $r$ from the private phase')  # shortened 2026-10-21 (print width)
    save(fig, name, caption)
    return cols
transfer_panel([('Intercept (turning rate)', 0.499, 0.763, 'private'), ('Detector weight', 0.301, 0.605, 'private'), ('Social position weight', 0.157, 0.298, 'social'),
                ('Social movement weight', 0.135, 0.167, 'social'), ('Position x payoff', -0.083, 0.116, 'selective'), ('Movement x payoff', 0.028, 0.090, 'selective'),
                ('In-degree (early 3 min to later)', 0.252, np.nan, 'role'), ('Out-degree (early 3 min to later)', 0.317, np.nan, 'role')],
               'fig2a_transfer_chirkov', 'Chirkov et al.: out-of-fold r of tracking-phase individual parameters predicted from a linear search-phase z (PLS, 4 dims; n = 568), and of later visibility roles predicted from the first-3-min movement z (PCA 4; n = 477). Black ticks = split-half reliability of the target (none computed for roles). Sources: results/phase1/baselines_ridge_eb.csv (pls_z4), results/phase2/planB_r.csv.')
transfer_panel([('Locality weight', 0.642, 0.831, 'private'), ('Block-visibility weight', 0.664, 0.760, 'private'), ('Reward-prediction weight', 0.254, 0.426, 'private'),
                ('Looking at others (out-degree)', 0.249, 0.586, 'social'), ('Successful-player proximity', -0.234, 0.445, 'selective'), ('Unsuccessful-player proximity', -0.240, 0.271, 'selective'),
                ('Seen by others (in-degree)', -0.002, 0.236, 'role'), ('Reward in group rounds', 0.573, 0.561, 'performance')],
               'fig2b_transfer_wu', 'Wu et al.: out-of-fold r (8 folds of 4 sessions, n = 128) of group-round targets predicted from the solo-round z (PCA 4 of 21 movement/foraging statistics). Black ticks = split-half reliability of the target across group rounds. Source: results/wu2025/transfer_r.csv (solo_z4).')
transfer_panel([('Stop latency in region', 0.366, 0.689, 'private'), ('Stopping in region', 0.316, 0.636, 'private'), ('Approach events', 0.564, 0.610, 'social'),
                ('Nearest-neighbour distance', 0.378, 0.591, 'social'), ('Facing stopped others', 0.060, 0.549, 'selective'), ('Selectivity index', -0.120, 0.246, 'selective'),
                ('Faced by others', 0.170, 0.216, 'role'), ('Score', 0.443, 0.412, 'performance')],
               'fig2c_transfer_hawkins', 'Hawkins et al. Exp 1: out-of-fold r (10 folds of whole games, n = 682, within experiment x group size x field) of second-half targets predicted from the first-half search-state z (PCA 4). Black ticks = first-vs-second-half reliability of the target. Source: results/hawkins2023/transfer_r.csv (z4).')

cols2 = {'private': C[0], 'social': C[1], 'selective': C[3], 'role': C[2], 'performance': C[4]}
save_legend([Line2D([], [], color='k', lw=1.2, label='split-half ceiling of the target')] + [Patch(color=cols2[k], label=l) for k, l in [('private', 'private processing'), ('social', 'social attraction'), ('selective', 'selective social learning'), ('role', 'being attended to'), ('performance', 'performance')]],
            'fig2_legend', 'Legend shared by fig2a-c: bar colour = trait class of the target; black tick = split-half ceiling', ncol=3)
# ---------------- Fig 3: decomposition of roles ----------------
dec = {  # dataset, role: (ctrl, +own, +others, +SD) marginal R2
    ('Chirkov', 'in'): (0.078, 0.163, 0.241, 0.252), ('Chirkov', 'out'): (0.070, 0.157, 0.205, 0.207),
    ('Wu', 'in'): (0.004, 0.015, 0.034, 0.042), ('Wu', 'out'): (0.002, 0.054, 0.059, 0.064),
    ('Hawkins', 'in'): (0.023, 0.039, 0.051, 0.063), ('Hawkins', 'out'): (0.000, 0.046, 0.054, 0.059)}
fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = 0; xt = []; xtl = []
for ds in ['Chirkov', 'Wu', 'Hawkins']:
    for role in ['in', 'out']:
        v = dec[(ds, role)]; inc = [v[1] - v[0], v[2] - v[1], v[3] - v[2]]; bottom = 0
        for j, (h, col, lab) in enumerate(zip(inc, [C[0], C[1], C[3]], ['own $z$', "others' mean $z$", 'group SD of $z$'])):
            ax.bar(x, h, bottom=bottom, width=0.7, color=col, label=lab if x == 0 else None); bottom += h
        xt.append(x); xtl.append({'in': 'attended', 'out': 'attending'}[role]); x += 1
    x += 0.6
ax.set_xticks(xt); ax.set_xticklabels(xtl, fontsize=8, rotation=45, ha='right')
import matplotlib.transforms as mtr
tf = mtr.blended_transform_factory(ax.transData, ax.transAxes)
for i, ds in enumerate(['Chirkov et al.', 'Wu et al.', 'Hawkins et al.']): ax.text(i * 2.6 + 0.5, -0.36, ds, ha='center', va='top', fontsize=8, transform=tf)
ax.set_ylabel('Increment in marginal $R^2$'); ax.set_ylim(0, 0.2); ax.legend(frameon=False, loc='upper right')
save(fig, 'fig3_roles_decomposition', 'Increment in marginal R2 of visibility roles (being attended to = in-degree; attending to others = out-degree) when adding, in order, own z, the other members\' mean z and the within-group SD of z to a mixed model with group random intercept and a performance control. Group SD adds <= 0.012 everywhere (LRT p 0.22-0.95, except Hawkins attended p 0.048). Sources: results/phase2/planB_decomposition.csv (control = score_3min), results/wu2025/decomposition.csv, results/hawkins2023/decomposition.csv.')

# ---------------- Fig 4: group performance prediction scatters ----------------
def zcell(s, key): return s.groupby(key).transform(lambda v: (v - v.mean()) / v.std())
# Chirkov: ABM main simulation (seed mean) vs observed score, z within condition x speed
sim = pd.read_parquet(P / 'results/phase3/sim_runs.parquet'); sim = sim[sim.variant == 'main'].groupby('group')['score'].mean()
obs = pd.read_csv(P / 'results/phase3/observed_group_metrics.csv').set_index('group')
d = obs[['score', 'cell', 'speed']].join(sim.rename('sim')).dropna(); d['obs_z'] = zcell(d['score'], d['cell']); d['sim_z'] = zcell(d['sim'], d['cell'])
fig, ax = plt.subplots(figsize=(2.15, 2.0))  # 2026-10-21: narrower so that three panels fit a 7.16-in double column at 1:1
for sp, col, lab in [('slow', C[0], 'slow resource'), ('fast', C[1], 'fast resource')]:
    m = d.speed == sp; ax.plot(d.sim_z[m], d.obs_z[m], 'o', ms=3, color=col, alpha=0.75, label=lab, mew=0)
r = np.corrcoef(d.sim_z, d.obs_z)[0, 1]; ax.text(0.04, 0.95, f'$r = {r:.2f}$ (103 groups)', transform=ax.transAxes, va='top', fontsize=8)
ax.set_xlabel('Predicted group score\n($z$ within condition)'); ax.set_ylabel('Observed group score ($z$)'); ax.legend(frameon=False, loc='lower right'); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
save(fig, 'fig4a_group_chirkov', 'Chirkov et al.: observed group score vs the score predicted by simulating each group with its members\' individually fitted ABM policies (leave-one-group-out, seed mean, 103 groups; z within condition x speed). r = 0.55 [0.42, 0.67]. Sources: results/phase3/sim_runs.parquet (variant main), results/phase3/observed_group_metrics.csv.')
# Wu: nominal (members' mean solo reward) vs group reward, per session x env
gr = pd.read_csv(P / 'results/wu2025/group_rounds.csv'); ge = gr.groupby(['session', 'env'])[['total_reward', 'nominal_reward']].mean().reset_index()
ge['obs_z'] = zcell(ge['total_reward'], ge['env']); ge['pred_z'] = zcell(ge['nominal_reward'], ge['env'])
fig, ax = plt.subplots(figsize=(2.15, 2.0))  # 2026-10-21: narrower so that three panels fit a 7.16-in double column at 1:1
for e, col, lab in [('random', C[0], 'random environment'), ('smooth', C[1], 'smooth environment')]:
    m = ge.env == e; ax.plot(ge.pred_z[m], ge.obs_z[m], 'o', ms=3, color=col, alpha=0.75, label=lab, mew=0)
r = np.corrcoef(ge.pred_z, ge.obs_z)[0, 1]; ax.text(0.04, 0.95, f'$r = {r:.2f}$ (32 groups $\\times$ 2)', transform=ax.transAxes, va='top', fontsize=8)
ax.set_xlabel("Members' mean solo reward\n($z$ within environment)"); ax.set_ylabel('Observed group reward ($z$)'); ax.legend(frameon=False, loc='lower right'); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
save(fig, 'fig4b_group_wu', 'Wu et al.: observed group reward (mean of the 4 group rounds per session x environment, 64 cells) vs the zero-shot prediction from the members\' mean solo-round reward in the same environment (z within environment). r = 0.55. Source: results/wu2025/group_rounds.csv.')
# Hawkins: members' mean first-half score vs second-half group score (residualised), games >= 2
gt = pd.read_csv(P / 'results/hawkins2023/group_table.csv')
fig, ax = plt.subplots(figsize=(2.15, 2.0))  # 2026-10-21: narrower so that three panels fit a 7.16-in double column at 1:1
for e, col, lab in [(1, C[0], 'Exp. 1 (binary spotlight)'), (3, C[1], 'Exp. 3 (gradient field)')]:
    m = gt.experiment == e; ax.plot(gt.mean_score1[m], gt.score2[m], 'o', ms=3, color=col, alpha=0.75, label=lab, mew=0)
r = np.corrcoef(gt.mean_score1, gt.score2)[0, 1]; ax.text(0.04, 0.95, f'$r = {r:.2f}$ ({len(gt)} groups)', transform=ax.transAxes, va='top', fontsize=8)
ax.set_xlabel("Members' mean first-half score\n($z$ within cell)"); ax.set_ylabel('Second-half group score ($z$)'); ax.legend(frameon=False, loc='lower right'); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
save(fig, 'fig4c_group_hawkins', 'Hawkins et al.: second-half group score vs the members\' mean first-half individual score (281 games with >= 2 players, Exp 1 and 3; both residualised within experiment x group size x field). r = 0.55 [0.44, 0.64]. Source: results/hawkins2023/group_table.csv.')

# ---------------- Fig 5: composition effect forest plot ----------------
rows = [('Chirkov, ABM policy', 'heterogeneous agents $-$ group-mean agents', 0.013, -0.030, 0.059, 'Chirkov'),
        ('Chirkov, neural policy', 'heterogeneous agents $-$ group-mean agents', 0.018, -0.074, 0.116, 'Chirkov'),
        ('Wu, grid ABM', 'heterogeneous agents $-$ group-mean agents', 0.162, -0.036, 0.370, 'Wu'),
        ('Wu, statistical', 'member means $+$ SDs $-$ member means', -0.113, -0.176, -0.044, 'Wu'),
        ('Hawkins, statistical', 'member means $+$ SDs $-$ member means', -0.026, -0.068, 0.012, 'Hawkins')]
fig, ax = plt.subplots(figsize=(2.8, 1.9))  # 2026-10-21: narrower so that the PDF is <= 3.4 in wide
yy = np.arange(len(rows))[::-1]
for yi, (lab, kind, d_, lo, hi, ds) in zip(yy, rows):
    ax.plot([lo, hi], [yi, yi], color=DS[ds], lw=1.2); ax.plot(d_, yi, 'o', color=DS[ds], ms=4)
ax.axvline(0, color='k', lw=0.5); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel('$\\Delta r$ from within-group heterogeneity (95\\% CI)'); ax.set_xlim(-0.25, 0.45)
save(fig, 'fig5_composition', 'Composition effect on group performance across datasets: change in out-of-sample r when within-group heterogeneity is added to a model that already carries the members\' mean traits (Chirkov: individually fitted agents minus group-mean agents, D-1 ABM policy and D-2 neural policy, 103 groups; Wu: same contrast in the authors\' grid ABM, 64 session x env cells, and member means + SDs minus means in leave-one-session-out ridge, 256 group rounds; Hawkins: means + SDs minus means, 169 games with >= 3 players). Session/group-cluster bootstrap 95 % CIs. Sources: results/phase3/eval_delta*.csv, results/wu2025/{sim_eval_delta,group_pred}.csv, results/hawkins2023/group_pred.csv.')
open(OUT / 'captions.txt', 'w').write('\n'.join(caps) + '\n')
import subprocess
for f in sorted(OUT.glob('*.pdf')):
    out = subprocess.run(['pdfinfo', str(f)], capture_output=True, text=True).stdout
    size = [l for l in out.splitlines() if l.startswith('Page size')]
    print(f.name, size[0] if size else '')
