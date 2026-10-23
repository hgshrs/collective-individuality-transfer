"""Conceptual figure (HANDOVER §12 item 6): the four directions in which individual 'style' measured in private was tested
for transfer, with the verdict of the three datasets. One panel, usetex Helvetica, 5.2 in wide (PLOS single column), fonts 8-9 pt, no title, tab10 colours.
Output: results/paper_figures/fig0_concept.{pdf,png} (+ caption line appended to captions.txt / INDEX.md by hand)
"""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / 'results/paper_figures'
plt.rcParams.update({'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
                     'text.latex.preamble': r'\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}',
                     'font.size': 8, 'savefig.transparent': True, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = plt.get_cmap('tab10').colors
fig = plt.figure(figsize=(5.0, 3.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-9, 100); ax.axis('off')
def box(x, y, w, h, title, body, fc='white', ec='0.3', lw=0.7, tsize=9, bsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.5', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 4.5, title, ha='center', va='center', fontsize=tsize, weight='bold')
    if body: ax.text(x + w / 2, y + (h - 9) / 2 + 0.5, body, ha='center', va='center', fontsize=bsize, linespacing=1.15)
def arrow(x0, y0, x1, y1, color='0.3', lw=0.9, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
# left: private phase -> z
box(0, 36, 21, 28, 'Private phase', 'alone, or away\nfrom the resource\n(position,\nheading, speed)', fc=(0.89, 0.93, 0.97), ec=C[0])
arrow(21, 50, 23, 50)
box(23, 36, 20, 28, 'Style $z$', 'linear, 4 dims\nfrom about 20\nmovement\nstatistics')
# right: four directions
ys = [82, 60, 38, 15]; H = 18; X = 47; W = 53
box(X, ys[0] - H / 2, W, H, '1 Private processing in the group', 'transfers up to the ceiling ($r$ 0.3--0.7)', fc=(0.90, 0.96, 0.90), ec=C[2])
box(X, ys[1] - H / 2, W, H, '2 Social attraction', 'partial ($r$ 0.1--0.6)', fc=(1.0, 0.96, 0.88), ec=C[1])
box(X, ys[2] - H / 2, W, H, '3 Selective social learning', 'reliable in the group (0.45--0.55)\nbut not transferable ($r \\approx 0$)', fc=(0.98, 0.90, 0.90), ec=C[3])
box(X, ys[3] - H / 2, W, H, '4 Roles and group performance', "own $z$ + others' mean, additive;\ngroup = members' mean, no composition", fc=(0.90, 0.96, 0.90), ec=C[2])
for y, col, ls in zip(ys, [C[2], C[1], C[3], C[2]], ['-', '-', (0, (2, 1.5)), '-']):
    arrow(43, 50, X - 0.5, y, color=col, ls=ls)
ax.text(0, -8.5, 'Same pattern in Chirkov et al. (3-D tracking), Wu et al. (Minecraft foraging)\nand Hawkins et al. (2-D hidden spotlight)', fontsize=8, color='0.3', ha='left', va='bottom', linespacing=1.2)
fig.savefig(OUT / 'fig0_concept.pdf'); fig.savefig(OUT / 'fig0_concept.png', dpi=220, transparent=False, facecolor='white')
import subprocess; print(subprocess.run(['pdfinfo', str(OUT / 'fig0_concept.pdf')], capture_output=True, text=True).stdout.split('Page size:')[1].split('\n')[0])
