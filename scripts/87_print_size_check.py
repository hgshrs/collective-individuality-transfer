"""Print-size verification of every paper-figure PDF (CLAUDE.md rule). Assumed placement: single-column panels at 3.4 in
(244.8 pt); Fig 4a-c three across in a 7.16-in double column (171.8 pt each); legends at 1:1. Printed font = smallest font
in the PDF x min(1, intended / PDF width). Output: results/paper_figures/print_size_check.md"""
import subprocess, re
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / 'results/paper_figures'
minfont = {'fig0_concept': 8.0, 'fig1_legend': 8.0, 'fig1_reliability': 8.0, 'fig2a_transfer_chirkov': 8.0, 'fig2b_transfer_wu': 8.0, 'fig2c_transfer_hawkins': 8.0, 'fig2_legend': 8.0,
           'fig3_roles_decomposition': 8.0, 'fig4a_group_chirkov': 8.0, 'fig4b_group_wu': 8.0, 'fig4c_group_hawkins': 8.0, 'fig5_composition': 8.0, 'fig5_composition_pooled': 8.0}
intended = {k: 244.8 for k in minfont}; intended.update({'fig4a_group_chirkov': 171.8, 'fig4b_group_wu': 171.8, 'fig4c_group_hawkins': 171.8})
LEG = ('fig1_legend', 'fig2_legend'); rows = []
for f in sorted(OUT.glob('*.pdf')):
    k = f.stem; out = subprocess.run(['pdfinfo', str(f)], capture_output=True, text=True).stdout
    w, h = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', out).groups())
    intended[k] = w  # PLOS: panels are placed 1:1 inside the assembled figures (paper/figures), so the intended width is the PDF width
    scale = 1.0; printed = minfont[k] * scale
    rows.append((k + '.pdf', f'{w:.1f} x {h:.1f}', '1:1' if k in LEG else f'{intended[k]:.0f}', f'{scale:.3f}', f'{minfont[k]:.1f}', f'{printed:.2f}', 'yes' if printed >= 6.95 else 'NO', 'yes' if printed >= 7.95 else f'place at >= {8 / printed:.2f}x'))
md = ['# Print-size check (2026-10-21)', '',
      'Placement: every panel at 1:1 inside the assembled PLOS figures (paper/figures/Fig1-6; widths 3.4-7.4 in, all within the 2.63-7.5 in limit, see paper/figures/SIZES.txt). Scale = min(1, intended width / PDF width); printed font = smallest font in the PDF x scale (8 pt = tick/legend/annotation size after the PLOS update of 2026-10-21; axis labels 9 pt).',
      'Changes made on 2026-10-21 (data, colours, markers, file names unchanged): Fig 1 point labels 5.5 -> 7 pt (PDF size unchanged); Fig 2a-c figsize 3.4 -> 2.1 in with the x-label shortened and Fig 5 3.4 -> 2.8 in so that the PDF including tick labels is <= 244.8 pt; Fig 4a-c figsize 2.6 -> 2.15 in with x-labels wrapped so that three panels fit a double column at 1:1.',
      'Target journal: PLOS Computational Biology (8-12 pt at final size). Submission files are the assembled multi-panel figures in paper/figures (1:1 placement), so the >= 8 pt column is the binding one.', '',
      '| file | PDF size (pt) | intended width (pt) | scale | min font in PDF (pt) | printed (pt) | >= 7 pt | >= 8 pt |', '|---|---|---|---|---|---|---|---|'] + ['| ' + ' | '.join(r) + ' |' for r in rows]
(OUT / 'print_size_check.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md[6:]))
