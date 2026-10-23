"""Assemble the single-panel PDFs in results/paper_figures into PLOS Computational Biology multi-panel figures.

PLOS requirements applied: one file per figure (Fig1..Fig6), panels labelled with bold upper-case letters (12 pt Helvetica)
at the top-left of each panel, panels placed at 1:1 (fonts 8-9 pt at final size), width 2.63-7.5 in, height <= 8.75 in,
TIFF (RGB, LZW, 600 dpi) for submission plus the PDF used to make it. Layout is done in LaTeX (standalone class), one .tex
per figure in paper/figures/build/. Usage: uv run python paper/figures/assemble.py [Fig1 ... S5_Fig]  (no argument = all)
"""
import subprocess, shutil, re
from pathlib import Path
from PIL import Image
P = Path(__file__).resolve().parents[2]; SRC = P / 'results/paper_figures'; OUT = P / 'paper/figures'; B = OUT / 'build'; B.mkdir(exist_ok=True)
Image.MAX_IMAGE_PIXELS = None
LABEL = r'{\sffamily\bfseries\fontsize{12}{14}\selectfont %s}'

def panel(label, pdf):
    """pdf: file name under results/paper_figures (or 'si/...'), or a nested list of rows of file names (a block of small panels)."""
    if isinstance(pdf, list):
        inner = ' \\\\[4pt] '.join('\\hspace{4pt}'.join(f'\\includegraphics{{{SRC / q}}}' for q in row) for row in pdf)
        head = f'\\makebox[0pt][l]{{{LABEL % label}}} \\\\[0.3ex] ' if label else ''
        return f'\\begin{{tabular}}[t]{{@{{}}l@{{}}}} {head}{inner} \\end{{tabular}}'
    g = f'\\includegraphics{{{SRC / pdf}}}'
    if label is None: return g
    lab = LABEL % label if label else '\\strut'  # '' = no letter but top-aligned like a labelled panel (e.g. a colour bar next to a panel)
    return (f'\\begin{{minipage}}[t]{{\\widthof{{{g}}}}}\\makebox[0pt][l]{{{lab}}}\\\\[0.3ex]{g}\\end{{minipage}}')

FIGS = {
    # name: list of rows; each row = list of (label, pdf) ; rows separated by vertical space
    'Fig1': [[(None, 'fig0_concept.pdf')]],
    'Fig2': [[('A', 'fig1_reliability.pdf')], [(None, 'fig1_legend.pdf')]],
    'Fig3': [[('A', 'fig2a_transfer_chirkov.pdf'), ('B', 'fig2b_transfer_wu.pdf')], [('C', 'fig2c_transfer_hawkins.pdf'), (None, 'fig2_legend.pdf')]],
    'Fig4': [[(None, 'fig3_roles_decomposition.pdf')]],
    'Fig5': [[('A', 'fig4a_group_chirkov.pdf'), ('B', 'fig4b_group_wu.pdf'), ('C', 'fig4c_group_hawkins.pdf')]],
    'Fig6': [[(None, 'fig5_composition_pooled.pdf')], [(None, 'fig1_legend.pdf')]],
    # supporting information (panels from scripts/89 in results/paper_figures/si). Numbering since 2026-10-23: S3 = environment validation (s5_*), S4 = Wu mobility ABM (s3_*), S5 = social rules (s4_*); panel file prefixes keep their original numbers
    'S1_Fig': [[('A', 'si/s1_a_score.pdf'), ('B', 'si/s1_b_tracking_time.pdf'), ('C', 'si/s1_c_pcr.pdf'), ('D', 'si/s1_d_med_dist.pdf')],
               [('E', 'si/s1_e_in_deg.pdf'), ('F', 'si/s1_f_out_deg.pdf'), ('G', 'si/s1_g_in_out_corr.pdf'), ('H', 'si/s1_h_disc_gain.pdf')],
               [('I', 'si/s1_i_controls.pdf')], [(None, 'si/s1_legend.pdf')]],
    'S2_Fig': [[('A', 'si/s2_a_score.pdf'), ('B', 'si/s2_b_tracking_time.pdf'), ('C', 'si/s2_c_pcr.pdf'), ('D', 'si/s2_d_med_dist.pdf')],
               [('E', 'si/s2_e_in_deg.pdf'), ('F', 'si/s2_f_out_deg.pdf'), ('G', 'si/s2_g_in_out_corr.pdf'), ('H', 'si/s2_h_disc_gain.pdf')],
               [('I', 'si/s2_i_controls.pdf')], [(None, 'si/s2_legend.pdf')]],
    'S4_Fig': [[('A', 'si/s3_a_no_mobility.pdf'), ('B', 'si/s3_b_mobility.pdf'), ('C', 'si/s3_c_variants.pdf')], [('D', 'si/s3_d_calibration.pdf')]],
    'S5_Fig': [[('A', 'si/s4_a_heatmap.pdf'), ('', 'si/s4_colorbar.pdf')], [('B', [['si/s4_b1_disc_gain.pdf', 'si/s4_b2_in_out_corr.pdf'], ['si/s4_b3_in_deg.pdf', 'si/s4_b4_tracking_time.pdf']])]],
    'S3_Fig': [[('A', 'si/s5_a_tracking_time.pdf'), ('B', 'si/s5_b_pcr.pdf')], [(None, 'si/s5_legend.pdf')]],
}
import sys
ONLY = sys.argv[1:]  # e.g. `assemble.py S1_Fig S2_Fig` builds only those (leaves the other files untouched)
sizes = []
for name, rows in FIGS.items():
    if ONLY and name not in ONLY: continue
    body = '\n\n\\vspace{8pt}\n\n'.join('\\hspace{6pt}'.join(panel(l, f) for l, f in row) for row in rows)
    tex = ('\\documentclass[varwidth=7.5in,border=2pt]{standalone}\n\\usepackage{graphicx,calc,helvet}\n\\renewcommand{\\familydefault}{\\sfdefault}\n'
           '\\begin{document}\n\\raggedright\n' + body + '\n\\end{document}\n')
    (B / f'{name}.tex').write_text(tex)
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{name}.tex'], cwd=B, capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-1500:]); raise SystemExit(f'{name} failed')
    shutil.copy(B / f'{name}.pdf', OUT / f'{name}.pdf')
    info = subprocess.run(['pdfinfo', str(OUT / f'{name}.pdf')], capture_output=True, text=True).stdout
    w, h = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups())
    subprocess.run(['pdftoppm', '-r', '600', '-png', '-singlefile', str(OUT / f'{name}.pdf'), str(B / name)], check=True)
    im = Image.open(B / f'{name}.png').convert('RGB'); im.save(OUT / f'{name}.tif', compression='tiff_lzw', dpi=(600, 600))
    im.resize((im.width // 4, im.height // 4), Image.LANCZOS).save(OUT / f'{name}.png', dpi=(150, 150))
    ok = (2.63 * 72 <= w <= 7.5 * 72) and h <= 8.75 * 72
    sizes.append(f'{name}\t{w / 72:.2f} x {h / 72:.2f} in\t{im.width} x {im.height} px @600 dpi\t{(OUT / f"{name}.tif").stat().st_size / 1e6:.1f} MB\t{"OK" if ok else "OUT OF RANGE"}')
old = {l.split('\t')[0]: l for l in (OUT / 'SIZES.txt').read_text().splitlines()} if (OUT / 'SIZES.txt').exists() else {}
for l in sizes: old[l.split('\t')[0]] = l
(OUT / 'SIZES.txt').write_text('\n'.join(old[k] for k in sorted(old, key=lambda k: (k.startswith('S'), k))) + '\n'); print('\n'.join(sizes))
