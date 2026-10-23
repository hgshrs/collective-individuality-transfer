"""Build the supplementary Zenodo archive: every tracked file that scripts/91_build_public_repo.py marks `zenodo`
(submission TIFFs, raw seed x group simulation outputs, participant- and pair-level result files), with their
repository paths preserved, plus README_ARCHIVE.md and SHA256SUMS.txt inside the zip. Nothing from data/ or
external/ is included (participant data stay with the original studies).

Usage: uv run python scripts/92_build_zenodo_archive.py [--out ~/public/collective-individuality-transfer_supplement_v1.0.0.zip]
"""
import argparse, hashlib, importlib.util, os, subprocess, time, zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build91', SRC / 'scripts/91_build_public_repo.py'); b91 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b91)
CODE_DOI, VERSION = '10.5281/zenodo.23582821', 'v1.0.0'
README = f"""# Supplementary archive of the analysis repository (release {VERSION})

Paper: Higashi H. (2026). Solitary movement style predicts collective foraging while whom to follow emerges only in company. Submitted.
Code and aggregate results: https://github.com/hgshrs/collective-individuality-transfer (Zenodo concept DOI {CODE_DOI}).

This archive holds the files that the code repository does not version, with the same relative paths, so that
unpacking it on top of a clone of the repository restores the complete `results/` and `paper/figures/` trees:

- `paper/figures/*.tif`: the submitted 600 dpi figures (the PDFs are in the repository; sizes in paper/figures/SIZES.txt).
- `results/**/sim_runs*.parquet`, `results/phase3/social_rules_runs.parquet`, `results/diagnostics/abm_het_real_groups.csv`:
  raw seed x group simulation outputs (the seed-mean evaluations derived from them are in the repository).
- participant-level and pair-level result files (per-participant teacher weights, z, fitted policy parameters,
  out-of-fold predictions, pairwise transfer errors, posterior means extracted from the published Stan fits of Wu
  et al.). Participant identifiers are those of the public datasets of Chirkov et al. 2026 (OSF ZRHWT),
  Wu et al. 2025 (github.com/charleywu/minecraftforaging) and Hawkins et al. 2023 (github.com/hawkrobe/emergent-sensing);
  the raw data themselves are not included and are obtained from those sources (see the repository README).

Every file is listed in MANIFEST.tsv (path, bytes, reason) and SHA256SUMS.txt. The scripts that produce each file
are named in the repository README ("What is not included, and how to regenerate it").

Licence: CC BY 4.0 (Hiroshi Higashi, 2026), as for the results and figures of the repository (LICENSE-docs).
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.expanduser(f'~/public/collective-individuality-transfer_supplement_{VERSION}.zip'))
    a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    dirty = [l for l in subprocess.run(['git', 'status', '--porcelain'], cwd=SRC, capture_output=True, text=True).stdout.splitlines() if not l.startswith('??')]
    if dirty: raise SystemExit('private repository has uncommitted tracked changes:\n' + '\n'.join(dirty))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SRC, capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(['git', 'ls-files'], cwd=SRC, capture_output=True, text=True).stdout.splitlines()
    files = [(p, reason) for p in sorted(tracked) for d, reason, z in [b91.classify(p)] if z]
    total = sum((SRC / p).stat().st_size for p, _ in files)
    print(f'{len(files)} files, {total / 2**20:.1f} MB uncompressed, from private commit {head[:7]}')
    sums, man = [], ['path\tbytes\treason']
    t0 = time.time()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, reason in files:
            b = (SRC / p).read_bytes(); sums.append(f'{hashlib.sha256(b).hexdigest()}  {p}'); man.append(f'{p}\t{len(b)}\t{reason}')
            z.writestr(zipfile.ZipInfo.from_file(SRC / p, p, strict_timestamps=False), b, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('README_ARCHIVE.md', README + f'\nBuilt by scripts/92_build_zenodo_archive.py from private commit {head} on {time.strftime("%Y-%m-%d")}.\n')
        z.writestr('MANIFEST.tsv', '\n'.join(man) + '\n'); z.writestr('SHA256SUMS.txt', '\n'.join(sums) + '\n')
    by = {}
    for p, reason in files: by[reason] = by.get(reason, 0) + 1
    for reason, n in sorted(by.items(), key=lambda x: -x[1]): print(f'  {n:4d}  {reason}')
    print(f'{out}: {out.stat().st_size / 2**20:.1f} MB zipped in {time.time() - t0:.0f} s; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
