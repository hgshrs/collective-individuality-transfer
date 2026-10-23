# Solitary movement style predicts collective foraging while whom to follow emerges only in company

Analysis code, pre-specified analysis plan, decision log and aggregate results for

> Higashi H. (2026). *Solitary movement style predicts collective foraging while whom to follow emerges only in company.* Submitted.

- Code repository: `https://github.com/hgshrs/collective-individuality-transfer` (this repository)
- Archived on Zenodo: concept DOI [10.5281/zenodo.23582821](https://doi.org/10.5281/zenodo.23582821) (all versions); release v1.0.0 = [10.5281/zenodo.23582822](https://doi.org/10.5281/zenodo.23582822)
- Supplementary archive (submission TIFFs, raw simulation outputs, participant-level result files; 80 MB zip): [10.5281/zenodo.23583135](https://doi.org/10.5281/zenodo.23583135)
- Pre-specified analysis plan and decision log (OSF registration, 2026-10-23): [https://osf.io/7gae9](https://osf.io/7gae9) (DOI [10.17605/OSF.IO/7GAE9](https://doi.org/10.17605/OSF.IO/7GAE9))

This repository is generated from the author's private working repository by `scripts/91_build_public_repo.py`;
nothing in it is edited by hand. The full commit history of the analysis (from the first commit of the specification,
2026-09-24, to the last analysis commit) is retained, with the files listed under *What is not included*
removed from every commit. Participant-level data are **not** redistributed; they are re-created from the three public
datasets by the scripts described below.

## Overview

The study asks whether the individual "style" that a person shows when moving alone (or away from the resource)
predicts what they do in a group, and whether the *combination* of individuals predicts how well the group does.
Three public datasets of collective foraging are analysed with the same pipeline:

1. **Chirkov, Kurvers, Romanczuk & Deffner (2026)**, *Nature Communications* 17:10133. 621 participants tracking a
   moving resource in a 3-D arena, alone or in groups of 4-5 (103 analysed groups), slow and fast resource, with and
   without payoff information (OSF project ZRHWT).
2. **Wu et al. (2025)**, Minecraft collective foraging. 128 participants in 32 sessions of 4, with solo and group
   rounds in random and smooth reward environments (`github.com/charleywu/minecraftforaging`).
3. **Hawkins et al. (2023)**, *Nature Human Behaviour* 7:1767, 2-D "collective sensing" with a hidden score field,
   Experiments 1 and 3 (`github.com/hawkrobe/emergent-sensing`).

In each dataset an individual representation *z* is estimated from private-phase movement and its transfer is tested
in four directions: private processing (search → tracking parameters), early behaviour → later visibility roles,
movement → voluntary payoff sharing, and individual *z* → heterogeneous agent-based simulation → group performance.
Group performance is predicted from the members' individually measured traits in all three datasets; the composition
effect (within-group heterogeneity beyond member means) is tested per dataset and pooled with an equivalence test.

## Repository layout

```
AGENT_SPEC_COLLECTIVE.md        execution specification written before the analysis (Japanese; v1.0, 2026-09-24)
collective_id_research_plan.md  research plan written before the analysis (Japanese; v0.1, 2026-09-24)
src/collective_id/              library: features, split, models, phase 1/2 routines, vectorised environment
scripts/                        numbered pipeline scripts (Python, R, shell); scripts/r/ brms teacher; scripts/tests/
configs/                        experiment constants (configs/data.yaml) and phase 1 defaults (configs/phase1.yaml)
results/                        aggregate results (CSV / JSON / MD) and figures, per phase and dataset
results/paper_figures/          one panel = one PDF (+ PNG preview); INDEX.md lists caption material and sizes
reports/                        phase reports, replication reports, data audit, diagnostics, follow-ups, decision log
paper/figures/                  assembled manuscript figures Fig1-Fig6 and S1_Fig-S5_Fig (PDF + PNG), assemble.py, SIZES.txt
pyproject.toml, uv.lock         Python environment (uv)
```

Reports: `reports/phase0_report.md` (data audit and reproduction of the original analyses), `phase1_report.md`
(search → tracking transfer), `phase2_report.md` (early behaviour → roles; payoff sharing), `phase3_report.md`
(group prediction by simulation), `wu2025_replication.md` and `hawkins2023_replication.md` (+ `_tables.md`),
`diagnostics_pre_phase3.md`, `followup_2026-10-21.md`, `summary_and_assessment_2026-10-20.md`, `data_audit.md`,
and `questions.md` (the decision log: every question raised by the agent and the decision taken, dated).

## Reproduction

### Environment

- Python 3.12 with [uv](https://docs.astral.sh/uv/): `uv sync` installs the locked environment (`uv.lock`), including
  PyTorch 2.11 from the CUDA 12.8 index (`[tool.uv.sources]` in `pyproject.toml`; change the index for a CPU-only
  machine) and `mesa==2.2.4` for the original agent-based model (ABM) of Chirkov et al.
- R 4.4.3 with brms 2.23 and cmdstanr is needed only for `scripts/r/fit_rotation_models.R` (validation of the
  empirical-Bayes teacher against the original Bayesian fit) and for `scripts/60*.R` (extraction of the published Stan
  fits of Wu et al.; base R + jsonlite only).
- GPU (the analysis used one RTX 5080, 16 GB): the empirical-Bayes teacher (`scripts/05`), the sequence encoders of
  phase 1 (`scripts/10`), the phase 2 CNN variant (`scripts/33`) and the phase 3 D-2 neural policy (`scripts/51-52`).
  Everything else runs on CPU; the ABM simulations use multiprocessing (24 cores were used; set
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for the multi-process scripts, otherwise BLAS oversubscription makes them
  about 10x slower).
- All random seeds are fixed in the scripts and configs (fold split seed 0; model seeds 0, 1, 2; simulation seeds are
  derived from fixed generators). Paths are resolved relative to the repository, so run every script from the
  repository root: `uv run python scripts/NN_....py`.

### Data (not redistributed; place as follows)

| Dataset | Source | Licence | Placement |
|---|---|---|---|
| Chirkov et al. 2026 | OSF project [ZRHWT](https://osf.io/zrhwt/) (`data/`, `code/`) | code MIT (V. Chirkov 2026), data CC0 1.0 | `data/raw/osf/` by `scripts/00_download_osf.py`; the OSF `code/` directory copied to `external/analysis/` (the ABM is run unmodified) |
| Wu et al. 2025 | `git clone https://github.com/charleywu/minecraftforaging` | GPL-3.0 | `external/wu2025/`; the Git LFS objects are required (`git lfs pull`, or fetch from `https://media.githubusercontent.com/media/charleywu/minecraftforaging/master/<path>`): `analysis/data/2021batch/*.feather` and the two Stan fits `modeling/Stanfits/s_conditional_ARS_weights.rdata`, `s_solo_ARS_weights.rdata` |
| Hawkins et al. 2023 | `git clone https://github.com/hawkrobe/emergent-sensing` | MIT (R. Hawkins) | `external/hawkins2023/`; Experiment 1 raw tick logs from the authors' archive `https://emergent-sensing.s3.us-east-2.amazonaws.com/exp1_raw_games.zip`, unzipped into `external/hawkins2023/data/experiment1/raw/` (Experiment 3 raw logs are in the repository) |

`scripts/00_download_osf.py` reads `data/raw/osf/_filelist.json`, a JSON list of `[path, size, download_url]` for
every file of the OSF project (obtainable from the OSF API v2 file listing of node `zrhwt`, or with `osfclient`:
`osf -p zrhwt ls`). The download is rate-limited by OSF (HTTP 429); the script retries and can be re-run. ABM
parameter-sweep outputs, prior-only fits and videos are skipped (`--all` fetches them; not needed).

### Execution order and cost

Wall-clock times are those recorded during the analysis (24-core CPU, RTX 5080); "min" = minutes on CPU unless stated.

| Stage | Scripts | What | Recorded cost |
|---|---|---|---|
| Phase 0 | `00` download; `01` audit; `02` behavioural reproduction; `03` ABM reproduction (mesa 2.2.4, original code); `04` 1 Hz features + fixed 10-fold group split | data, reproduction of the original figures | download: hours (9.2 GB, rate-limited); `03`: 3,000 runs in 3.5 min (4 processes) |
| Teacher | `scripts/r/fit_rotation_models.R` (+`run_fit_queue.sh`); `05_teacher_eb.py`; `06_compare_teachers.py`; `09` report | per-participant random effects of the original rotation model | brms, 20 % subsample for validation: 98 min; full brms fit abandoned (> 1 day); empirical-Bayes teacher on GPU: about 10 min |
| Phase 1 | `16_phase1_run_all.sh` (= `11` ridge/PLS baselines, `10` train, `12`/`17` evaluate, `14` simulated-agent recovery, `15` ICC); `20-23` learning-rate sweeps (optional) | search → tracking transfer | encoder training: seconds to a few minutes per fold on GPU (150 fold x seed runs in the main pipeline); 50k-epoch sweep: about 75 GPU-min per fold, about 15 h with 6 parallel runs |
| Diagnostics | `24_diag_abm_heterogeneity.py`; `25_diag_composition_observed.py` | ABM sensitivity; observed composition | `24`: 34,100 runs in about 9 min (20 cores) |
| Phase 2 | `30` build; `31` plan B; `32` plan C; `33` CNN variant; `34` report | first 3 min → roles; payoff sharing | minutes; CNN about 1 GPU-min |
| Phase 3 D-1 | `40` fit policies; `41` leave-one-group-out simulation (+`--tag _mps2`); `42` observed metrics; `43` evaluate; `44` extrapolate; `45` report | group prediction with the original ABM policy | `40`: 3 min; `41`: 202,200 runs in 1.1 h (21 processes) + 51,500 runs in 14 min; `44`: about 50 min (3 processes) |
| Phase 3 D-2 | `50` environment validation; `51` train z-conditioned policy (GPU); `52` simulate | neural policy in the vectorised environment | `52`: 88,900 runs in 4 min (GPU); `51` not timed |
| Wu 2025 | `60`, `60b` (R) Stan extraction; `61` parse; `62` choice-model fits; `63` features; `64` individual; `65` group; `66` simulate; `67` evaluate; `68` tables; `69` figures | replication | `66`: 47,872 runs in about 70 min (20 processes); others minutes |
| Hawkins 2023 | `70` parse; `71` features; `72` teacher; `73` analysis; `74` figures; `75` tables | replication | minutes |
| Follow-ups | `81` pooled composition; `82` in-group selectivity; `83` + `85` Wu mobility ABM; `84` social-rule search; `90` A-H2 with the linear z | 2026-10-21/22 follow-ups | `83`: 44,800 runs in 9 min (16 processes); `84`: 41,200 runs (not timed); others minutes |
| Paper | `80` figure panels; `86` concept figure; `89` SI panels; `87` print-size check; `paper/figures/assemble.py` (LaTeX standalone); `88` numbers | manuscript figures and numbers | minutes |

### Numbers, figures and tables of the manuscript

`scripts/88_paper_numbers.py` regenerates `paper/numbers.md`, the table of every number cited in the manuscript with
its source file. Run `uv run python scripts/88_paper_numbers.py` from a clone of this repository: the output is
byte-identical to the author's copy. Four sections (0, 2c-ter, 4c-bis, 4l) need participant-level inputs that are
not redistributed; for those the script inserts the lines stored in `results/paper_numbers_cache.json` (written when
the script last ran with the full data) and says so on standard output. Every other section is recomputed from the
aggregate files in `results/`.

| `numbers.md` section | Content | Script | Result file(s) |
|---|---|---|---|
| 0 | sample sizes | `40`, `63`, `71`, `73` | `results/phase3/policy_params_all.csv` (*), `data/processed/*` (*), `results/hawkins2023/z_first_half.csv` (*) |
| 1a-1e | split-half reliability | `06`, `64`, `73`, `72`, `40` | `phase0/split_half_reliability.csv`, `wu2025/ind_reliability.csv`, `hawkins2023/reliability.csv`, `hawkins2023/teacher_reliability.csv`, `phase3/policy_reliability.csv` |
| 2a-2e | private → social transfer | `11`, `31`, `64`, `73` | `phase1/baselines_ridge_eb.csv`, `phase2/planB_r.csv`, `planB_delta.csv`, `planB_cluster_bootstrap.csv`, `wu2025/transfer_r.csv`, `hawkins2023/transfer_r.csv` |
| 2c-ter | group-cluster bootstrap of 2c | `88` (cached) | `phase2/planB_oof_predictions.csv` (*) |
| 3a-3c | decomposition of roles | `31`, `65`, `73` | `phase2/planB_decomposition.csv`, `wu2025/decomposition.csv`, `hawkins2023/decomposition.csv` |
| 4a-4e | Chirkov group prediction (D-1, D-2) | `43`, `52`+`43` | `phase3/eval_r.csv`, `eval_delta.csv`, `shuffle_test.csv`, `eval_r_d2.csv`, `eval_delta_d2.csv` |
| 4c-bis | superseded shuffle null | `88` (cached) | `phase3/sim_runs_mps_v1_no_persistence_match.parquet` (*) |
| 4f-4k | Wu and Hawkins group prediction | `65`, `67`, `85`, `73` | `wu2025/group_pred.csv`, `sim_eval.csv`, `sim_mobility_eval.csv`, `sim_mobility_delta.csv`, `sim_mobility_calibration.csv`, `hawkins2023/group_pred.csv` |
| 4l | mobility vs individual performance | `88` (cached) | `data/processed/*` (*), `phase3/policy_params_all.csv` (*) |
| 5a-5c | composition effect, pooled equivalence | `81`, `82`, `43`, `65`, `73` | `pooled/composition_pooled.csv`, `pooled/ingroup_selectivity_*.csv`, `phase3/composition_regression.csv`, `wu2025/group_composition.csv`, `hawkins2023/group_composition.csv` |
| 6a-6b | social-rule search | `84` | `phase3/social_rules_search.csv`, `social_rules_levels.csv` |
| 7 | power calculations | `88` | computed in the script |
| 8a-8b | cross-individual transfer (A-H2) | `90` | `phase1/ah2_linear_z.csv`, `pooled/ah2_wu.csv`, `pooled/ah2_hawkins.csv` |
| 8c-8d | voluntary payoff sharing (plan C) | `32` | `phase2/planC_r.csv`, `planC_H2_correlations.csv` |

(*) participant-level or run-level input, not redistributed (see below); the section is served from the cache.

Figures: every panel is one PDF in `results/paper_figures/` (`INDEX.md` gives the caption material, the source
files and the PDF size of each panel). `paper/figures/assemble.py` (LaTeX `standalone`) places the panels into the
submitted figures; `paper/figures/SIZES.txt` records their physical sizes.

| Manuscript figure | Panels (`results/paper_figures/`) | Script |
|---|---|---|
| Fig 1 | `fig0_concept.pdf` | `86` |
| Fig 2 | `fig1_reliability.pdf` + `fig1_legend.pdf` | `80` |
| Fig 3 | `fig2a_transfer_chirkov.pdf`, `fig2b_transfer_wu.pdf`, `fig2c_transfer_hawkins.pdf` + `fig2_legend.pdf` | `80` |
| Fig 4 | `fig3_roles_decomposition.pdf` | `80` |
| Fig 5 | `fig4a_group_chirkov.pdf`, `fig4b_group_wu.pdf`, `fig4c_group_hawkins.pdf` | `80` |
| Fig 6 | `fig5_composition_pooled.pdf` + `fig1_legend.pdf` | `80` (data from `81`) |
| S1 Fig | `si/s1_*` (D-1: observed vs predicted, all metrics; controls) | `89` |
| S2 Fig | `si/s2_*` (D-2, likewise) | `89` |
| S3 Fig | `si/s5_a`, `si/s5_b`, `si/s5_legend` (vectorised environment vs the original ABM) | `89` |
| S4 Fig | `si/s3_a`-`s3_d` (Wu grid ABM without / with the mobility term) | `89` |
| S5 Fig | `si/s4_a_heatmap`, `si/s4_colorbar`, `si/s4_b1`-`s4_b4` (social-rule search) | `89` |

Tables of the supplementary information are built from the same `numbers.md` sections (8c, 8d, 5b, 5c and 6).

## What is not included, and how to regenerate it

| Excluded | Why | How to regenerate |
|---|---|---|
| `data/raw/`, `data/processed/` | participant-level data of the three studies (raw downloads and derived 1 Hz / 20 Hz / 8 Hz feature tables, fold assignments) | `scripts/00-04` (Chirkov; download hours, features minutes), `61-63` (Wu; minutes), `70-71` (Hawkins; minutes) |
| `external/` | third-party code and data (Chirkov OSF `code/`, Wu and Hawkins repositories); not redistributed, see the licences above | clone / download as in *Data* |
| `results/**` participant-level files: per-participant teacher weights, *z*, policy parameters and predictions (`phase1/pred_*`, `z_*`, `sim_z_*`, `nll_b_phase1_b_z4_eb.csv`, `phase2/*oof_predictions.csv`, `z_planB_*oof.csv`, `z_planB_cnn_seed*.csv`, `phase3/policy_params_*.csv`, `d2_z.csv`, `d2_predictive.csv`, `policy_vs_linear_predictive.csv`, `diagnostics/composition_individuals.csv`, `hawkins2023/teacher_weights.csv`, `z_first_half.csv`, `wu2025/choice_weights.csv`, `z_solo_pca4.csv`, `sim_mobility_params.csv`, `stan_vID_*.csv`, `phase0/audit_*_per_participant.csv`); pair-level transfer errors (`phase1/cross_individual_pairs_*.csv`, `ah2_linear_z_pairs.csv`) | one row per participant (or pair of participants) of a public dataset; re-created from the public data | `05`, `11`, `12`, `17`, `31-33`, `40`, `51`, `25`, `72-73`, `60`, `62`, `64`, `83`, `01`, `90` (see the cost table); also in the supplementary Zenodo archive 10.5281/zenodo.23583135 |
| `results/**/sim_runs*.parquet`, `phase3/social_rules_runs.parquet`, `diagnostics/abm_het_real_groups.csv` | raw seed x group simulation outputs (up to 202,200 rows per file); the seed-mean evaluations derived from them are included | `41`, `52`, `66`, `83`, `84`, `24` (see the cost table); also in the supplementary Zenodo archive 10.5281/zenodo.23583135 |
| `results/phase1/fig_*_lr*`, `*_ep*k_*`, `*_tuned*` figures | per-variant diagnostic figures of the learning-rate sweeps (their summary tables and verdict files are included) | `12` on each sweep variant |
| `paper/figures/*.tif`, `paper/figures/build/` | 600 dpi submission TIFFs (in the supplementary Zenodo archive 10.5281/zenodo.23583135) and LaTeX build files | `paper/figures/assemble.py` |
| `paper/numbers.md`, `paper/figure_legends.md`, `paper/WRITING_BRIEF.md` | manuscript drafting material; `numbers.md` is regenerated by `scripts/88` | `uv run python scripts/88_paper_numbers.py` |
| `reports/HANDOVER.md`, `reports/summary_undergrad_*.html` | internal hand-over notes and a lay summary (Japanese) | not needed |
| `refs/`, `runs/`, `logs/` | third-party PDFs; model checkpoints, brms fits and logs | `runs/` is re-created by the scripts |

## Pre-specified analysis plan and decision log

`AGENT_SPEC_COLLECTIVE.md` (execution specification, with the hypotheses, pass/fail criteria and prohibitions of each
phase) and `collective_id_research_plan.md` (research plan) were written before any analysis and committed first
(commit `0a7c67d02ce0d8858662d69e13a9d5726724e28e`, 2026-09-24; the hash of the same commit in the author's private history,
cited in the OSF registration, is `20bce998dbe4d9fb449ffd7a7a84930b7b640a09`; the two differ only because the
excluded files were removed from the public history). Both documents are reproduced verbatim (Japanese) and are also
registered on OSF together with the decision log. This is **not** a pre-registration: the plan was not deposited
before data access, and several pre-specified tests were changed during the analysis. Every such change is dated in
`reports/questions.md` and in the phase reports (for example, the group-cluster bootstrap adopted for hypothesis
B-H1 on 2026-10-20, the reduced form of phase 3, the fixed ABM noise parameter).

## Use of an AI coding agent

The analysis was implemented and executed by an AI coding agent (Claude Code, Anthropic) following the specification
in `AGENT_SPEC_COLLECTIVE.md`; the author wrote the specification and the research plan, took every decision recorded
in `reports/questions.md`, verified the results against the original publications and the source data, and wrote the
manuscript. The agent's reports (`reports/`) are reproduced as written at the time; where the manuscript and a report
disagree, `paper/numbers.md` (generated from the result files) is authoritative. This repository, including this
README, was assembled by the same agent from the private working repository and reviewed by the author.

## Licence, citation, contact

- Code (`src/`, `scripts/`, `configs/`, `paper/figures/assemble.py`): MIT License, see `LICENSE`.
- Documents, reports, results and figures: Creative Commons Attribution 4.0 International (CC BY 4.0), see
  `LICENSE-docs`. Third-party data and code are **not** included and keep their own licences (see *Data*).
- Please cite the paper (above) and, for the code, this repository via `CITATION.cff` (Zenodo concept DOI 10.5281/zenodo.23582821).
- Contact: Hiroshi Higashi, hgshrs@kansai-u.ac.jp
