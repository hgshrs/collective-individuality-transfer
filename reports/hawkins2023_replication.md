# Hawkins et al. (2023) replication — Collective-ID analyses on a third dataset (2026-10-21)

Tables: `reports/hawkins2023_replication_tables.md` (`scripts/75_hk_report.py`). Figures: `results/hawkins2023/fig_*.png|pdf`. Scripts 70–75. Decision log: `reports/questions.md` (2026-10-21).

## 0. Why this dataset

Hawkins, Berdahl, Pentland, Tenenbaum, Goodman & Krafft (2023, *Nat Hum Behav* 7:1767) ran a 2-D browser "collective sensing" task: avatars move in a 480 × 285 px arena; a hidden circular scoring region ("spotlight", diameter 50 px) drifts at 17 px/s between random targets; participants see the positions and headings of all others but not their scores. This is the closest public analogue of Chirkov's *moving* resource with hidden payoffs (Chirkov's NP condition). Experiment 1: 294 complete 5-min games (682 participants) in groups of 1–6 (after dropouts: 125 × 1, 65 × 2, 45 × 3, 19 × 4, 24 × 5, 16 × 6), click-to-set-goal, keys for fast/stop, binary score feedback; four pre-generated fields. Experiment 3: 224 complete 6-min games (451 participants; 1–5 players), gradient field (spotlight + spatially correlated noise at 10 % or 25 %), keyboard turning, own score shown as a percentage; the waiting-room logs (≈ 80 s of solitary movement before the game, no score field) are also available. Positions at 8 Hz.

What the dataset gives: a hidden moving target with inference of others' success from their stopping (the paper's point), ≈ 170 groups of ≥ 3 for composition (Exp 1: 104, Exp 3: 65), and a built-in search/scoring state split (outside / inside the spotlight) that mirrors Chirkov's search/tracking phases. What it lacks: no visual-field restriction (everyone sees everyone; roles are "facing" rather than "seeing"), no solo condition for the same person (except Exp 3's waiting room), a single 5–6-min session, and only the first half can serve as the individual measurement for the second half.

## 1. Pipeline

- `70_hk_parse.py`: raw tick logs → `data/processed/hawkins2023/{ticks,games,waiting}.parquet`; completeness and dropout rules as in the authors' R code.
- `71_hk_features.py`: per participant × half (ticks < / ≥ 1200) × state (search = outside the region, score = inside): speed-state fractions, path length, turning, exploration extent, click rate / distance / turn (Exp 1); stop latency after entering the region, fraction of in-region ticks stopped, exits; social: nearest/centroid distance, facing others (45° cone, the authors' criterion), facing stopped vs moving others, selectivity index, being faced (in-degree analogue), approach events (fast + facing one other ≥ 0.5 s). Waiting-room movement statistics for Exp 3.
- `72_hk_teacher.py`: goal-direction model per participant (Exp 1: at each goal click; Exp 3: per second): signed turn ~ angle to others' centroid (b_cen) + angle to the nearest stopped other (b_stop; the payoff-inference weight) + angle to the nearest other (b_nn) + in-region interactions; empirical-Bayes shrinkage towards the population, fitted on all / first / second half.
- `73_hk_analysis.py`: reliability, transfer (first-half search-state z → second-half targets, 10 folds of whole games, within experiment × group size × field), decomposition, group composition.

## 2. Noise ceilings (first vs second half, within experiment × group size × field)

| quantity | Exp 1 | Exp 3 | Chirkov / Wu analogue |
|---|---|---|---|
| private movement statistics (speed-state fractions, path length, turning, click rate/distance/turn) | 0.54–0.92 | 0.51–0.86 | Wu solo stats 0.70–0.97 |
| scoring-state behaviour: fraction stopped inside the region / stop latency / exits | 0.64 / 0.69 / 0.53 | — / — / 0.58 | Chirkov persistence 0.84 |
| social: approach events / nearest-neighbour distance / facing stopped others / selectivity index | 0.61 / 0.59 / 0.55 / 0.25 | 0.63 / 0.61 / 0.59 / 0.42 | Wu out-degree 0.59 |
| being faced (in-degree analogue) | 0.22 | 0.29 | Wu in-degree 0.24 |
| teacher: attraction to others' centroid b_cen / to stopped others b_stop / in-region interactions | 0.43 / 0.16 / ≤ 0.05 | 0.27 / 0.23 / 0.12 | Chirkov SocPos 0.30, SocMov 0.17, quality interactions 0.09–0.12 |
| score | 0.41 | 0.64 | Wu 0.56 |

Same ordering a third time: private movement style and gross social proximity are reliable; the *selective* part of social learning (turning towards stopped = probably-scoring others, selectivity index, in-region interactions) sits at 0.16–0.42; being targeted by others is barely reliable at 0.2–0.3.

## 3. Search-state z

PCA of first-half search-state (outside the region) statistics within experiment (Exp 1: 44/17/13/9 %; Exp 3: 54/27/10/9 %). z0 = speed and activity (fast fraction, path length, turning, click rate); z1 = exploration extent with long straight clicks vs frequent short re-targeting; z2 = turn variability and click-distance variability; z3 = stopping while searching (Exp 1) / coverage (Exp 3).

## 4. Transfer: first-half search phase → second-half targets (out-of-fold r over 682 / 450 participants; Δr vs first-half score; game-cluster bootstrap)

**Exp 1 (binary hidden spotlight, click control)**
- **Scoring-state (tracking) behaviour is predicted**: fraction of in-region time spent stopped r = 0.32 [0.23, 0.40] (ceiling 0.64; Δr vs score +0.28, p < 0.001); stop latency 0.37 (ceiling 0.69; Δr +0.39); number of region entries/exits 0.58 (ceiling 0.52, i.e. at ceiling). This is the Chirkov A-H1 transfer (search phase → tracking-phase private parameters), reproduced with a *moving hidden* target.
- **Gross social proximity/approach is predicted**: approach events 0.56 (ceiling 0.61), nearest-neighbour distance 0.38 (0.59), time within 50 px 0.38 (0.59), facing others 0.19 (0.34). Teacher b_cen 0.53 and b_nn 0.35 are predicted *above* their own split-half reliability (0.43, 0.17), which says these weight estimates are noisy but the underlying attraction trait is carried by movement style (speed, click rate).
- **Payoff-selective social learning is not predicted**: facing *stopped* others 0.06 [−0.02, 0.14] (ceiling 0.55); selectivity index −0.12 (0.25); in-region interaction weights ≈ 0. Teacher b_stop 0.33 vs ceiling 0.16 is the one exception, but b_stop is confounded with b_cen (stopped others are usually near the centroid); the behavioural measures that isolate selectivity (facing stopped *vs* moving, selectivity index) are at zero.
- **Being faced (in-degree analogue)**: 0.17 (ceiling 0.22), not above the first-half score baseline (Δr −0.05).
- **Score**: z 0.44, first-half score 0.41, z + score 0.50 (ceiling 0.41): performance transfers; z adds Δr +0.04 (p 0.18) over first-half score.

**Exp 3 (gradient field, keyboard, plus the pre-game waiting room)**
- Smaller effects overall (no stopping, no clicks: the search-state z is 7 statistics). Approach 0.29 (ceiling 0.63), facing others 0.12, score 0.19 (first-half score 0.63 = ceiling); facing stopped others 0.08, selectivity −0.25, being faced −0.07: no selectivity transfer, as in Exp 1.
- **True solo source (waiting room, ≈ 80 s of movement with no score field)**: predicts second-half approach 0.20 [0.08, 0.30], score 0.26 [0.13, 0.39], b_cen 0.16, nothing else. So even an uncontaminated 80-second solitary sample carries the activity-related part of later social behaviour, and none of the selective part — the Wu 2025 finding with a much shorter solo phase.

## 5. Decomposition of second-half individual outcomes (895 participants in games of ≥ 2; own first-half z → others' mean z → group SD z; game random intercept)

| outcome | + own z | + others' mean z | + group SD z |
|---|---|---|---|
| being faced (in-degree analogue) | p 0.006 | p 0.020 | p 0.048 |
| facing others (out-degree analogue) | p < 0.001 | p 0.09 | p 0.53 |
| facing stopped others | p < 0.001 | p < 0.001 | **p 0.003** |
| approach events | p < 0.001 (R²m 0.01 → 0.18) | p 0.06 | p 0.86 |
| nearest-neighbour distance | p < 0.001 | p < 0.001 (0.06 → 0.14) | p 0.96 |
| score | p < 0.001 | (fit degenerate) | p 0.84 |

Own traits dominate approach and facing; others' mean traits matter for distance and for being faced; the within-group spread adds nothing to performance or spacing. The one clear SD effect, facing stopped others, is mechanical: a group with heterogeneous stopping provides stopped targets to face (z3 loads on stopping). The marginal in-degree SD effect (p 0.048) does not survive the six-outcome comparison.

## 6. Group level (games with ≥ 3 players: Exp 1 104, Exp 3 65; second-half group mean score, within experiment × size × field)

- **Zero-shot**: members' mean first-half score predicts the second-half group score r = 0.55 [0.44, 0.64] (281 games of ≥ 2; raw, un-residualised r 0.95 because group size, field and practice are shared). Leave-one-game-out ridge on member means (z + score) r 0.47; adding the SD block lowers it (Δr −0.03 [−0.07, 0.01]); in-sample ΔR² of the SD block 0.033, permutation p 0.19; each SD's Spearman ρ with group score ≤ 0.12 (p ≥ 0.13); the SD-only model has r < 0.
- Same conclusion as Chirkov Phase 3 and Wu 2025 §6: collective performance follows the members' individual level; heterogeneity in the measured traits contributes nothing detectable with ≈ 170 groups.

## 7. Reading

1. **Third dataset, same structure.** With a moving hidden target and 8 Hz 2-D trajectories, the search phase predicts the tracking phase (stopping inside the region, stop latency: r 0.32–0.37, ceilings 0.64–0.69), gross social attraction/approach transfers (0.38–0.56, at or near ceiling), payoff-selective social learning does not transfer (facing stopped others 0.06 / 0.08 against ceilings 0.55 / 0.59; selectivity ≈ 0), roles are additive (own traits + others' mean; SD only through a mechanical channel), and group performance is the mean of the members (r 0.47–0.55; SD block ΔR² 0.03, p 0.19).
2. **Selective social learning is a social-context trait**, again: here it is reliable within the game (facing stopped others 0.55–0.59) but invisible in the private phase of the same game and in the solitary waiting room — exactly the Wu 2025 pattern (SuccessfulProximity 0.45 reliable, not transferable). Across the three datasets the dissociation is: private processing and general social proximity transfer from solitary behaviour; *whom* one follows does not.
3. **For the paper**: the Chirkov persistence → tracking result, the additive-roles result and the no-composition result now hold in three task families (3-D immersive with visual-field constraint; Minecraft patches; 2-D hidden moving spotlight), ≈ 400 groups in total (103 + 32 + 169 with ≥ 3 members). The negative composition result is no longer a single-dataset, single-measure statement, and its upper bound can be pooled (Δr vs the mean-only model: +0.01, −0.11, −0.03; ΔR² 0.008–0.033).
4. Limits here: first half vs second half is confounded with practice (both predictor and target are in the social context), in-degree analogues are weak (facing cone, no FOV occlusion), the teacher's social weights have low reliability and b_stop is collinear with b_cen, no simulation-based prediction (the authors' particle-filter agents have no per-participant parameters), and groups of ≥ 3 are only 169.
