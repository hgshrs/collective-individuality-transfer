# Pre-Phase-3 diagnostics: can measured individual heterogeneity matter at the group level?

Date: 2026-10-02. Scripts: `scripts/24_diag_abm_heterogeneity.py`, `scripts/25_diag_composition_observed.py`. Outputs: `results/diagnostics/`.
Motivation: Phase 1 showed that only the tracking-phase **Intercept** (split-half r 0.76) and **Detector** (0.61) random effects are reliably measured within a 15-min session; the social weights (0.09–0.30) are not. Before committing the Phase 3 budget (80 GPU-h + simulation), two cheap go/no-go checks were run. Neither fits anything to group outcomes (AGENT_SPEC §6.2).

## Diagnostic 1 — ABM sensitivity to measured heterogeneity (≈ 9 min, 34,100 runs, 20 cores)

Mapping of the statistical model (|rot| ~ Exp(exp(a_i + b_i·det))) onto the paper's ABM tracking rule (rot = I_i + S_i·det): I_i = 50°·exp(u_i), S_i = 35°·exp(u_i)·(1 + v_i / b_cell), with u_i, v_i the participant's full-session EB random effects (Intercept, Detector) and b_cell the cell fixed effect. k = 1, other parameters as in Fig 5B. No fitting.

**A. Real groups** (103 groups, 100 seeds each): members' mapped parameters (`hetero`), the same group with all members at the group-mean random effects (`homo_mean`), and default agents (`homo_default`, the null).

Between-group ICC of the simulated group outcome (group identity across seeds):

| metric | variant | slow n=4 | slow n=5 | fast n=4 | fast n=5 |
|---|---|---|---|---|---|
| score | hetero | 0.40 | 0.48 | 0.03 | 0.08 |
| score | homo_mean | 0.61 | 0.60 | 0.05 | 0.09 |
| score | homo_default | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| tracking time | hetero | 0.33 | 0.41 | 0.02 | 0.05 |

→ The ABM **is** sensitive to heterogeneity of the measured size, strongly at slow speed (group identity explains 40–60 % of the variance of the 900-s outcome) and weakly at fast speed (3–9 %). The simulated outcome is driven by the group **mean** parameters (r with u_mean, v_mean = 0.6–0.9).

Composition (hetero − homo_mean, same mean, only the spread differs): heterogeneous groups do **worse** than their mean-agent counterpart (slow: −1,300 points, paired p = 0.001; fast: −52, p = 0.001). The difference correlates with the within-group SD of Detector only at fast speed (r = 0.37, p = 0.007; slow r = 0.20, n.s.).

**B. Parametric sweep** (random 5-member draws from the empirical RE pool, scaled by s; 200 seeds):

| s (× measured SD) | slow k=1 score | slow k=10 score | fast k=1 | fast k=10 |
|---|---|---|---|---|
| 0 | 6,558 | 16,494 | 1,415 | 1,740 |
| 0.5 | 7,186 | 16,068 | 1,297 | 1,777 |
| 1 | 6,241 | 14,429 | 1,246 | 1,507 |
| 2 | 4,802 | 9,494 | 1,123 | 1,231 |

→ In the ABM, heterogeneity per se lowers mean group performance monotonically (Jensen-type effect of the exp() mapping: wide-turning members lose the resource). The paper's ABM has no mechanism by which a *mixture* of private-information styles helps.

**C. Zero-shot prediction of the observed groups** (`abm_pred_vs_observed_groups.csv`; observed = mean member final score / time on resource; nothing fitted):

| speed | predictor | r with observed score | r with observed tracking time |
|---|---|---|---|
| slow (52 groups) | ABM hetero | 0.57 | 0.49 |
| slow | ABM homo_mean | 0.63 | 0.52 |
| slow | group-mean Intercept RE (no ABM) | 0.64 | 0.55 |
| fast (51 groups) | ABM hetero | 0.09 | 0.06 |
| fast | ABM homo_mean | 0.08 | 0.08 |
| pooled, z within cell (103) | ABM hetero | 0.38 | 0.32 |
| pooled, z within cell | ABM homo_mean | 0.41 | 0.35 |

Δr (hetero − homo_mean, pooled) = −0.02, bootstrap 95 % CI [−0.14, 0.08]. Caveat: the full-session random effects are estimated from the same session as the outcome, so part of the slow-speed correlation is within-session persistence (the strict first-half → second-half version is Diagnostic 2).

## Diagnostic 2 — observed composition effects (no simulation)

Individual random effects from the **first half** (runs/eb/tracking_first_ranef.csv); group outcomes from the **second half** (t ≥ 450 s: mean member gross score, tracking-time proportion), z-scored within condition × speed. 103 groups (64 × 5, 39 × 4), 476 participants.

Group-level nested OLS (`composition_group_models.csv`); membership-permutation null within cell, 2,000 permutations (`composition_permutation.csv`):

| outcome | RE set | ΔR² means (M1−M0) | perm p | ΔR² SDs (M2−M1) | perm p | LOO-CV R²: M0 / M1 / M2 / M3(+h1) | SD increment given means + h1: p |
|---|---|---|---|---|---|---|---|
| score h2 | Intercept + Detector | 0.163 | < 0.001 | 0.022 | 0.35 | −0.15 / −0.02 / −0.04 / 0.10 | 0.26 |
| tracking h2 | Intercept + Detector | 0.128 | 0.001 | 0.036 | 0.19 | −0.15 / −0.07 / −0.07 / 0.02 | 0.16 |
| score h2 | all 6 | 0.198 | 0.001 | 0.120 | 0.076 | −0.15 / −0.08 / −0.06 / 0.02 | 0.20 |
| tracking h2 | all 6 | 0.162 | 0.014 | 0.137 | 0.047 | −0.15 / −0.13 / −0.09 / −0.04 | 0.10 |

Univariate: within-group SD of Detector correlates negatively with second-half outcome (r = −0.24, p = 0.013 for both outcomes) — the same sign as the ABM — but the increment vanishes once the group means are in the model. The all-6 SD increment is carried by the quality-interaction SD (p ≈ 0.07), i.e. by the least reliable dimension (split-half r 0.12).

Individual-level mixed model (second-half score ~ cell + own 6 REs + others' mean REs + group SD REs + (1 | group), ML, LRT): own Intercept and Detector strongly predict own outcome (b = 0.47, 0.46 SD per RE unit, p < 1e-4; social REs p > 0.2); others' mean REs add nothing (LRT p = 0.27 / 0.20); group SD REs are marginal (p = 0.02 / 0.01, driven by Detector SD negative and PosQual SD positive). A leave-self-out SD definition gave p = 2e-4 but is anti-conservative (own extremeness leaks into the others' SD); the group-SD version matches the permutation test and is the one reported.

## Reading

1. **The ABM can express the heterogeneity we can measure**, and the measured group-mean traits already predict observed group performance at slow speed (r ≈ 0.6 zero-shot, r ≈ 0.4 pooled within cell). D-H1 as pre-registered (Δr ≥ 0.10 vs the homogeneous ABM and vs the average-agent control, which are constant within a cell and therefore have r ≈ 0) is **likely to pass**, essentially through the group means. At fast speed there is almost no between-group signal, in the ABM (ICC 0.03–0.09) or in the data (r ≈ 0.08); Phase 3 power comes from the 52 slow groups.
2. **Composition (D-H2) is unlikely to be demonstrable.** In the observed data the spread of the reliable traits adds ΔR² ≈ 0.02–0.04 (perm p 0.19–0.35), and the ABM predicts no benefit of mixing — heterogeneous groups do slightly worse than their mean-agent counterpart, and the heterogeneous ABM does not predict observed groups better than the mean-agent ABM (Δr −0.02 [−0.14, 0.08]). The only composition signal with a consistent sign (Detector-spread hurts, r ≈ −0.24 univariate, ABM paired effect) is small and not independent of the means.
3. **Caveat on the pre-registered z-shuffle control (§6.3-3, control 4):** replacing a group's members with other participants changes the group *mean* as well as the combination; the shuffle control will therefore lose mainly through the mean effect, and D-H2 would "pass" trivially. A **mean-preserving** shuffle (or the SD-increment regression above) is the informative test. Proposal logged in questions.md; the original control would still be run as written.
4. **Cost:** D-1 with the linear mapping needs no GPU; 103 groups × 200 seeds × 5 variants ≈ 50 min on 20 cores. The expensive parts of Phase 3 (D-2 neural policy, 80 GPU-h) are unlikely to change conclusion 2 because the limiting factor is measurement reliability of the social traits, not policy fidelity.

## Recommendation

Proceed to Phase 3 **in reduced form**: D-1 with the linear RE→ABM mapping (plus the hypernetwork variant only if the linear one fails the individual-level rotation check), leave-one-group-out as specified, all four controls plus a mean-preserving shuffle, and report D-H2 as a composition analysis with effect sizes. Defer D-2 unless D-1 shows a residual that policy fidelity could plausibly explain. Phase 2 (Plan B, roles) remains worthwhile and cheap. The expected headline is "collective performance is predicted by the average measured traits of the members; the combination adds little at n ≈ 100 groups", which answers the original paper's open question in the negative direction for composition and in the positive direction for individual heterogeneity.
