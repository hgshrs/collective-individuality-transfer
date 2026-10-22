# Paper figures (results/paper_figures). Single-panel PDFs; fonts for PLOS Computational Biology (8-12 pt at final size): axis labels 9 pt, ticks/legend/annotations 8 pt, Fig 1 point labels 8 pt, concept figure 8-9 pt. Panels are placed 1:1 in the assembled submission figures paper/figures/Fig1-6 (assemble.py; TIFF 600 dpi). Mapping: Fig1 = fig0_concept; Fig2 = fig1_reliability + fig1_legend; Fig3 = fig2a-c + fig2_legend; Fig4 = fig3_roles_decomposition; Fig5 = fig4a-c; Fig6 = fig5_composition_pooled + fig1_legend (fig5_composition kept as the unpooled variant). Colours tab10: C0 blue, C1 orange, C2 green, C3 red, C4 purple. History: 2026-10-21 Fig 1 labels enlarged, Fig 2/4/5 narrowed, fig0/fig5_pooled added, then all fonts raised to PLOS sizes (data unchanged).

fig0_concept.pdf	Conceptual figure (scripts/86): individual style z extracted from the private phase (alone, or away from the resource) and the four directions in which its transfer was tested: (1) private processing in the group (transfers up to the split-half ceiling, r 0.3-0.7), (2) social attraction (partial, r 0.1-0.6), (3) selective social learning (reliable in the group, 0.45-0.55, but not transferable, r ~ 0; dashed arrow), (4) roles and group performance (own z + others' mean, additive; group performance = members' mean, no composition effect). Same pattern in all three datasets.
fig1_legend.pdf	Legend for fig1_reliability and fig5_composition: dataset colours (tab10 C0 Chirkov, C1 Wu, C2 Hawkins) (legend only; saved with ~2 % padding, no further trim needed)
fig1_reliability.pdf	Split-half reliability (first vs second half of the session) of individual parameters, grouped by trait class, for the three datasets (colour = dataset; each point is one parameter, named beside it). Private processing 0.60-0.83; social attraction 0.17-0.61; selective (payoff-dependent) social learning 0.09-0.55 but <= 0.27 for the Chirkov quality interactions and the Hawkins selectivity measures; being attended to 0.22-0.24. Sources: results/phase0/split_half_reliability.csv, results/wu2025/{choice_reliability,ind_reliability}.csv, results/hawkins2023/{reliability,teacher_reliability}.csv.
fig2a_transfer_chirkov.pdf	Chirkov et al.: out-of-fold r of tracking-phase individual parameters predicted from a linear search-phase z (PLS, 4 dims; n = 568), and of later visibility roles predicted from the first-3-min movement z (PCA 4; n = 477). Black ticks = split-half reliability of the target (none computed for roles). Sources: results/phase1/baselines_ridge_eb.csv (pls_z4), results/phase2/planB_r.csv.
fig2b_transfer_wu.pdf	Wu et al.: out-of-fold r (8 folds of 4 sessions, n = 128) of group-round targets predicted from the solo-round z (PCA 4 of 21 movement/foraging statistics). Black ticks = split-half reliability of the target across group rounds. Source: results/wu2025/transfer_r.csv (solo_z4).
fig2c_transfer_hawkins.pdf	Hawkins et al. Exp 1: out-of-fold r (10 folds of whole games, n = 682, within experiment x group size x field) of second-half targets predicted from the first-half search-state z (PCA 4). Black ticks = first-vs-second-half reliability of the target. Source: results/hawkins2023/transfer_r.csv (z4).
fig2_legend.pdf	Legend shared by fig2a-c: bar colour = trait class of the target; black tick = split-half ceiling (legend only; saved with ~2 % padding, no further trim needed)
fig3_roles_decomposition.pdf	Increment in marginal R2 of visibility roles (being attended to = in-degree; attending to others = out-degree) when adding, in order, own z, the other members' mean z and the within-group SD of z to a mixed model with group random intercept and a performance control. Group SD adds <= 0.012 everywhere (LRT p 0.22-0.95, except Hawkins attended p 0.048). Sources: results/phase2/planB_decomposition.csv (control = score_3min), results/wu2025/decomposition.csv, results/hawkins2023/decomposition.csv.
fig4a_group_chirkov.pdf	Chirkov et al.: observed group score vs the score predicted by simulating each group with its members' individually fitted ABM policies (leave-one-group-out, seed mean, 103 groups; z within condition x speed). r = 0.55 [0.42, 0.67]. Sources: results/phase3/sim_runs.parquet (variant main), results/phase3/observed_group_metrics.csv.
fig4b_group_wu.pdf	Wu et al.: observed group reward (mean of the 4 group rounds per session x environment, 64 cells) vs the zero-shot prediction from the members' mean solo-round reward in the same environment (z within environment). r = 0.55. Source: results/wu2025/group_rounds.csv.
fig4c_group_hawkins.pdf	Hawkins et al.: second-half group score vs the members' mean first-half individual score (281 games with >= 2 players, Exp 1 and 3; both residualised within experiment x group size x field). r = 0.55 [0.44, 0.64]. Source: results/hawkins2023/group_table.csv.
fig5_composition.pdf	Composition effect on group performance across datasets: change in out-of-sample r when within-group heterogeneity is added to a model that already carries the members' mean traits (Chirkov: individually fitted agents minus group-mean agents, D-1 ABM policy and D-2 neural policy, 103 groups; Wu: same contrast in the authors' grid ABM, 64 session x env cells, and member means + SDs minus means in leave-one-session-out ridge, 256 group rounds; Hawkins: means + SDs minus means, 169 games with >= 3 players). Session/group-cluster bootstrap 95 % CIs. Sources: results/phase3/eval_delta*.csv, results/wu2025/{sim_eval_delta,group_pred}.csv, results/hawkins2023/group_pred.csv.
fig5_composition_pooled.pdf	As fig5_composition plus the pooled composition effect (scripts/81; results/pooled/composition_pooled.csv): stratified cluster bootstrap over all units; 'one per dataset' = Chirkov D-1 ABM + Wu statistical + Hawkins statistical (528 units, 304 clusters): Delta r -0.04 [-0.07, -0.01], one-sided 95 % upper bound -0.01, TOST equivalence bound 0.06; 'all five' contrasts (695 units): Delta r +0.01 [-0.03, 0.05], upper bound 0.04. Black diamonds = pooled.

fig0_concept.pdf	245.98 x 180 pts
fig1_legend.pdf	135.019 x 37.2022 pts
fig1_reliability.pdf	246.016 x 233.418 pts
fig2_legend.pdf	283.35 x 25.6065 pts
fig2a_transfer_chirkov.pdf	241.545 x 166.89 pts
fig2b_transfer_wu.pdf	236.029 x 166.89 pts
fig2c_transfer_hawkins.pdf	227.954 x 166.89 pts
fig3_roles_decomposition.pdf	241.338 x 189.344 pts
fig4a_group_chirkov.pdf	153.908 x 160.119 pts
fig4b_group_wu.pdf	153.908 x 160.119 pts
fig4c_group_hawkins.pdf	153.908 x 160.266 pts
fig5_composition.pdf	244.644 x 144.714 pts
fig5_composition_pooled.pdf	246.316 x 166.89 pts

## Supporting information panels (results/paper_figures/si, scripts/89; assembled into paper/figures/S1_Fig-S5_Fig by assemble.py). S1_Fig = s1_a-h + s1_i + s1_legend (4 x 2 scatter panels, controls bar, legend); S2_Fig = s2_* likewise (D-2); S3_Fig = s3_a, s3_b, s3_c (row) + s3_d; S4_Fig = s4_a_heatmap + s4_colorbar (row) + 2 x 2 block s4_b1-b4; S5_Fig = s5_a, s5_b + s5_legend. Same fonts as the main panels (ticks/legend/annotations 8 pt, labels 9 pt).

si/s1_a_score.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Score, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_b_tracking_time.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Tracking time, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_c_pcr.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Efficiency, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_d_med_dist.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Median distance, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_e_in_deg.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) In-degree, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_f_out_deg.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Out-degree, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_g_in_out_corr.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) In--out correlation, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 103 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_h_disc_gain.pdf	Chirkov D-1 (ABM policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Discoverer gain, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r.csv; n = 77 groups. Sources: results/phase3/sim_runs.parquet, observed_group_metrics.csv, eval_r.csv.
si/s1_i_controls.pdf	Chirkov D-1 (ABM policy): r within cell between observed group metrics and the seed-mean prediction of each simulation variant (main, ctrl2b_gmean, ctrl3_nominal, ctrl4b_mps, ctrl4_shuffle, ctrl1_paper, ctrl2_avg); out-degree r ~ 0.48 for every variant reflects group size (4 vs 5 members) and is not policy information. Source: results/phase3/eval_r.csv.
si/s1_legend.pdf	Legend for s1_i_controls: simulation variants (legend only; saved with ~2 % padding, no further trim needed)
si/s2_a_score.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Score, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_b_tracking_time.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Tracking time, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_c_pcr.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Efficiency, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_d_med_dist.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Median distance, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_e_in_deg.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) In-degree, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_f_out_deg.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Out-degree, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_g_in_out_corr.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) In--out correlation, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 103 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_h_disc_gain.pdf	Chirkov D-2 (neural policy): observed vs predicted (seed mean of the members'-own-policies simulation, leave-one-group-out) Discoverer gain, z within condition x speed; blue slow, orange fast; r [95 % CI] from eval_r_d2.csv; n = 77 groups. Sources: results/phase3/sim_runs_d2.parquet, observed_group_metrics.csv, eval_r_d2.csv.
si/s2_i_controls.pdf	Chirkov D-2 (neural policy): r within cell between observed group metrics and the seed-mean prediction of each simulation variant (main, ctrl2b_gmean, ctrl3_nominal, ctrl1_pop, ctrl2_avg); out-degree r ~ 0.48 for every variant reflects group size (4 vs 5 members) and is not policy information. Source: results/phase3/eval_r_d2.csv.
si/s2_legend.pdf	Legend for s2_i_controls: simulation variants (legend only; saved with ~2 % padding, no further trim needed)
si/s3_a_no_mobility.pdf	Wu: observed group reward (mean of the 4 group rounds per session x environment, 64 cells) vs the zero-shot grid-ABM simulation with the members' fitted weights, without mobility term (variant main); blue random, orange smooth environment (as fig4b); r [95 % CI] from sim_mobility_eval.csv. Sources: results/wu2025/sim_runs.parquet, group_rounds.csv.
si/s3_b_mobility.pdf	Wu: observed group reward (mean of the 4 group rounds per session x environment, 64 cells) vs the zero-shot grid-ABM simulation with the members' fitted weights, with mobility term (variant main_mob); blue random, orange smooth environment (as fig4b); r [95 % CI] from sim_mobility_eval.csv. Sources: results/wu2025/sim_runs_mobility.parquet, group_rounds.csv.
si/s3_c_variants.pdf	Wu: r within environment (64 cells; session-cluster bootstrap 95 % CI) for the four simulation variants without (open) and with (filled) the per-agent mobility term p_i = solo path length / max. Source: results/wu2025/sim_mobility_eval.csv.
si/s3_d_calibration.pdf	Wu: absolute group reward per environment (mean over cells and runs) for the simulation without mobility (main), with mobility (main_mob), group-mean agents with mobility (gmean_mob) and the observed value. Source: results/wu2025/sim_mobility_calibration.csv.
si/s4_a_heatmap.pdf	Chirkov: social-rule search, r within cell between the seed-mean simulation (103 groups x 50 seeds) and the observed group metric for the baseline ABM rule (R0) and seven single-element modifications (R1 nearest other; R2 best-informed quality; R3 scanning while searching; R4 heading alignment; R5 tracker-only social information; R6 calmer tracking; R7 = R5 + R3); colour = r (diverging, symmetric +-0.6; colour bar in s4_colorbar). Source: results/phase3/social_rules_search.csv.
si/s4_colorbar.pdf	Colour bar for s4_a_heatmap (r within cell, -0.6 to +0.6) (colour bar only; bbox tight, ~0 pt padding)
si/s4_b1_disc_gain.pdf	Chirkov: level (mean over the 103 groups) of simulated Discoverer gain for rules R0-R7 (points) and the observed level (dashed line). Source: results/phase3/social_rules_levels.csv.
si/s4_b2_in_out_corr.pdf	Chirkov: level (mean over the 103 groups) of simulated In--out correlation for rules R0-R7 (points) and the observed level (dashed line). Source: results/phase3/social_rules_levels.csv.
si/s4_b3_in_deg.pdf	Chirkov: level (mean over the 103 groups) of simulated In-degree for rules R0-R7 (points) and the observed level (dashed line). Source: results/phase3/social_rules_levels.csv.
si/s4_b4_tracking_time.pdf	Chirkov: level (mean over the 103 groups) of simulated Tracking time for rules R0-R7 (points) and the observed level (dashed line). Source: results/phase3/social_rules_levels.csv.
si/s5_a_tracking_time.pdf	Environment validation: Tracking time of homogeneous paper-parameter agents on the Fig 5B grid (k in 0.1, 1, 10; resource speed 0.2-1.0; solo = triangles, 5 agents = circles; colour = resource speed), mesa ABM (100 runs) vs the vectorised re-implementation (200 runs); dashed = identity; r = 0.997. Source: results/phase3/env_validation.csv.
si/s5_b_pcr.pdf	Environment validation: Efficiency (points per tracking step) of homogeneous paper-parameter agents on the Fig 5B grid (k in 0.1, 1, 10; resource speed 0.2-1.0; solo = triangles, 5 agents = circles; colour = resource speed), mesa ABM (100 runs) vs the vectorised re-implementation (200 runs); dashed = identity; r = 0.996. Source: results/phase3/env_validation.csv.
si/s5_legend.pdf	Legend for s5_a/b: colour = resource speed, marker = solo vs group (legend only; saved with ~2 % padding, no further trim needed)

si/s1_a_score.pdf	125.27 x 121.628 pts
si/s1_b_tracking_time.pdf	125.27 x 121.628 pts
si/s1_c_pcr.pdf	125.27 x 121.628 pts
si/s1_d_med_dist.pdf	125.27 x 121.628 pts
si/s1_e_in_deg.pdf	125.27 x 121.628 pts
si/s1_f_out_deg.pdf	125.27 x 121.628 pts
si/s1_g_in_out_corr.pdf	125.27 x 121.628 pts
si/s1_h_disc_gain.pdf	130.987 x 121.628 pts
si/s1_i_controls.pdf	419.643 x 141.488 pts
si/s1_legend.pdf	423.635 x 29.2646 pts
si/s2_a_score.pdf	125.27 x 121.628 pts
si/s2_b_tracking_time.pdf	125.27 x 121.628 pts
si/s2_c_pcr.pdf	125.27 x 121.628 pts
si/s2_d_med_dist.pdf	125.27 x 121.628 pts
si/s2_e_in_deg.pdf	130.987 x 121.628 pts
si/s2_f_out_deg.pdf	125.27 x 121.628 pts
si/s2_g_in_out_corr.pdf	125.27 x 121.628 pts
si/s2_h_disc_gain.pdf	130.987 x 121.628 pts
si/s2_i_controls.pdf	419.643 x 141.488 pts
si/s2_legend.pdf	306.716 x 29.095 pts
si/s3_a_no_mobility.pdf	155.762 x 163.424 pts
si/s3_b_mobility.pdf	155.762 x 163.424 pts
si/s3_c_variants.pdf	189.714 x 149.872 pts
si/s3_d_calibration.pdf	285.139 x 128.568 pts
si/s4_a_heatmap.pdf	279.186 x 182.34 pts
si/s4_b1_disc_gain.pdf	143.081 x 111.936 pts
si/s4_b2_in_out_corr.pdf	149.667 x 114.686 pts
si/s4_b3_in_deg.pdf	147.512 x 111.936 pts
si/s4_b4_tracking_time.pdf	143.081 x 111.936 pts
si/s4_colorbar.pdf	64.0351 x 177.296 pts
si/s5_a_tracking_time.pdf	170.05 x 163.043 pts
si/s5_b_pcr.pdf	164.695 x 163.208 pts
si/s5_legend.pdf	366.773 x 29.2646 pts
