"""Phase 0 data audit (AGENT_SPEC §2.2): expected vs measured. Writes results/phase0/audit_*.csv and reports/data_audit.md."""
import json
import numpy as np, pandas as pd, polars as pl
from collective_id.paths import RAW_DATA, RESULTS, REPORTS, RAW
out = RESULTS / "phase0"; out.mkdir(parents=True, exist_ok=True)
L = []
def sec(t): L.append(f"\n## {t}\n")
def row(t): L.append(t)

s = pd.read_csv(RAW_DATA / "processed/subject_data.csv")
ts = pl.read_csv(RAW_DATA / "processed/time_series_data.csv")
parts = ts.select("participant", "signaling", "resource").unique().to_pandas()
s_an = s[s.player_id.isin(parts.participant)].copy()  # analysed participants (paper's 621)

sec("Files on OSF (osfstorage)")
fl = json.load(open(RAW / "_filelist.json"))
row(f"- {len(fl)} files, {sum((x[1] or 0) for x in fl)/1e9:.2f} GB total; downloaded selectively (no parameter_sweep outputs, no *_prior.rds, no videos).")

sec("Participants")
row(f"- subject_data.csv rows: {len(s)} (all recruited main-experiment participants with a trace file); is_valid==True: {int(s.is_valid.sum())}.")
row(f"- Paper's analysed sample = participants in time_series_data.csv: **{len(parts)}** (expected 621).")
ct = pd.crosstab(parts.resource, parts.signaling)
exp = pd.DataFrame({"A": [73, 71], "FP": [74, 86], "NP": [72, 80], "VP": [86, 79]}, index=["fast", "slow"])
row("- Per cell (measured / expected):")
for r in ct.index:
    row("  - " + r + ": " + ", ".join(f"{c}={ct.loc[r,c]}/{exp.loc[r,c]}" for c in ["A", "NP", "VP", "FP"]))
row(f"- All cells match: **{bool((ct[exp.columns] == exp).all().all())}**")
row("- Exclusion logic (index.Rmd): is_valid & duplicates_percentage < 25 & (valid_group_size >= 4 | A). is_valid (step 1): nan% < 5 and (A: resource duplicate frames < 5%; groups: time_in_experiment < 15.75 min).")
inv = s[~s.player_id.isin(parts.participant)]
row(f"- Not analysed: {len(inv)} (is_valid False: {int((~inv.is_valid).sum())}; valid but dup>=25% or group<4: {int(inv.is_valid.sum())}).")
ct.to_csv(out / "audit_participants_per_cell.csv")

sec("Groups")
g = s_an[s_an.condition_signaling != "A"].groupby("group").agg(n=("player_id", "size"), cond=("condition_signaling", "first"), res=("condition_resource", "first"))
row(f"- Groups in analysed sample: **{len(g)}** (expected ≈95). Size 5: {int((g.n==5).sum())}, size 4: {int((g.n==4).sum())} (paper keeps groups with one invalid member).")
gt = pd.crosstab([g.res, g.cond], g.n); gt.to_csv(out / "audit_groups_by_cell.csv")
row("- Groups by cell × size:\n\n" + gt.to_markdown() + "\n")
# 4-person groups: is the 5th member's trace present?
g4 = g[g.n == 4].index
s_all_in_g4 = s[s.group.isin(g4)]
row(f"- For the {len(g4)} four-person groups, subject_data lists {len(s_all_in_g4)} members in total ({len(s_all_in_g4)-4*len(g4)} excluded 5th members whose raw 10 Hz traces exist in processed_traces.csv but are excluded from analysis). The excluded member still physically influenced the group.")

sec("Time series (1 Hz, time_series_data.csv)")
sp = ts.group_by("participant").agg(pl.col("step").min().alias("smin"), pl.col("step").max().alias("smax"), pl.len().alias("n")).to_pandas()
row(f"- rows: {ts.height}; steps per participant: min {sp.n.min()}, max {sp.n.max()} (step range {sp.smin.min():.0f}–{sp.smax.max():.0f}; the paper drops the first 20 s and last 20 s of the 900 s session → 861 steps).")
st = ts.group_by("state").len().to_pandas().set_index("state")["len"]
row(f"- state counts: " + ", ".join(f"{k}={v} ({v/ts.height*100:.1f}%)" for k, v in st.items()) + ". 'mixed' = 1 s windows crossing the 40 m boundary (excluded by the paper's tracking/search models).")
nulls = ts.null_count().to_pandas().T; nulls = nulls[nulls[0] > 0]
row("- Columns with nulls (social features undefined when nobody is visible): " + ", ".join(f"{i}={int(v)}" for i, v in nulls[0].items()))
row("- Key variable mapping used by the paper's models (index.Rmd): Rotation=rotation_change_mean_raw (net rotation deg/s, clipped ±15 deg per frame), DetectorChange=distance_change_lag1_cat2 (binned −6..6; Detector=−DetectorChange/6), OwnDistFromResource=first_distance_from_resource, VisPlayersDirection=mean_degree_visible_players_first, SocialMovementDirection=social_movement_direction, SocInfoQualityRaw=soc_info_quality_raw (mean visible-other distance − own distance).")
sp.to_csv(out / "audit_steps_per_participant.csv", index=False)

sec("Time series (10 Hz, processed_traces.csv)")
lf = pl.scan_csv(RAW_DATA / "processed_traces/processed_traces.csv", infer_schema_length=100000)
q = lf.group_by("participant").agg(pl.len().alias("n"), pl.col("time").min().alias("tmin"), pl.col("time").max().alias("tmax"), pl.col("x").is_null().sum().alias("x_null")).collect().to_pandas()
row(f"- participants: {len(q)} (includes the {len(q)-len(parts)} non-analysed); samples per participant: median {q.n.median():.0f}, min {q.n.min()}, max {q.n.max()}; time 0–{q.tmax.max():.1f} s at 10 Hz (expected 9000 samples).")
qa = q[q.participant.isin(parts.participant)]
row(f"- analysed participants: samples min {qa.n.min()}, max {qa.n.max()}; participants with any null x: {int((qa.x_null>0).sum())}, max null frames {qa.x_null.max()} ({qa.x_null.max()/90:.2f}% of 9000).")
row("- columns: " + ", ".join(lf.collect_schema().names()))
row("- Not included at 10 Hz: score (recomputable from distance: 40−d points/s inside 40 m; VP sharing cost 1 pt/s) and resource position (available in raw_traces).")
q.to_csv(out / "audit_10hz_per_participant.csv", index=False)

sec("Raw traces (data/raw_traces)")
row("- Format: per session file (1 or 5 player IDs in the file name). .csv columns: time(ms), player_id ('resource' or ID), x, z, rotation, signaling, score. .txt: Time_ms id positions_x positions_z rotation_y is_signaling score (ids 0..4 = file-name order, resource row first). 10 Hz.")
row("- Score column present in raw traces (needed for VP net/gross scores). Download of all 580 raw-trace files is rate-limited by OSF (HTTP 429); completion is tracked in logs/download.log.")

sec("Constants (paper Methods) — to be verified against traces")
row("- resource radius 40 m, 40 pts/s centre → 1 pt/s edge; speed slow 2.8 m/s (40%), fast 5.6 m/s (80%); avatar 7 m/s, 150°/s; FOV 90° (code uses Unity's 91.49284°, visibility if |angle| < 45.75°); arena 400 m diameter; 15 min.")
v = s_an
row(f"- subject_data: time_in_experiment {v.time_in_experiment.min():.2f}–{v.time_in_experiment.max():.2f} min; duplicates_percentage max {v.duplicates_percentage.max():.0f}%; final_score {v.final_score.min()}–{v.final_score.max()}; time_on_resource {v.time_on_resource.min():.2f}–{v.time_on_resource.max():.2f} min.")

sec("Discrepancies / notes")
row("- Groups: 103 (65 of size 5, 38 of size 4) vs the plan's ≈95 estimate ((621−144)/5 = 95.4 assumes all groups of 5). 4-person groups are kept as in the paper; the leave-one-group-out design in Phase 3 must handle n=4 groups (simulate 4 agents, or 5 with the excluded member — decision logged in questions.md).")
row("- Resource discovery events: paper text says 'all searching for at least 5 s ... stayed for at least 5 s'; code (localize_events) uses 20 frames = 2 s for both, with a 2 s offset removed in R. Measured definition (code) is used here.")
row("- Frame-rate exclusion (paper: >5% below 10 Hz) is implemented as: A → resource duplicate frames <5%; groups → time_in_experiment <15.75 min. No per-frame frame-rate variable exists; duplicates_percentage (A) and time_in_experiment (groups) are the available nuisance proxies.")

(REPORTS / "data_audit.md").write_text("# Data audit — OSF ZRHWT (Chirkov et al. 2026)\n\nGenerated by scripts/01_data_audit.py on 2026-09-24. All numbers measured from the downloaded files.\n" + "\n".join(L) + "\n")
print("\n".join(L))
