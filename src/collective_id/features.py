"""1 Hz feature table for Phase 1+ built from the paper's processed 1 Hz table (time_series_data.csv).

Feature definitions follow index.Rmd + rotation_magnitude_{tracking,search}.Rmd exactly, except that the two
angle normalisers use geometric constants (FOV half-width 45.75°, ±180°) instead of sample maxima (see questions.md).
"""
import numpy as np, pandas as pd
from .paths import RAW_DATA, PROCESSED

FOV_HALF = 91.49284 / 2      # Unity FOV used by the paper's code
MAX_ROT = 150.0              # deg/s
COND = ["A", "NP", "VP", "FP"]; SPEED = ["fast", "slow"]

def load_1hz() -> pd.DataFrame:
    ts = pd.read_csv(RAW_DATA / "processed/time_series_data.csv")
    s = pd.read_csv(RAW_DATA / "processed/subject_data.csv")[["player_id", "group", "duplicates_percentage", "time_in_experiment", "final_score", "time_on_resource", "points_collection_speed", "own_signaling_percentage"]]
    return ts.merge(s, left_on="participant", right_on="player_id", how="left").drop(columns="player_id")

def build_features(ts: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"participant": ts.participant, "group": ts.group, "cond": ts.signaling, "speed": ts.resource,
                      "t": ts.step.astype(int), "state": ts.state, "no_move": ts.no_movement_times})
    rot = ts.rotation_change_mean_raw / MAX_ROT                       # net rotation in [-1, 1]
    d["rot_signed"] = rot
    d["rot_abs"] = rot.abs() + 0.01                                    # paper's RotationAbs (Exponential, ub 1.02)
    with_soc = (ts.out_degree > 0)
    d["with_soc"] = with_soc.astype(int)
    d["n_vis"] = ts.out_degree
    d["in_deg"] = ts.in_degree
    # private (tracking only)
    d["detector"] = -ts.distance_change_lag1_cat2.astype(float) / 6.0   # >0: moving away from centre
    d["dist_raw"] = ts.first_distance_from_resource
    d["dist_track"] = (ts.first_distance_from_resource - 20) / 20        # tracking normalisation
    d["dist_search"] = (ts.first_distance_from_resource - 200) / 200     # search normalisation
    # social
    pos = ts.mean_degree_visible_players_first.abs().clip(upper=FOV_HALF) / FOV_HALF
    mov = ts.social_movement_direction.abs().clip(upper=180) / 180
    d["soc_pos"] = np.where(with_soc, (pos - 0.5) * 2, 0.0)
    d["soc_mov"] = np.where(with_soc, (mov - 0.5) * 2, 0.0)
    d["soc_pos"] = d.soc_pos.fillna(0.0); d["soc_mov"] = d.soc_mov.fillna(0.0)
    q = ts.soc_info_quality_raw
    qt = np.select([~with_soc, q < -10, q < 0, q < 10], [0.0, 1.0, 0.5, -0.5], default=-1.0)
    qs_bin = np.where(ts.best_distance_from_resource_visible_players < 40, 1.0, -1.0)
    d["soc_qual_track"] = qt
    d["soc_qual_search"] = np.where(with_soc, qs_bin, 0.0)
    d["soc_qual"] = np.where(ts.state == "tracking", d.soc_qual_track, d.soc_qual_search)
    d["soc_qual_raw"] = q.fillna(0.0)
    d["is_signaling"] = ts.is_signaling.fillna(0.0)
    d["speed_ms"] = np.nan  # filled from 10 Hz later if needed
    for c in ["duplicates_percentage", "time_in_experiment", "final_score", "time_on_resource", "points_collection_speed", "own_signaling_percentage"]:
        d[c] = ts[c]
    return d.sort_values(["participant", "t"]).reset_index(drop=True)

def build_and_save() -> pd.DataFrame:
    d = build_features(load_1hz())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d.to_parquet(PROCESSED / "features_1hz.parquet", index=False)
    return d
