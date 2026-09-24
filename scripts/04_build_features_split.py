"""Build data/processed/features_1hz.parquet and data/processed/split_v1.json (fixed for all phases)."""
import pandas as pd, numpy as np
from collective_id.features import build_and_save
from collective_id.split import make_split, save_split
from collective_id.paths import PROCESSED
d = build_and_save()
meta = d.groupby("participant").agg(group=("group", "first"), cond=("cond", "first"), speed=("speed", "first")).reset_index()
sp = make_split(meta, n_folds=10, seed=0); save_split(sp)
meta["fold"] = meta.participant.map(sp["fold_of_participant"])
print(d.shape); print(d.describe().T[["mean", "std", "min", "max"]])
print(pd.crosstab([meta.cond, meta.speed], meta.fold))
# group integrity
g = meta[meta.cond != "A"].groupby("group").fold.nunique(); assert (g == 1).all(), "group split across folds"
print("groups intact:", (g == 1).all(), "| participants:", len(meta), "| folds:", meta.fold.value_counts().sort_index().tolist())
