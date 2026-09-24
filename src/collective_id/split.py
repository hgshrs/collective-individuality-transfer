"""Participant-level 10-fold split, stratified by condition × speed, assigning whole groups to folds (AGENT_SPEC §2.4)."""
import json, numpy as np, pandas as pd
from .paths import PROCESSED

def make_split(meta: pd.DataFrame, n_folds: int = 10, seed: int = 0) -> dict:
    """meta: one row per participant with columns participant, group, cond, speed."""
    rng = np.random.default_rng(seed)
    fold_of_participant = {}
    fold_of_group = {}
    for (cond, speed), m in meta.groupby(["cond", "speed"]):
        units = sorted(m.group.unique()) if cond != "A" else sorted(m.participant.unique())
        units = list(units); rng.shuffle(units)
        # greedy balanced assignment by participant count
        sizes = m.groupby("group").size() if cond != "A" else None
        load = np.zeros(n_folds)
        for u in units:
            n = int(sizes[u]) if cond != "A" else 1
            f = int(np.argmin(load)); load[f] += n
            if cond != "A":
                fold_of_group[u] = f
                for p in m[m.group == u].participant: fold_of_participant[p] = f
            else:
                fold_of_participant[u] = f
    return {"n_folds": n_folds, "seed": seed, "stratify": "cond x speed, group-wise", "fold_of_participant": fold_of_participant, "fold_of_group": fold_of_group}

def save_split(split: dict, name: str = "split_v1.json"):
    json.dump(split, open(PROCESSED / name, "w"), indent=1)

def load_split(name: str = "split_v1.json") -> dict:
    return json.load(open(PROCESSED / name))
