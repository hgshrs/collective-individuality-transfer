"""Search-phase summary statistics per participant (linear baseline for Phase 1, §4.2-1/4)."""
import numpy as np, pandas as pd

def search_summary(d: pd.DataFrame, t_min=None, t_max=None) -> pd.DataFrame:
    m = d.state == "searching"
    if t_min is not None: m &= d.t >= t_min
    if t_max is not None: m &= d.t < t_max
    s = d[m]
    def per(g):
        soc = g[g.with_soc == 1]
        return pd.Series({
            "rot_abs_mean": g.rot_abs.mean(), "rot_abs_sd": g.rot_abs.std(), "rot_signed_mean": g.rot_signed.mean(),
            "rot_zero_frac": (g.rot_abs <= 0.011).mean(), "rot_large_frac": (g.rot_abs > 0.5).mean(),
            "rot_autocorr": g.rot_signed.autocorr(1) if len(g) > 3 else 0.0,
            "soc_frac": (g.with_soc == 1).mean(), "n_vis_mean": g.n_vis.mean(),
            "resp_pos": np.corrcoef(soc.soc_pos, soc.rot_abs)[0, 1] if len(soc) > 5 else 0.0,
            "resp_mov": np.corrcoef(soc.soc_mov, soc.rot_abs)[0, 1] if len(soc) > 5 else 0.0,
            "resp_qual": np.corrcoef(soc.soc_qual_search, soc.rot_abs)[0, 1] if len(soc) > 5 and soc.soc_qual_search.std() > 0 else 0.0,
            "dist_mean": g.dist_raw.mean(), "dist_sd": g.dist_raw.std(), "dist_max": g.dist_raw.max(),
            "search_time": len(g), "no_move_frac": (g.no_move > 0).mean(),
        })
    out = s.groupby("participant").apply(per, include_groups=False).fillna(0.0)
    return out
