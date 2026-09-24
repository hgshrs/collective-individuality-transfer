"""Compare provisional EB teacher with brms teacher (fixed effects and per-participant random effects), and
split-half reliability of random effects for whichever estimator has first/second fits."""
import numpy as np, pandas as pd
from pathlib import Path
from collective_id.paths import RUNS, RESULTS
out = RESULTS / "phase0"; rows = []
for est in ["eb", "brms"]:
    for m in ["tracking", "search"]:
        a, b = RUNS / est / f"{m}_first_ranef.csv", RUNS / est / f"{m}_second_ranef.csv"
        if a.exists() and b.exists():
            A = pd.read_csv(a).pivot(index="participant", columns="coef", values="estimate"); B = pd.read_csv(b).pivot(index="participant", columns="coef", values="estimate"); idx = A.index.intersection(B.index)
            for c in A.columns:
                if "withSocInfo0" in c: continue
                rows.append(dict(estimator=est, model=m, coef=c, r_first_second=np.corrcoef(A.loc[idx, c], B.loc[idx, c])[0, 1], spearman=A.loc[idx, c].corr(B.loc[idx, c], method="spearman"), n=len(idx)))
rel = pd.DataFrame(rows); rel.to_csv(out / "split_half_reliability.csv", index=False); print(rel.round(3).to_string(index=False) if len(rel) else "no split-half fits yet")
cmp_rows = []
for m, e, b, label in [("tracking", RUNS / "eb" / "tracking_all_ranef.csv", RUNS / "brms" / "tracking_all_ranef.csv", "eb_full_vs_brms_full"),
                       ("tracking", RUNS / "eb" / "tracking_all_ranef.csv", RUNS / "brms" / "tracking_all_frac0.2_ranef.csv", "eb_full_vs_brms_sub20"),
                       ("tracking", RUNS / "eb" / "tracking_sub20_ranef.csv", RUNS / "brms" / "tracking_all_frac0.2_ranef.csv", "eb_sub20_vs_brms_sub20"),
                       ("search", RUNS / "eb" / "search_all_ranef.csv", RUNS / "brms" / "search_all_ranef.csv", "eb_full_vs_brms_full")]:
    if e.exists() and b.exists():
        E = pd.read_csv(e).pivot(index="participant", columns="coef", values="estimate"); Bm = pd.read_csv(b).pivot(index="participant", columns="coef", values="estimate"); idx = E.index.intersection(Bm.index)
        for c in E.columns:
            if c in Bm.columns: cmp_rows.append(dict(comparison=label, model=m, coef=c, r_eb_brms=np.corrcoef(E.loc[idx, c], Bm.loc[idx, c])[0, 1], sd_eb=E.loc[idx, c].std(), sd_brms=Bm.loc[idx, c].std(), n=len(idx)))
        fe_e = pd.read_csv(str(e).replace("_ranef", "_fixef")); fe_b = pd.read_csv(str(b).replace("_ranef", "_fixef"))
        fe_e["Estimate_brms"] = fe_e.coef.map(fe_b.set_index("coef").Estimate)   # EB names 'SignalingTypeX:ResourceSpeedY:feat'; brms intercept-cell names differ
        fe_e.to_csv(out / f"fixef_{label}_{m}.csv", index=False)
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(out / "teacher_eb_vs_brms.csv", index=False); print(cmp.round(3).to_string(index=False) if len(cmp) else "brms full fits not yet available")
