"""§4.2-1 / §4.2-4: upper-bound feel + linear baselines. Predict tracking-phase teacher random effects from
search-phase summary statistics with ridge regression (leave-one-participant-out within the fixed 10-fold scheme:
we use the 10 folds with fold-internal standardisation, which is LOPO-equivalent in spirit and matches the encoder).
Also: condition x speed mean baseline, grand mean baseline, search-time-only baseline."""
import argparse, json, numpy as np, pandas as pd, yaml
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from collective_id.paths import ROOT, RESULTS, PROCESSED, CONFIGS
from collective_id.data import load_features
from collective_id.summary_stats import search_summary
from collective_id.split import load_split
from collective_id.phase1 import load_teacher
ap = argparse.ArgumentParser(); ap.add_argument("--teacher", default=None); ap.add_argument("--tag", default="ridge"); a = ap.parse_args()
cfg = yaml.safe_load(open(CONFIGS / "phase1.yaml")); coefs = cfg["target_coefs"]
teacher, teacher_sd = load_teacher(a.teacher or cfg["teacher"], coefs)
d = load_features(); meta = d.groupby("participant").agg(cond=("cond", "first"), speed=("speed", "first")).loc[teacher.index]
S = search_summary(d).reindex(teacher.index).fillna(0.0)
S.to_csv(PROCESSED / "search_summary_stats.csv")
sp = load_split(); fold = pd.Series({p: sp["fold_of_participant"][p] for p in teacher.index})
onehot = pd.get_dummies(meta.cond.astype(pd.CategoricalDtype(["A", "NP", "VP", "FP"]))).astype(float).join(pd.get_dummies(meta.speed.astype(pd.CategoricalDtype(["fast", "slow"]))).astype(float))
preds = {k: pd.DataFrame(index=teacher.index, columns=coefs, dtype=float) for k in ["grand_mean", "cond_mean", "ridge_summary", "ridge_summary_cond", "search_time_only", "pls_z4"]}
Zlin = pd.DataFrame(index=teacher.index, columns=[f"z{i}" for i in range(4)], dtype=float)  # linear 4-d z (PLS scores) from search summary stats
for f in sorted(fold.unique()):
    tr, te = fold[fold != f].index, fold[fold == f].index
    preds["grand_mean"].loc[te] = teacher.loc[tr].mean().to_numpy()
    cm = teacher.loc[tr].join(meta.loc[tr]).groupby(["cond", "speed"]).mean()
    preds["cond_mean"].loc[te] = cm.loc[list(zip(meta.loc[te].cond, meta.loc[te].speed))].to_numpy()
    mu, sd = S.loc[tr].mean(), S.loc[tr].std() + 1e-9
    Xtr, Xte = ((S.loc[tr] - mu) / sd).to_numpy(), ((S.loc[te] - mu) / sd).to_numpy()
    for k, (A, B) in {"ridge_summary": (Xtr, Xte), "ridge_summary_cond": (np.hstack([Xtr, onehot.loc[tr].to_numpy()]), np.hstack([Xte, onehot.loc[te].to_numpy()])),
                      "search_time_only": (np.hstack([Xtr[:, [S.columns.get_loc("search_time")]], onehot.loc[tr].to_numpy()]), np.hstack([Xte[:, [S.columns.get_loc("search_time")]], onehot.loc[te].to_numpy()]))}.items():
        for c in coefs:
            m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(A, teacher.loc[tr, c]); preds[k].loc[te, c] = m.predict(B)
    # linear z: 4-component PLS from summary stats (+condition one-hot as extra predictors) -> all 6 targets jointly
    Ytr = ((teacher.loc[tr] - teacher.loc[tr].mean()) / (teacher.loc[tr].std() + 1e-9)).to_numpy()
    pls = PLSRegression(n_components=4, scale=False).fit(np.hstack([Xtr, onehot.loc[tr].to_numpy()]), Ytr)
    Bte = np.hstack([Xte, onehot.loc[te].to_numpy()])
    preds["pls_z4"].loc[te] = pls.predict(Bte) * (teacher.loc[tr].std() + 1e-9).to_numpy() + teacher.loc[tr].mean().to_numpy()
    Zlin.loc[te] = pls.transform(Bte)
rows = []
for k, P in preds.items():
    for c in coefs:
        r = np.corrcoef(P[c].astype(float), teacher[c])[0, 1] if P[c].std() > 0 else 0.0
        rows.append(dict(model=k, coef=c, r=r, rmse=float(np.sqrt(((P[c] - teacher[c]) ** 2).mean()))))
    P.to_csv(RESULTS / "phase1" / f"pred_{k}_{a.tag}.csv")
res = pd.DataFrame(rows); res.to_csv(RESULTS / "phase1" / f"baselines_{a.tag}.csv", index=False); Zlin.to_csv(RESULTS / "phase1" / f"z_linear_pls4_{a.tag}.csv")
print(res.pivot(index="coef", columns="model", values="r").round(3))
print("teacher RE sd per coef:", teacher.std().round(3).to_dict())
print("mean posterior sd per coef:", teacher_sd.mean().round(3).to_dict())
