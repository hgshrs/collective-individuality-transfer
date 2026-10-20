"""Phase 2 shared utilities: fold-wise linear z (PCA of first-3-min own-movement statistics), ridge prediction,
paired comparisons (participant bootstrap + mixed model with group random intercept), variance decomposition."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning); warnings.filterwarnings("ignore", message=".*singular.*")
from scipy import stats
from .paths import PROCESSED, RESULTS
from .split import load_split

ALPHAS = np.logspace(-2, 4, 13)
STAT_COLS = ["rot_abs_mean", "rot_abs_sd", "rot_abs_p90", "rot_zero_frac", "rot_large_frac", "rot_autocorr", "rot_flip_rate",
             "no_move_frac", "speed_mean", "speed_sd", "moving_frac", "dist_mean", "dist_sd", "dist_min", "dist_max", "track_frac", "n_switch"]
COND = ["NP", "VP", "FP"]; SPEED = ["fast", "slow"]


def load_phase2():
    F3 = pd.read_csv(PROCESSED / "phase2_first3min_stats.csv", index_col=0)
    T = pd.read_csv(PROCESSED / "phase2_targets.csv").set_index("participant")
    sp = load_split(); fold = pd.Series(sp["fold_of_participant"])
    return F3, T, fold


def onehot(meta):
    oh = pd.DataFrame(index=meta.index)
    for c in COND: oh[f"c_{c}"] = (meta.cond == c).astype(float)
    oh["s_slow"] = (meta.speed == "slow").astype(float)
    return oh


def fold_pca_z(F3, fold, parts, n_comp=4):
    """Out-of-fold PCA scores (fit on training participants of each fold). Components are aligned to the all-sample PCA
    by sign/ordering via Procrustes-free greedy matching so folds can be pooled descriptively."""
    X = F3.loc[parts, STAT_COLS]
    ref = PCA(n_comp).fit((X - X.mean()) / (X.std() + 1e-9))
    Z = pd.DataFrame(index=parts, columns=[f"z{i}" for i in range(n_comp)], dtype=float)
    for f in sorted(fold.loc[parts].unique()):
        tr = [p for p in parts if fold[p] != f]; te = [p for p in parts if fold[p] == f]
        mu, sd = X.loc[tr].mean(), X.loc[tr].std() + 1e-9
        pca = PCA(n_comp).fit((X.loc[tr] - mu) / sd)
        # align components to reference (sign and order) using loading correlations
        C = pca.components_ @ ref.components_.T; order, signs, used = [], [], set()
        for i in range(n_comp):
            j = int(np.argmax([abs(C[i, k]) if k not in used else -1 for k in range(n_comp)])); used.add(j); order.append(j); signs.append(np.sign(C[i, j]) or 1.0)
        S = pca.transform((X.loc[te] - mu) / sd)
        for i, (j, s) in enumerate(zip(order, signs)): Z.loc[te, f"z{j}"] = S[:, i] * s
    return Z, ref


def oof_ridge(Xdf, y, fold, parts, extra=None):
    """Out-of-fold RidgeCV predictions; standardisation from training participants only; `extra` columns (one-hots)
    appended unscaled."""
    pred = pd.Series(index=parts, dtype=float)
    for f in sorted(fold.loc[parts].unique()):
        tr = [p for p in parts if fold[p] != f and not np.isnan(y[p])]; te = [p for p in parts if fold[p] == f]
        if Xdf is None or Xdf.shape[1] == 0:
            A = extra.loc[tr].to_numpy(); B = extra.loc[te].to_numpy()
        else:
            mu, sd = Xdf.loc[tr].mean(), Xdf.loc[tr].std() + 1e-9
            A, B = ((Xdf.loc[tr] - mu) / sd).to_numpy(), ((Xdf.loc[te] - mu) / sd).to_numpy()
            if extra is not None: A = np.hstack([A, extra.loc[tr].to_numpy()]); B = np.hstack([B, extra.loc[te].to_numpy()])
        m = RidgeCV(alphas=ALPHAS).fit(A, y[tr]); pred[te] = m.predict(B)
    return pred


def oof_pls(Xdf, y, fold, parts, extra, n_comp=4):
    pred = pd.Series(index=parts, dtype=float)
    for f in sorted(fold.loc[parts].unique()):
        tr = [p for p in parts if fold[p] != f and not np.isnan(y[p])]; te = [p for p in parts if fold[p] == f]
        mu, sd = Xdf.loc[tr].mean(), Xdf.loc[tr].std() + 1e-9
        A = np.hstack([((Xdf.loc[tr] - mu) / sd).to_numpy(), extra.loc[tr].to_numpy()]); B = np.hstack([((Xdf.loc[te] - mu) / sd).to_numpy(), extra.loc[te].to_numpy()])
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        m = PLSRegression(n_components=n_comp, scale=False).fit(A, (y[tr] - ym) / ys); pred[te] = m.predict(B).ravel() * ys + ym
    return pred


def corr(pred, y):
    m = pred.notna() & y.notna()
    return float(np.corrcoef(pred[m], y[m])[0, 1]) if m.sum() > 3 and pred[m].std() > 0 else np.nan


def paired_delta(pred_m, pred_b, y, groups, n_boot=2000, seed=0):
    """Δr (model − baseline) with participant bootstrap CI and two-sided p; mixed model on per-participant
    standardised squared-error difference (baseline − model) with group random intercept."""
    m = pred_m.notna() & pred_b.notna() & y.notna(); pm, pb, yy = pred_m[m].to_numpy(), pred_b[m].to_numpy(), y[m].to_numpy(); n = m.sum()
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if pm[i].std() == 0 or pb[i].std() == 0: continue
        d.append(np.corrcoef(pm[i], yy[i])[0, 1] - np.corrcoef(pb[i], yy[i])[0, 1])
    d = np.array(d); obs = np.corrcoef(pm, yy)[0, 1] - (np.corrcoef(pb, yy)[0, 1] if pb.std() > 0 else 0.0)
    p_boot = 2 * min((d <= 0).mean(), (d >= 0).mean())
    ys = yy.std() + 1e-9
    de = pd.DataFrame({"d_err": ((pb - yy) ** 2 - (pm - yy) ** 2) / ys ** 2, "group": groups[m].to_numpy()})
    try:
        mm = smf.mixedlm("d_err ~ 1", de, groups=de.group).fit(reml=True); mixed_p, mixed_d = float(mm.pvalues["Intercept"]), float(mm.params["Intercept"])
    except Exception:
        mixed_p, mixed_d = np.nan, float(de.d_err.mean())
    return dict(delta_r=obs, ci_lo=float(np.quantile(d, .025)), ci_hi=float(np.quantile(d, .975)), p_boot=float(p_boot), mixed_d_err=mixed_d, mixed_p=mixed_p, n=int(n))


def boot_r_ci(pred, y, n_boot=2000, seed=0):
    m = pred.notna() & y.notna(); p, yy = pred[m].to_numpy(), y[m].to_numpy(); n = m.sum()
    if n < 5: return np.nan, np.nan, np.nan, int(n)
    rng = np.random.default_rng(seed); rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if p[i].std() > 0: rs.append(np.corrcoef(p[i], yy[i])[0, 1])
    return float(np.corrcoef(p, yy)[0, 1]), float(np.quantile(rs, .025)), float(np.quantile(rs, .975)), int(n)


def _fit_mixed(f, d, groups):
    for meth in ("lbfgs", "powell", "nm", "bfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return smf.mixedlm(f, d, groups=groups).fit(reml=False, method=meth, maxiter=2000)
        except Exception:
            continue
    return None


def decompose(df, y, zcols, extra_fixed, group_col="group"):
    """Mixed models (ML) for y: cell -> + own z -> + others' mean z -> + group SD z. Returns marginal R² of each block,
    group ICC, and LRT p per block."""
    cell = "C(cond) * C(speed)"
    f0 = f"{y} ~ {cell}" + ("".join(f" + {e}" for e in extra_fixed))
    f1 = f0 + "".join(f" + {z}" for z in zcols)
    f2 = f1 + "".join(f" + om_{z}" for z in zcols)
    f3 = f2 + "".join(f" + gs_{z}" for z in zcols)
    d = df.dropna(subset=[y]).copy(); d[y] = (d[y] - d[y].mean()) / d[y].std()
    out, prev = [], None
    for name, f in (("cell+ctrl", f0), ("+own z", f1), ("+others mean z", f2), ("+group SD z", f3)):
        m = _fit_mixed(f, d, d[group_col])
        if m is None:   # group variance at the boundary and singular Hessian -> OLS (identical likelihood at var_group = 0)
            m = smf.ols(f, d).fit(); vg = 0.0; ve = float(m.scale); var_f = float(np.var(m.fittedvalues)); m_params = m.params
        else:
            var_f = float(np.var(np.asarray(m.model.exog) @ np.asarray(m.fe_params)))   # fixed-effects-only prediction
            vg, ve = float(m.cov_re.iloc[0, 0]), float(m.scale)
        r2m = var_f / (var_f + vg + ve); icc = vg / (vg + ve)
        nfe = lambda mm: len(getattr(mm, "fe_params", mm.params)); lrt = 2 * (m.llf - prev.llf) if prev is not None else np.nan; df_ = (nfe(m) - nfe(prev)) if prev is not None else np.nan
        out.append(dict(outcome=y, model=name, r2_marginal=r2m, icc_group=icc, var_group=vg, var_resid=ve, llf=m.llf, aic=m.aic, lrt=lrt, df=df_, p_lrt=float(stats.chi2.sf(lrt, df_)) if prev is not None else np.nan, n=int(m.nobs)))
        prev = m
    return pd.DataFrame(out)


def add_composition(df, zcols):
    d = df.copy()
    for z in zcols:
        s = d.groupby("group")[z].transform("sum"); n = d.groupby("group")[z].transform("size")
        d[f"om_{z}"] = (s - d[z]) / (n - 1); d[f"gs_{z}"] = d.groupby("group")[z].transform(lambda x: x.std(ddof=0))
    return d
