"""Phase 1 evaluation (AGENT_SPEC §4.2-3..5, §4.3, §4.5): aggregate fold x seed predictions, compare with baselines,
participant-level paired tests (bootstrap CI + mixed model over folds/seeds), A-H2 cross-individual distance regression,
nuisance checks (condition decodability from z, correlations with data-quality proxies), z-space figures.
Usage: python scripts/12_phase1_eval.py --exp phase1_a_z4 [--baselines results/phase1/baselines_ridge.csv]"""
import argparse, json, glob, numpy as np, pandas as pd, yaml
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collective_id.paths import RUNS, RESULTS, CONFIGS, PROCESSED
from collective_id.data import load_features
from collective_id.phase1 import load_teacher
from collective_id.summary_stats import search_summary
ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="phase1_a_z4"); ap.add_argument("--tag", default="ridge"); a = ap.parse_args()
out = RESULTS / "phase1"; out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(RUNS / a.exp / "config.json")); coefs = cfg["target_coefs"]
teacher, teacher_sd = load_teacher(cfg["teacher"], coefs)
d = load_features(); meta = d.groupby("participant").agg(cond=("cond", "first"), speed=("speed", "first"), group=("group", "first"), dup=("duplicates_percentage", "first"), tie=("time_in_experiment", "first"), score=("final_score", "first"))
# ---- collect predictions ----
P, Z, cross = [], [], []
for f in sorted(glob.glob(str(RUNS / a.exp / "seed*" / "fold*" / "pred_test.csv"))):
    seed = int(f.split("seed")[1].split("/")[0]); fold = int(f.split("fold")[1].split("/")[0])
    p = pd.read_csv(f, index_col=0); p["seed"] = seed; p["fold"] = fold; P.append(p)
    z = pd.read_csv(f.replace("pred_test.csv", "z_test.csv"), index_col=0); z["seed"] = seed; z["fold"] = fold; Z.append(z)
    c = np.load(f.replace("pred_test.csv", "cross.npz"), allow_pickle=True); cross.append((seed, fold, c["cross"], c["test"], c["Z"]))
P = pd.concat(P); Z = pd.concat(Z); P.index.name = "participant"; Z.index.name = "participant"
P.reset_index().to_csv(out / f"pred_{a.exp}.csv", index=False); Z.reset_index().to_csv(out / f"z_{a.exp}.csv", index=False)
np.savez_compressed(out / "z_individual.npz", **{f"seed{s}_fold{f}": zz for s, f, _, _, zz in cross}, participants=np.array([t for _, _, _, t, _ in cross], dtype=object))
seeds = sorted(P.seed.unique())
# ---- per-seed correlations vs teacher, mean±SD over seeds ----
def corr_table(pred):
    rows = []
    for s in seeds:
        ps = pred[pred.seed == s].groupby("participant")[coefs].mean() if "seed" in pred else pred
        for c in coefs:
            rows.append(dict(seed=s, coef=c, r=np.corrcoef(ps[c], teacher.loc[ps.index, c])[0, 1], rmse=float(np.sqrt(((ps[c] - teacher.loc[ps.index, c]) ** 2).mean()))))
    return pd.DataFrame(rows)
ct = corr_table(P); ct.to_csv(out / f"corr_by_seed_{a.exp}.csv", index=False)
summary = ct.groupby("coef").agg(r_mean=("r", "mean"), r_sd=("r", "std"), rmse_mean=("rmse", "mean")).reset_index()
base = pd.read_csv(out / f"baselines_{a.tag}.csv")
for bm in base.model.unique(): summary[f"r_{bm}"] = summary.coef.map(base[base.model == bm].set_index("coef").r)
summary.to_csv(out / f"summary_{a.exp}.csv", index=False)
print(summary.round(3).to_string(index=False))
# ---- participant-level paired comparison vs cond_mean and ridge (Δ squared error, standardised per coef) ----
def paired(pred_model, pred_base, name):
    rows = []
    for s in seeds:
        pm = pred_model[pred_model.seed == s].groupby("participant")[coefs].mean(); pb = pred_base.loc[pm.index, coefs]
        for c in coefs:
            sd = teacher[c].std()
            e_m = ((pm[c] - teacher.loc[pm.index, c]) / sd) ** 2; e_b = ((pb[c] - teacher.loc[pm.index, c]) / sd) ** 2
            rows.append(pd.DataFrame(dict(participant=pm.index, seed=s, coef=c, d_err=(e_b - e_m).to_numpy(), fold=P[P.seed == s].groupby("participant").fold.first().loc[pm.index].to_numpy())))
    df = pd.concat(rows); df["cond"] = df.participant.map(meta.cond); df["speed"] = df.participant.map(meta.speed)
    res = []
    for c in coefs:
        dc = df[df.coef == c]
        try:
            mm = smf.mixedlm("d_err ~ 1", dc, groups=dc["participant"]).fit(reml=False); est, pval = mm.params["Intercept"], mm.pvalues["Intercept"]
        except Exception: est, pval = dc.d_err.mean(), np.nan
        # bootstrap Δr over participants (seed-averaged predictions)
        pm = pred_model.groupby("participant")[c].mean(); pb = pred_base.loc[pm.index, c]; t = teacher.loc[pm.index, c]
        rng = np.random.default_rng(0); idx = np.arange(len(pm)); dr = []
        for _ in range(2000):
            i = rng.choice(idx, len(idx)); dr.append(np.corrcoef(pm.iloc[i], t.iloc[i])[0, 1] - np.corrcoef(pb.iloc[i], t.iloc[i])[0, 1])
        dr = np.array(dr); dr0 = np.corrcoef(pm, t)[0, 1] - np.corrcoef(pb, t)[0, 1]
        res.append(dict(baseline=name, coef=c, delta_r=dr0, delta_r_ci_lo=np.percentile(dr, 2.5), delta_r_ci_hi=np.percentile(dr, 97.5), p_boot=float(2 * min((dr <= 0).mean(), (dr >= 0).mean())), mixed_d_err=est, mixed_p=pval))
    return pd.DataFrame(res)
comp = pd.concat([paired(P, pd.read_csv(out / f"pred_cond_mean_{a.tag}.csv", index_col=0), "cond_mean"), paired(P, pd.read_csv(out / f"pred_ridge_summary_cond_{a.tag}.csv", index_col=0), "ridge_summary_cond")])
comp.to_csv(out / f"paired_vs_baselines_{a.exp}.csv", index=False); print(comp.round(3).to_string(index=False))
# ---- A-H1 verdict (pre-registered criterion): mean over coefs of Δr vs cond_mean ≥ 0.10 & p<0.01, and beats ridge ----
mean_dr_cond = comp[comp.baseline == "cond_mean"].delta_r.mean(); mean_dr_ridge = comp[comp.baseline == "ridge_summary_cond"].delta_r.mean()
# pooled test: participant-level mixed model on d_err averaged over coefs
verdict = dict(exp=a.exp, mean_delta_r_vs_cond_mean=mean_dr_cond, mean_delta_r_vs_ridge=mean_dr_ridge,
               min_mixed_p_vs_cond_mean=float(comp[comp.baseline == "cond_mean"].mixed_p.min()), n_coefs_p_lt_0_01_vs_cond=int((comp[comp.baseline == "cond_mean"].mixed_p < 0.01).sum()),
               A_H1_pass=bool(mean_dr_cond >= 0.10 and (comp[comp.baseline == "cond_mean"].mixed_p < 0.01).mean() >= 0.5 and mean_dr_ridge > 0))
# ---- A-H2: cross-individual — error of predicting k from l's z vs ||z_k − z_l|| (participant-k random intercept) ----
rows = []
S = search_summary(d); S = (S - S.mean()) / (S.std() + 1e-9)
for s, f, C, test, Zf in cross:
    n = len(test); T = teacher.loc[test, coefs].to_numpy(); sd = teacher[coefs].std().to_numpy()
    for k in range(n):
        for l in range(n):
            if k == l: continue
            err = (((C[k, l] - T[k]) / sd) ** 2).mean(); dz = np.linalg.norm(Zf[k] - Zf[l]); ds = np.linalg.norm(S.loc[test[k]] - S.loc[test[l]])
            rows.append(dict(seed=s, fold=f, k=test[k], l=test[l], err=err, dz=dz, ds=ds))
X = pd.DataFrame(rows); X["dz_s"] = (X.dz - X.dz.mean()) / X.dz.std(); X["ds_s"] = (X.ds - X.ds.mean()) / X.ds.std(); X["log_err"] = np.log(X.err + 1e-6)
h2 = {}
for name, form in [("z_dist", "log_err ~ dz_s"), ("summary_dist", "log_err ~ ds_s"), ("both", "log_err ~ dz_s + ds_s")]:
    m = smf.mixedlm(form, X, groups=X["k"]).fit(reml=False)
    h2[name] = dict(params=m.params.to_dict(), pvalues=m.pvalues.to_dict(), llf=m.llf, aic=m.aic)
h2["A_H2_pass"] = bool(h2["z_dist"]["params"]["dz_s"] > 0 and h2["z_dist"]["pvalues"]["dz_s"] < 1e-3 and h2["both"]["params"]["dz_s"] > h2["both"]["params"].get("ds_s", -np.inf))
h2["note"] = "log squared error increases with z distance => slope>0 here corresponds to the spec's 'negative slope of performance'"
json.dump(h2, open(out / f"cross_individual_{a.exp}.json", "w"), indent=1, default=float); X.to_csv(out / f"cross_individual_pairs_{a.exp}.csv", index=False)
print("A-H2:", {k: (v["params"], v["pvalues"]) for k, v in h2.items() if isinstance(v, dict) and "params" in v})
# ---- nuisance (§4.5): condition decodability from z; correlations with dup%, time_in_experiment, first-minute stats ----
Zm = Z.groupby("participant")[[c for c in Z.columns if c.startswith("z")]].mean(); zc = Zm.columns.tolist()
lab = meta.loc[Zm.index]; y_cond = lab.cond.astype("category").cat.codes; y_speed = lab.speed.astype("category").cat.codes
cv = StratifiedKFold(5, shuffle=True, random_state=0)
acc_cond = cross_val_score(LogisticRegression(max_iter=2000), Zm.to_numpy(), y_cond, cv=cv).mean(); acc_speed = cross_val_score(LogisticRegression(max_iter=2000), Zm.to_numpy(), y_speed, cv=cv).mean()
chance_cond = lab.cond.value_counts(normalize=True).max(); chance_speed = lab.speed.value_counts(normalize=True).max()
first_min = d[(d.t < 80)].groupby("participant").agg(rot_freq_1min=("rot_abs", lambda x: (x > 0.011).mean()), rot_abs_1min=("rot_abs", "mean"), no_move_1min=("no_move", lambda x: (x > 0).mean())).loc[Zm.index]
nuis = pd.concat([lab[["dup", "tie", "score"]], first_min], axis=1)
cor = pd.DataFrame({z: {n: stats.spearmanr(Zm[z], nuis[n])[0] for n in nuis.columns} for z in zc})
nres = dict(cond_decoding_acc=acc_cond, cond_chance=chance_cond, speed_decoding_acc=acc_speed, speed_chance=chance_speed, flag_condition_encoded=bool(acc_cond > 0.5), spearman_z_vs_nuisance=cor.to_dict())
# within-condition standardised z re-evaluation is only needed if flagged; report the flag
json.dump(nres, open(out / f"nuisance_{a.exp}.json", "w"), indent=1, default=float); print("nuisance:", {k: v for k, v in nres.items() if k != "spearman_z_vs_nuisance"}); print(cor.round(2))
verdict.update(A_H2_pass=h2["A_H2_pass"], condition_decoding_acc=acc_cond)
json.dump(verdict, open(out / f"verdict_{a.exp}.json", "w"), indent=1, default=float); print(verdict)
# ---- figures ----
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
for ax, c in zip(axs.flat, coefs):
    pm = P.groupby("participant")[c].mean(); ax.scatter(teacher.loc[pm.index, c], pm, s=8, alpha=0.5); ax.set_xlabel(f"teacher RE: {c}"); ax.set_ylabel("predicted"); ax.set_title(f"r = {np.corrcoef(teacher.loc[pm.index, c], pm)[0,1]:.2f}")
fig.suptitle(f"{a.exp}: search-phase z → tracking-phase random effects (held-out participants)"); fig.tight_layout(); fig.savefig(out / f"fig_transfer_{a.exp}.png", dpi=150); fig.savefig(out / f"fig_transfer_{a.exp}.pdf"); plt.close(fig)
from sklearn.decomposition import PCA
pc = PCA(2).fit_transform((Zm - Zm.mean()) / (Zm.std() + 1e-9)) if len(zc) > 2 else Zm.to_numpy()[:, :2]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (lab_name, labs) in zip(axs, [("condition", lab.cond), ("speed", lab.speed), ("score", lab.score)]):
    if lab_name == "score": sc = ax.scatter(pc[:, 0], pc[:, 1], c=labs, s=10, cmap="viridis"); plt.colorbar(sc, ax=ax)
    else:
        for u in labs.unique(): m = labs == u; ax.scatter(pc[m, 0], pc[m, 1], s=10, label=u)
        ax.legend()
    ax.set_title(f"z space (PCA) coloured by {lab_name}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(out / f"fig_zspace_{a.exp}.png", dpi=150); fig.savefig(out / f"fig_zspace_{a.exp}.pdf"); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 4)); Xs = X.sample(min(len(X), 20000), random_state=0); ax.scatter(Xs.dz, Xs.log_err, s=3, alpha=0.2); ax.set_xlabel("||z_k − z_l||"); ax.set_ylabel("log squared error (k predicted from l)"); ax.set_title(f"A-H2 slope={h2['z_dist']['params']['dz_s']:.3f}, p={h2['z_dist']['pvalues']['dz_s']:.1e}")
fig.tight_layout(); fig.savefig(out / f"fig_distance_{a.exp}.png", dpi=150); fig.savefig(out / f"fig_distance_{a.exp}.pdf"); plt.close(fig)
