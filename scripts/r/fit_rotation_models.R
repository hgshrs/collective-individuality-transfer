#!/usr/bin/env Rscript
# Re-estimate the paper's computational model of rotation magnitude (Chirkov et al. 2026, Fig. 4)
# using the paper's own formula/priors (code/statistical_analysis/rotation_computational_model/*.Rmd).
# Usage: Rscript fit_rotation_models.R --model tracking|search --half all|first|second
#        [--frac 1.0] [--chains 4] [--iter 5000] [--warmup 2000] [--threads 6] [--tag name] [--seed 4242]
suppressPackageStartupMessages({library(dplyr); library(readr); library(brms); library(posterior); library(cmdstanr)})
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) { i <- match(paste0("--", name), args); if (is.na(i)) default else args[i + 1] }
model   <- get_arg("model", "tracking")
half    <- get_arg("half", "all")
frac    <- as.numeric(get_arg("frac", "1.0"))
chains  <- as.integer(get_arg("chains", "4"))
iter    <- as.integer(get_arg("iter", "5000"))
warmup  <- as.integer(get_arg("warmup", "2000"))
threads <- as.integer(get_arg("threads", "6"))
seed    <- as.integer(get_arg("seed", "4242"))
tag     <- get_arg("tag", paste0(model, "_", half, if (frac < 1) paste0("_frac", frac) else ""))
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..", ".."))
out_dir <- file.path(root, "runs", "brms"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat("model:", model, "half:", half, "frac:", frac, "chains:", chains, "iter:", iter, "warmup:", warmup, "threads:", threads, "\n")

# ---- data (identical to index.Rmd + the model Rmd) ----
ts <- read_csv(file.path(root, "data/raw/osf/data/processed/time_series_data.csv"), show_col_types = FALSE,
               col_select = c(participant, signaling, resource, state, no_movement_times, step, out_degree,
                              soc_info_quality_raw, social_movement_direction, distance_change_lag1_cat2,
                              first_distance_from_resource, mean_degree_visible_players_first,
                              rotation_change_mean_raw, best_distance_from_resource_visible_players,
                              soc_info_quality_raw_best))
ts <- ts %>% mutate(
  Participant = as.factor(participant),
  SignalingType = factor(signaling, levels = c("A", "NP", "VP", "FP")),
  ResourceSpeed = relevel(as.factor(resource), ref = "fast"),
  State = state,
  DetectorChange = as.numeric(distance_change_lag1_cat2),
  SocInfoQualityRaw = round(as.numeric(soc_info_quality_raw), 2),
  SocInfoQualityRawBest = round(as.numeric(soc_info_quality_raw_best), 2),
  VisPlayersDistFromResource = round(as.numeric(best_distance_from_resource_visible_players), 2),
  OwnDistFromResource = round(as.numeric(first_distance_from_resource), 2),
  OutDegree = as.numeric(out_degree),
  NoMovementTimes = as.numeric(no_movement_times),
  SocialMovementDirection = round(as.numeric(social_movement_direction), 4),
  Rotation = round(as.numeric(rotation_change_mean_raw), 4),
  VisPlayersDirection = round(as.numeric(mean_degree_visible_players_first), 4),
  Time = as.numeric(step)) %>%
  mutate(SocInfoQualityBin = factor(case_when(
    (State == "tracking") & (SocInfoQualityRawBest < 0) ~ 1,
    (State == "tracking") & (SocInfoQualityRawBest > 0) ~ 0,
    (State == "searching") & (VisPlayersDistFromResource < 40) ~ 1,
    (State == "searching") & (VisPlayersDistFromResource > 40) ~ 0,
    .default = 0), levels = c(0, 1)))

if (half == "first") ts <- ts %>% filter(Time < 450)
if (half == "second") ts <- ts %>% filter(Time >= 450)
if (frac < 1) { set.seed(1); keep <- ts %>% distinct(Participant, SignalingType, ResourceSpeed) %>% group_by(SignalingType, ResourceSpeed) %>% slice_sample(prop = frac) %>% pull(Participant); ts <- ts %>% filter(Participant %in% keep) }

if (model == "tracking") {
  d <- ts %>% filter(State == "tracking", NoMovementTimes < 3) %>%
    mutate(SocQual = as.numeric(case_when(OutDegree == 0 ~ 0, SocInfoQualityRaw < -10 ~ 1, SocInfoQualityRaw < 0 ~ 0.5, SocInfoQualityRaw < 10 ~ -0.5, .default = -1))) %>%
    mutate(withSocInfo = factor(ifelse(OutDegree > 0, 1, 0), levels = c(0, 1))) %>%
    mutate(Rotation = Rotation / 150) %>%
    mutate(SocMovDir = ifelse(withSocInfo == 0, 0, (abs(SocialMovementDirection / max(abs(SocialMovementDirection), na.rm = TRUE)) - 0.5) * 2)) %>%
    mutate(RotationAbs = abs(Rotation) + 0.01) %>%
    mutate(Detector = -1 * DetectorChange / 6) %>%
    mutate(DistFromRes = (OwnDistFromResource - 20) / 20) %>%
    mutate(SocPosInFOV = ifelse(withSocInfo == 0, 0, (abs(VisPlayersDirection / max(abs(VisPlayersDirection), na.rm = TRUE)) - 0.5) * 2)) %>%
    select(Participant, Time, SignalingType, ResourceSpeed, RotationAbs, withSocInfo, Detector, DistFromRes, SocPosInFOV, SocMovDir, SocQual)
  form <- brmsformula(
    RotationAbs | trunc(ub = 1.02) ~ SignalingType * ResourceSpeed +
      SignalingType:ResourceSpeed:Detector + SignalingType:ResourceSpeed:DistFromRes +
      SignalingType:ResourceSpeed:withSocInfo:SocPosInFOV + SignalingType:ResourceSpeed:withSocInfo:SocMovDir +
      SignalingType:ResourceSpeed:withSocInfo:SocQual:SocPosInFOV + SignalingType:ResourceSpeed:withSocInfo:SocQual:SocMovDir +
      (1 + Detector + withSocInfo:SocPosInFOV + withSocInfo:SocMovDir + withSocInfo:SocQual:SocPosInFOV + withSocInfo:SocQual:SocMovDir | Participant),
    family = exponential())
} else {
  d <- ts %>% filter(State == "searching") %>%
    mutate(SocQual = as.numeric(case_when(OutDegree == 0 ~ 0, SocInfoQualityBin == 1 ~ 1, SocInfoQualityBin == 0 ~ -1, .default = -1))) %>%
    mutate(withSocInfo = factor(ifelse(OutDegree > 0, 1, 0), levels = c(0, 1))) %>%
    mutate(Rotation = Rotation / 150) %>%
    mutate(SocMovDir = ifelse(withSocInfo == 0, 0, (abs(SocialMovementDirection / max(abs(SocialMovementDirection), na.rm = TRUE)) - 0.5) * 2)) %>%
    mutate(RotationAbs = abs(Rotation) + 0.01) %>%
    mutate(DistFromRes = (OwnDistFromResource - 200) / 200) %>%
    mutate(SocPosInFOV = ifelse(withSocInfo == 0, 0, (abs(VisPlayersDirection / max(abs(VisPlayersDirection), na.rm = TRUE)) - 0.5) * 2)) %>%
    select(Participant, Time, SignalingType, ResourceSpeed, RotationAbs, withSocInfo, DistFromRes, SocPosInFOV, SocMovDir, SocQual)
  form <- brmsformula(
    RotationAbs | trunc(ub = 1.02) ~ SignalingType * ResourceSpeed +
      SignalingType:ResourceSpeed:withSocInfo:SocPosInFOV + SignalingType:ResourceSpeed:withSocInfo:SocMovDir +
      SignalingType:ResourceSpeed:withSocInfo:SocQual:SocPosInFOV + SignalingType:ResourceSpeed:withSocInfo:SocQual:SocMovDir +
      (1 + withSocInfo:SocPosInFOV + withSocInfo:SocMovDir + withSocInfo:SocQual:SocPosInFOV + withSocInfo:SocQual:SocMovDir | Participant),
    family = exponential())
}
cat("rows:", nrow(d), "participants:", n_distinct(d$Participant), "\n")

# ---- priors (paper): N(0,1) intercept, N(0,0.1) sd, N(0,0.5) b; constant(0) for structurally absent terms ----
pri <- prior(normal(0, 1), class = "Intercept") + prior(normal(0, 0.1), class = "sd") + prior(normal(0, 0.5), class = b)
gp <- get_prior(form, d)
zero_sd <- gp %>% filter(class == "sd", group == "Participant", grepl("withSocInfo0", coef))
for (cf in zero_sd$coef) pri <- pri + prior_string("constant(0)", class = "sd", coef = cf, group = "Participant")
bcoefs <- gp %>% filter(class == "b", coef != "") %>% pull(coef)
zero_b <- bcoefs[grepl("withSocInfo0", bcoefs)]
# Alone condition cannot have social terms: paper fixes A x social x quality terms and two A social terms to 0
zero_b <- c(zero_b, bcoefs[grepl("^SignalingTypeA:", bcoefs) & grepl("SocQual", bcoefs)])
if (model == "tracking") zero_b <- c(zero_b, "SignalingTypeA:ResourceSpeedfast:withSocInfo1:SocPosInFOV", "SignalingTypeA:ResourceSpeedslow:withSocInfo1:SocMovDir")
zero_b <- intersect(unique(zero_b), bcoefs)
for (cf in zero_b) pri <- pri + prior_string("constant(0)", class = "b", coef = cf)
cat("constant(0) sd terms:", length(zero_sd$coef), " constant(0) b terms:", length(zero_b), "\n")

t0 <- Sys.time()
fit <- brm(formula = form, data = d, prior = pri, chains = chains, cores = chains, seed = seed,
           warmup = warmup, iter = iter, backend = "cmdstanr", threads = threading(threads),
           file = file.path(out_dir, paste0(tag, ".rds")), file_refit = "on_change", save_pars = save_pars(all = FALSE), refresh = 200)
el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat("fit time (min):", el, "\n")

# ---- export ----
fe <- as.data.frame(fixef(fit, probs = c(0.05, 0.95))); fe$coef <- rownames(fe)
write_csv(fe, file.path(out_dir, paste0(tag, "_fixef.csv")))
re <- ranef(fit, probs = c(0.05, 0.95))$Participant  # array participant x stat x coef
re_df <- do.call(rbind, lapply(dimnames(re)[[3]], function(cf) data.frame(participant = dimnames(re)[[1]], coef = cf,
            estimate = re[, "Estimate", cf], sd = re[, "Est.Error", cf], q05 = re[, "Q5", cf], q95 = re[, "Q95", cf])))
write_csv(re_df, file.path(out_dir, paste0(tag, "_ranef.csv")))
# participant-level total (fixed + random) weights per feature, for interpretability
sm <- summary(fit)
diag <- data.frame(model = model, half = half, frac = frac, n_rows = nrow(d), n_participants = n_distinct(d$Participant),
                   chains = chains, iter = iter, warmup = warmup, fit_min = el,
                   max_rhat = max(rhat(fit), na.rm = TRUE), min_bulk_ess = min(sm$fixed$Bulk_ESS, na.rm = TRUE),
                   n_divergent = sum(nuts_params(fit, pars = "divergent__")$Value))
write_csv(diag, file.path(out_dir, paste0(tag, "_diag.csv")))
print(diag)
