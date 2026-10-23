# Extract per-participant posterior-mean varying effects (v_ID) from the paper's Stan fits (Wu et al. 2025)
.f <- sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)); P <- if (length(.f)) normalizePath(file.path(dirname(.f[1]), "..")) else getwd()  # repository root
ids <- readLines(file.path(P, "data/processed/wu2025/stan_id_order.txt"))
loadRData <- function(f){ e <- new.env(); nm <- load(f, envir = e); get(nm[1], envir = e) }
out <- list()
for (tag in c("conditional_ARS", "solo_ARS")) {
  s <- loadRData(file.path(P, "external/wu2025/modeling/Stanfits", paste0("s_", tag, "_weights.rdata")))
  cat(tag, ":", class(s), names(s), "\n")
  v <- s$v_ID
  cat("v_ID dims:", dim(v), "\n")
  m <- apply(v, c(2, 3), mean); sdv <- apply(v, c(2, 3), sd)
  stopifnot(nrow(m) == length(ids))
  if (tag == "conditional_ARS") {
    nm <- c("Locality", "BlockVisibility", "RewardPrediction", "SuccessfulProximity", "UnsuccessfulProximity", "AdaptLocality", "AdaptSuccessful", "AdaptUnsuccessful")
  } else {
    nm <- c("Locality", "BlockVisibility", "RewardPrediction", "AdaptLocality")
  }
  stopifnot(ncol(m) == length(nm))
  df <- data.frame(id = ids, m); names(df)[-1] <- nm
  dfs <- data.frame(id = ids, sdv); names(dfs)[-1] <- paste0(nm, "_sd")
  write.csv(merge(df, dfs, by = "id"), file.path(P, "results/wu2025", paste0("stan_vID_", tag, ".csv")), row.names = FALSE)
  # population weights
  if (!is.null(s$weights)) { w <- apply(s$weights, c(2, 3), mean); print(w) }
  if (!is.null(s$sigma_ID)) print(apply(s$sigma_ID, 2, mean))
}
