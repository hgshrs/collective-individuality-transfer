.f <- sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)); P <- if (length(.f)) normalizePath(file.path(dirname(.f[1]), "..")) else getwd()  # repository root
loadRData <- function(f){ e <- new.env(); nm <- load(f, envir = e); get(nm[1], envir = e) }
library(jsonlite)
out <- list()
for (tag in c("conditional_ARS", "solo_ARS")) {
  s <- loadRData(file.path(P, "external/wu2025/modeling/Stanfits", paste0("s_", tag, "_weights.rdata")))
  w <- apply(s$weights, c(2, 3), mean)           # N_feat x 2 (random, smooth)
  bt <- if (length(dim(s$beta_t_ind)) == 3) apply(s$beta_t_ind, c(2, 3), mean) else matrix(apply(s$beta_t_ind, 2, mean), nrow = 1)
  sig <- apply(s$sigma_ID, 2, mean)
  sg <- apply(s$sigma_group, 2, mean)
  vg <- apply(s$v_group, c(2, 3), mean)
  out[[tag]] <- list(weights = w, beta_t_ind = bt, sigma_ID = sig, sigma_group = sg, v_group = vg,
                     dims = list(weights = dim(s$weights), beta_t_ind = dim(s$beta_t_ind), v_group = dim(s$v_group)))
}
write_json(out, file.path(P, "results/wu2025/stan_population.json"), digits = 8, pretty = TRUE)
cat("done\n")
