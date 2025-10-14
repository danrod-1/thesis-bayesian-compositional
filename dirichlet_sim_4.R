.libPaths(c("/lustre/home/drodionov/R/x86_64-pc-linux-gnu-library/4.4", "/lustre/software/R/4.4.2_10gcc_AuthenticAMD/lib64/R/library"))

args <- commandArgs(trailingOnly = TRUE)

start_sim_id <- as.integer(args[1])
end_sim_id   <- as.integer(args[2])
overwrite    <- as.logical(args[3])
n_obs        <- as.integer(args[4])
n_train      <- as.integer(args[5])
theta        <- as.numeric(args[6])
rho          <- as.numeric(args[7])
output_dir   <- args[8]


library(data.table)
library(MASS)
library(mvtnorm)
library(MCMCpack)
library(dplyr)
library(cmdstanr)
library(compositions)
library(posterior)
library(bayestestR)
library(zCompositions)

evaluate_model_betas <- function(fit, beta_matrix) {
  param_name <- "B"
  draws_df <- as_draws_df(fit$draws(param_name))
  
  summary_df <- describe_posterior(draws_df, ci = 0.95, centrality = c("mean", "median"))
  summary_df$selected <- summary_df$pd > 0.95 & summary_df$CI_low * summary_df$CI_high > 0
  
  true_param_names <- paste0(param_name, "[", row(beta_matrix), ",", col(beta_matrix), "]")
  true_param_values <- as.vector(beta_matrix)
  names(true_param_values) <- true_param_names
  
  active_names <- names(true_param_values[true_param_values != 0])
  active_values <- true_param_values[active_names]
  
  beta_draws_df <- as_draws_df(fit$draws("B"))
  active_draws_df <- beta_draws_df[, active_names, drop = FALSE]
  
  bias_mse_summary <- summarise_draws(active_draws_df, "mean", "sd", ~quantile(.x, probs = c(0.025, 0.975))) |>
    mutate(
      true_value = as.numeric(active_values),
      bias = mean - true_value,
      mse = (mean - true_value)^2,
      covered = true_value >= `2.5%` & true_value <= `97.5%`
    )
  
  bias_mean <- mean(bias_mse_summary$bias, na.rm = TRUE)
  mse_mean <- mean(bias_mse_summary$mse, na.rm = TRUE)
  coverage_percent <- mean(bias_mse_summary$covered, na.rm = TRUE) * 100
  
  selected_names <- summary_df$Parameter[summary_df$selected]
  true_nonzero_names <- names(true_param_values[true_param_values != 0])
  n_selected <- sum(summary_df$selected)
  n_true_nonzero <- length(true_nonzero_names)
  n_recovered <- sum(true_nonzero_names %in% selected_names)
  jaccard <- length(intersect(true_nonzero_names, selected_names)) / 
    length(union(true_nonzero_names, selected_names))
  n_false_positives <- length(setdiff(selected_names, true_nonzero_names))
  n_false_negatives <- length(setdiff(true_nonzero_names, selected_names))
  
  return(
    tibble::tibble(
      n_true_nonzero = n_true_nonzero,
      n_recovered = n_recovered,
      n_selected = n_selected,
      jaccard = jaccard,
      n_false_positives = n_false_positives,
      n_false_negatives = n_false_negatives,
      coverage_percent = coverage_percent,
      bias_mean = bias_mean,
      mse_mean = mse_mean
    )
  )
}


evaluate_klc <- function(Y_true, Y_pred) {
  stopifnot(all(dim(Y_true) == dim(Y_pred)))
  if (any(Y_true <= 0) || any(Y_pred <= 0)) {
    stop("All values in compositional matrices must be strictly positive.")
  }
  
  D <- ncol(Y_true)
  klc_vals <- numeric(nrow(Y_true))
  
  for (i in 1:nrow(Y_true)) {
    r1 <- Y_true[i, ] / Y_pred[i, ]
    r2 <- Y_pred[i, ] / Y_true[i, ]
    A1 <- mean(r1)
    A2 <- mean(r2)
    klc_vals[i] <- (D / 2) * log(A1 * A2)
  }
  
  return(mean(klc_vals))
}


extract_prediction_matrix <- function(fit, variable, C, N) {
  means <- posterior::summarise_draws(fit$draws(variable), "mean")$mean
  full_matrix <- matrix(means, ncol = C, byrow = FALSE)
  full_matrix[1:N, ]
}

evaluate_compositional_predictions <- function(Y_true, Y_pred) {
  stopifnot(nrow(Y_true) == nrow(Y_pred), ncol(Y_true) == ncol(Y_pred))
  if (any(Y_true <= 0, na.rm = TRUE) || any(Y_pred <= 0, na.rm = TRUE)) {
    stop("Compositional inputs must be strictly positive (no zeros or NAs).")
  }
  
  Y_true_acomp <- acomp(Y_true)
  Y_pred_acomp <- acomp(Y_pred)
  
  compute_totvar <- function(X) {
    V <- variation(acomp(X))
    D <- ncol(X)
    sum(V) / (2 * D)
  }
  
  totvar_true <- compute_totvar(Y_true)
  totvar_pred <- compute_totvar(Y_pred)
  R2_T <- totvar_pred / totvar_true
  
  dA_sq <- function(x, y) sum((clr(x) - clr(y))^2)
  center_true <- mean(Y_true_acomp)
  
  num <- sum(sapply(1:nrow(Y_true), function(i) dA_sq(Y_true[i, ], Y_pred[i, ])))
  den <- sum(sapply(1:nrow(Y_true), function(i) dA_sq(Y_true[i, ], center_true)))
  R2_A <- 1 - num / den
  
  KLC <- evaluate_klc(Y_true, Y_pred)
  
  return(data.frame(R2_T = R2_T, R2_A = R2_A, KLC = KLC))
}

compute_coverage_percent <- function(draws_df, Y_true) {
  Y_rep_only <- draws_df[, !grepl("^\\.", names(draws_df))]
  lower <- apply(Y_rep_only, 2, quantile, probs = 0.025, na.rm = TRUE)
  upper <- apply(Y_rep_only, 2, quantile, probs = 0.975, na.rm = TRUE)
  Y_true_flat <- as.vector(Y_true)
  stopifnot(length(Y_true_flat) == length(lower))
  coverage_mask <- !is.na(lower) & !is.na(upper)
  covered <- (Y_true_flat[coverage_mask] >= lower[coverage_mask]) &
    (Y_true_flat[coverage_mask] <= upper[coverage_mask])
  mean(covered) * 100
}

count_nas <- function(fit, var) {
  draws_df <- as_draws_df(fit$draws(var))
  sum(is.na(draws_df))
}


summarize_beta_results <- function(results_list) {
  summary_df <- lapply(names(results_list), function(model_name) {
    res <- results_list[[model_name]]
    
    n_true <- res$n_true_nonzero
    n_selected <- res$total_selected
    n_tp <- res$n_recovered
    n_fp <- length(res$false_positives)
    n_fn <- n_true - n_tp
    percent_guessed <- if (n_true == 0) NA else round(100 * n_tp / n_true, 1)
    
    data.frame(
      model = model_name,
      true_nonzero = n_true,
      selected = n_selected,
      true_positives = n_tp,
      false_positives = n_fp,
      false_negatives = n_fn,
      percent_guessed = percent_guessed
    )
  }) %>% bind_rows()
  
  return(summary_df)
}

replace_zeros_if_needed <- function(Y) {
  if (any(Y == 0)) {
    message("Replacing zeros using CZM method...")
    return(as.matrix(cmultRepl(Y, method = "CZM", label = 0)))
  } else {
    return(as.matrix(Y)) 
  }
}

generate_beta <- function(
    C = 3,
    P = 50,
    P_informative = 5,
    shared_covariates = c(2, 3),
    beta_min = 0.2,
    beta_max = 1.0,
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  
  C_eff <- C - 1
  beta <- matrix(0, nrow = C_eff, ncol = P)
  
  for (k in 1:C_eff) {
    informative_vars <- shared_covariates
    
    remaining_pool <- setdiff(2:P, informative_vars)
    n_unique <- P_informative - length(shared_covariates)
    unique_vars <- sample(remaining_pool, n_unique)
    
    informative_vars <- c(informative_vars, unique_vars)
    beta[k, informative_vars] <- runif(P_informative, beta_min, beta_max) * sample(c(-1, 1), P_informative, replace = TRUE)
  }
  
  rownames(beta) <- paste0("Component_", 1:C_eff)
  colnames(beta) <- paste0("X", 1:P)
  
  beta_full <- rbind(beta, rep(0, P)) 
  return(beta_full)
}

generate_dirichlet <- function(
    n_obs = 800,
    n_train = 600,
    C = 3,
    P_cont = 42,
    P_dummy = 7,
    beta,
    theta = 90,
    rho = 0.4,
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  
  intercept <- rep(1, n_obs)
  
  Sigma_X <- matrix(rho, nrow = P_cont, ncol = P_cont)
  diag(Sigma_X) <- 1
  X_cont_raw <- mvrnorm(n = n_obs, mu = rep(0, P_cont), Sigma = Sigma_X)
  X_cont <- scale(X_cont_raw, center = TRUE, scale = TRUE)
  
  n_cat <- 5
  n_bin <- 2
  min_per_cat <- 2
  max_per_cat <- 10
  
  stopifnot(P_dummy >= n_bin + n_cat * min_per_cat)
  
  cat_dummy_counts <- rep(min_per_cat, n_cat)
  remaining <- P_dummy - n_bin - sum(cat_dummy_counts)
  i <- 1
  while (remaining > 0) {
    if (cat_dummy_counts[i] < max_per_cat) {
      cat_dummy_counts[i] <- cat_dummy_counts[i] + 1
      remaining <- remaining - 1
    }
    i <- if (i == n_cat) 1 else i + 1
  }
  
  cat_levels <- cat_dummy_counts + 1
  
  cat_vars <- lapply(cat_levels, function(k) sample(1:k, n_obs, replace = TRUE))
  dummy_mats <- lapply(cat_vars, function(v) model.matrix(~ factor(v) - 1)[, -1])
  
  bin_vars <- replicate(n_bin, sample(0:1, n_obs, replace = TRUE), simplify = FALSE)
  bin_mats <- lapply(bin_vars, function(v) model.matrix(~ factor(v) - 1)[, -1])
  
  X_dummy <- do.call(cbind, c(dummy_mats, bin_mats))
  
  X <- cbind(intercept, X_cont, X_dummy)
  colnames(X) <- paste0("X", 1:ncol(X))
  
  eta <- X %*% t(beta)
  mu_dirichlet <- t(apply(eta, 1, function(row) {
    exp_row <- exp(row)
    exp_row / sum(exp_row)
  }))
  
  Y <- matrix(0, nrow = n_obs, ncol = C)
  for (i in 1:n_obs) {
    alpha_i <- mu_dirichlet[i, ] * theta
    Y[i, ] <- rdirichlet(1, alpha_i)
  }
  
  Y <- replace_zeros_if_needed(Y)
  
  idx_train <- sample(1:n_obs, n_train)
  idx_test <- setdiff(1:n_obs, idx_train)
  
  X_train <- X[idx_train, , drop = FALSE]
  Y_train <- Y[idx_train, , drop = FALSE]
  X_test  <- X[idx_test, , drop = FALSE]
  Y_test  <- Y[idx_test, , drop = FALSE]
  
  alr_transform <- function(Y, ref = ncol(Y)) {
    log(Y[, -ref] / Y[, ref])
  }
  
  stan_data_alr <- list(
    N = n_train,
    D = C - 1,
    P = ncol(X),
    X = X_train,
    Z = alr_transform(Y_train),
    N_new = nrow(X_test),
    X_new = X_test
  )
  
  stan_data_dirichlet <- list(
    N = n_train,
    D = C,
    P = ncol(X),
    X = X_train,
    Y = unname(split(Y_train, seq(nrow(Y_train)))),
    N_new = nrow(X_test),
    X_new = X_test
  )
  
  return(list(
    stan_data = stan_data_alr,
    stan_data_dirichlet = stan_data_dirichlet,
    Y_train = Y_train,
    Y_test = Y_test,
    params = list(
      beta = beta,
      theta = theta
    )
  ))
}


fit_normal <- function(
    data,
    stan_model,
    sim_id,
    output_dir,
    model_name = "normal",
    iter_sampling = 1000,
    iter_warmup = 500,
    chains = 4,
    parallel_chains = 4,
    adapt_delta = 0.9,
    max_treedepth = 12,
    seed = 1
) {
  stan_data   <- data$stan_data
  Y_train     <- data$Y_train
  Y_test      <- data$Y_test
  beta_true   <- data$params$beta[1:(nrow(data$params$beta) - 1), , drop = FALSE]
  
  D     <- stan_data$D
  N     <- stan_data$N
  N_new <- stan_data$N_new
  
  fit <- stan_model$sample(
    data = stan_data,
    seed = seed,
    iter_sampling = iter_sampling,
    iter_warmup = iter_warmup,
    chains = chains,
    parallel_chains = parallel_chains,
    refresh = 0,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth
  )
  
  mu_train <- extract_prediction_matrix(fit, "Y_hat", D + 1, N)
  mu_test  <- extract_prediction_matrix(fit, "Y_hat_pred", D + 1, N_new)
  
  pred_train <- evaluate_compositional_predictions(Y_train, mu_train) |>
    dplyr::mutate(set = "train", model = model_name, sim = sim_id)
  pred_test <- evaluate_compositional_predictions(Y_test, mu_test) |>
    dplyr::mutate(set = "test", model = model_name, sim = sim_id)
  pred_df <- dplyr::bind_rows(pred_train, pred_test)
  
  beta_selection_eval <- evaluate_model_betas(fit, beta_true) |>
    dplyr::mutate(model = model_name, sim = sim_id)
  
  readr::write_csv(pred_df,
                   file.path(output_dir, "prediction_results.csv"),
                   append = file.exists(file.path(output_dir, "prediction_results.csv")))
  
  readr::write_csv(beta_selection_eval,
                   file.path(output_dir, "beta_results.csv"),
                   append = file.exists(file.path(output_dir, "beta_results.csv")))
  
  rm(fit, mu_train, mu_test,
     pred_train, pred_test, pred_df,
     beta_selection_eval)
  gc()
}

fit_student <- function(
    data,
    stan_model,
    sim_id,
    output_dir,
    model_name = "student",
    iter_sampling = 1000,
    iter_warmup = 500,
    chains = 4,
    parallel_chains = 4,
    adapt_delta = 0.9,
    max_treedepth = 12,
    seed = 1
) {
  stan_data   <- data$stan_data
  Y_train     <- data$Y_train
  Y_test      <- data$Y_test
  beta_true   <- data$params$beta[1:(nrow(data$params$beta) - 1), , drop = FALSE]
  
  D     <- stan_data$D
  N     <- stan_data$N
  N_new <- stan_data$N_new
  
  fit <- stan_model$sample(
    data = stan_data,
    seed = seed,
    iter_sampling = iter_sampling,
    iter_warmup = iter_warmup,
    chains = chains,
    parallel_chains = parallel_chains,
    refresh = 0,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth
  )
  
  mu_train <- extract_prediction_matrix(fit, "Y_hat", D + 1, N)
  mu_test  <- extract_prediction_matrix(fit, "Y_hat_pred", D + 1, N_new)
  
  pred_train <- evaluate_compositional_predictions(Y_train, mu_train) |>
    dplyr::mutate(set = "train", model = model_name, sim = sim_id)
  pred_test <- evaluate_compositional_predictions(Y_test, mu_test) |>
    dplyr::mutate(set = "test", model = model_name, sim = sim_id)
  pred_df <- dplyr::bind_rows(pred_train, pred_test)
  
  beta_selection_eval <- evaluate_model_betas(fit, beta_true) |>
    dplyr::mutate(model = model_name, sim = sim_id)
  
  readr::write_csv(pred_df,
                   file.path(output_dir, "prediction_results.csv"),
                   append = file.exists(file.path(output_dir, "prediction_results.csv")))
  
  readr::write_csv(beta_selection_eval,
                   file.path(output_dir, "beta_results.csv"),
                   append = file.exists(file.path(output_dir, "beta_results.csv")))
  
  rm(fit, mu_train, mu_test,
     pred_train, pred_test, pred_df,
     beta_selection_eval)
  gc()
}

fit_dirichlet <- function(
    data,
    stan_model,
    sim_id,
    output_dir,
    model_name = "dirichlet",
    iter_sampling = 1000,
    iter_warmup = 500,
    chains = 4,
    parallel_chains = 4,
    adapt_delta = 0.9,
    max_treedepth = 12,
    seed = 1
) {
  stan_data   <- data$stan_data_dirichlet
  Y_train     <- data$Y_train
  Y_test      <- data$Y_test
  beta_true   <- data$params$beta
  theta_true  <- data$params$theta
  
  D     <- stan_data$D
  N     <- stan_data$N
  N_new <- stan_data$N_new
  
  fit <- stan_model$sample(
    data = stan_data,
    seed = seed,
    iter_sampling = iter_sampling,
    iter_warmup = iter_warmup,
    chains = chains,
    parallel_chains = parallel_chains,
    refresh = 0,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth
  )
  
  mu_train <- extract_prediction_matrix(fit, "mu", D, N)
  mu_test  <- extract_prediction_matrix(fit, "mu_pred", D, N_new)
  
  pred_train <- evaluate_compositional_predictions(Y_train, mu_train) |>
    dplyr::mutate(set = "train", model = model_name, sim = sim_id)
  pred_test <- evaluate_compositional_predictions(Y_test, mu_test) |>
    dplyr::mutate(set = "test", model = model_name, sim = sim_id)
  pred_df <- dplyr::bind_rows(pred_train, pred_test)
  
  beta_selection_eval <- evaluate_model_betas(fit, beta_true) |>
    dplyr::mutate(model = model_name, sim = sim_id)
  
  # === Theta Summary ===
  theta_summary <- summarise_draws(
    as_draws_df(fit$draws("theta")),
    "mean", "median", "sd", ~quantile(.x, probs = c(0.025, 0.975))
  ) |>
    dplyr::mutate(model = model_name, sim = sim_id, true_value = theta_true)
  
  # === Write results ===
  readr::write_csv(pred_df,
                   file.path(output_dir, "prediction_results.csv"),
                   append = file.exists(file.path(output_dir, "prediction_results.csv")))
  
  readr::write_csv(beta_selection_eval,
                   file.path(output_dir, "beta_results.csv"),
                   append = file.exists(file.path(output_dir, "beta_results.csv")))
  
  readr::write_csv(theta_summary,
                   file.path(output_dir, "theta_results.csv"),
                   append = file.exists(file.path(output_dir, "theta_results.csv")))
  
  rm(fit, mu_train, mu_test,
     pred_train, pred_test, pred_df,
     beta_selection_eval, theta_summary)
  gc()
}

run_simulations_dirichlet <- function(
    start_sim_id,
    end_sim_id,
    C = 3,
    P = 50,
    P_informative = 5,
    shared_covariates = c(2, 3),
    beta_min = 0.2,
    beta_max = 1.0,
    n_obs = 800,
    n_train = 600,
    theta = 90,
    rho = 0.4,
    seed = 1,
    beta_seed = 10,
    output_dir = "results_dirichlet/",
    iter_sampling = 1000,
    iter_warmup = 500,
    chains = 4,
    parallel_chains = 4,
    adapt_delta = 0.9,
    max_treedepth = 12,
    P_dummy = 7,
    verbose = TRUE,
    overwrite = FALSE
) {
  if (overwrite) {
    if (dir.exists(output_dir)) {
      unlink(output_dir, recursive = TRUE)
    }
  }
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  alr_student_model <- cmdstan_model("ALR_student_regularized_horseshoe.stan")
  alr_normal_model  <- cmdstan_model("ALR_normal_regularized_horseshoe.stan")
  dirichlet_model   <- cmdstan_model("dirichlet_horseshoe.stan")
  
  beta <- generate_beta(
    C = C, P = P, P_informative = P_informative,
    shared_covariates = shared_covariates,
    beta_min = beta_min, beta_max = beta_max,
    seed = beta_seed
  )
  readr::write_csv(as.data.frame(beta), file.path(output_dir, "beta_true.csv"))
  
  for (sim_id in start_sim_id:end_sim_id) {
    sim_seed <- seed + sim_id
    
    if (verbose) cat("Running simulation", sim_id, "\n")
    
    data <- generate_dirichlet(
      n_obs = n_obs, n_train = n_train, C = C,
      P_cont = P - P_dummy - 1, P_dummy = P_dummy,
      beta = beta, theta = theta, rho = rho,
      seed = sim_seed
    )
    
    fit_student(data, alr_student_model, sim_id = sim_id, output_dir = output_dir,
                iter_sampling = iter_sampling, iter_warmup = iter_warmup,
                chains = chains, parallel_chains = parallel_chains,
                adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                seed = sim_seed)
    
    fit_normal(data, alr_normal_model, sim_id = sim_id, output_dir = output_dir,
               iter_sampling = iter_sampling, iter_warmup = iter_warmup,
               chains = chains, parallel_chains = parallel_chains,
               adapt_delta = adapt_delta, max_treedepth = max_treedepth,
               seed = sim_seed)
    
    fit_dirichlet(data, dirichlet_model, sim_id = sim_id, output_dir = output_dir,
                  iter_sampling = iter_sampling, iter_warmup = iter_warmup,
                  chains = chains, parallel_chains = parallel_chains,
                  adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                  seed = sim_seed)
  }
  
  invisible(NULL)
}


run_simulations_dirichlet(
  start_sim_id = start_sim_id,
  end_sim_id   = end_sim_id,
  C = 4,
  P = 60,
  P_informative = 6,
  shared_covariates = c(1,2,48),
  beta_min = 0.2,
  beta_max = 1.0,
  n_obs = n_obs,
  n_train = n_train,
  theta = theta,
  rho = rho,
  seed = 1 + start_sim_id,
  beta_seed = 10,
  output_dir = output_dir,
  iter_sampling = 1000,
  iter_warmup = 500,
  chains = 4,
  parallel_chains = 4,
  adapt_delta = 0.9,
  max_treedepth = 15,
  P_dummy = 20,
  verbose = TRUE,
  overwrite = overwrite
)


