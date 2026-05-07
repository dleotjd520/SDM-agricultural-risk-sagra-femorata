################################################################################
# 2. (SDM) Evaluation, variable importance and PDP
#
# This script is designed to run AFTER:
#   1.Data preprocessing and species distribution modeling.R
#
# It reads the public-release outputs produced by Script 1:
#   outputs/01_sdm/03_sdm_training_data.rds
#   outputs/01_sdm/04_sdm_output_RF.rds
#   outputs/01_sdm/04_sdm_output_MaxEnt.rds
#   outputs/01_sdm/01_occurrence_environment.rds
#
# Main steps:
#   1) Load RF and MaxEnt model outputs from Script 1
#   2) Calculate repeated ensemble performance
#   3) Calculate final ensemble threshold using pooled out-of-fold test predictions
#   4) Calculate threshold-based ensemble performance
#   5) Calculate ensemble variable importance using DALEX
#   6) Calculate ensemble PDPs using a common grid
#   7) Save summary tables and optional figures
################################################################################

rm(list = ls())

################################################################################
# 0. CONFIGURATION
################################################################################

cfg <- list(
  # Project directory
  project_dir = getwd(),

  # Input directory: must match cfg$output_dir in Script 1
  sdm_input_dir = "outputs/01_sdm",

  # Output directory for this script
  output_dir = "outputs/02_sdm_evaluation_varimp_pdp",

  # Expected files produced by Script 1
  training_data_file = "03_sdm_training_data.rds",
  occurrence_environment_file = "01_occurrence_environment.rds",
  rf_output_file = "04_sdm_output_RF.rds",
  maxent_output_file = "04_sdm_output_MaxEnt.rds",

  # Ensemble and repeated evaluation
  n_repeats = 10,
  seed = 910520,

  # DALEX settings
  pdp_grid_n = 1000,
  pdp_sample_n = 1000,
  variable_importance_loss = "one_minus_auc",

  # Figure options
  save_figures = TRUE,
  figure_width = 5,
  figure_height = 6,
  figure_res = 500
)

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# 1. PACKAGES
################################################################################

required_packages <- c(
  "maxnet", "ranger", "dismo", "MLmetrics", "DALEX"
)

missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Please install the following packages before running this script: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(maxnet)
  library(ranger)
  library(dismo)
  library(MLmetrics)
  library(DALEX)
})

################################################################################
# 2. HELPER FUNCTIONS
################################################################################

read_required_rds <- function(path) {
  if (!file.exists(path)) {
    stop("Required input file not found: ", path, call. = FALSE)
  }
  readRDS(path)
}

get_ranger_predictions <- function(model, newdata) {
  pred <- predict(model, newdata)
  if ("predictions" %in% names(pred)) return(as.numeric(pred$predictions))
  if ("prediction" %in% names(pred)) return(as.numeric(pred$prediction))
  stop("Could not find ranger prediction values in predict() output.", call. = FALSE)
}

predict_rf <- function(model, newdata, vars) {
  get_ranger_predictions(model, newdata[, vars, drop = FALSE])
}

predict_maxent <- function(model, newdata, vars) {
  as.numeric(predict(model, newdata[, vars, drop = FALSE], type = "cloglog"))
}

predict_ensemble <- function(rf_model, maxent_model, newdata, vars) {
  prf <- predict_rf(rf_model, newdata, vars)
  pme <- predict_maxent(maxent_model, newdata, vars)
  (prf + pme) / 2
}

compute_mcc <- function(y_true, y_pred_class) {
  tp <- sum(y_pred_class == 1 & y_true == 1)
  tn <- sum(y_pred_class == 0 & y_true == 0)
  fp <- sum(y_pred_class == 1 & y_true == 0)
  fn <- sum(y_pred_class == 0 & y_true == 1)

  tp <- as.numeric(tp)
  tn <- as.numeric(tn)
  fp <- as.numeric(fp)
  fn <- as.numeric(fn)

  denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (is.na(denom) || denom == 0) return(NA_real_)
  ((tp * tn) - (fp * fn)) / denom
}

calc_metrics <- function(y_true, y_pred_raw, y_pred_class) {
  sens <- MLmetrics::Sensitivity(y_true = y_true, y_pred = y_pred_class, positive = 1)
  spec <- MLmetrics::Specificity(y_true = y_true, y_pred = y_pred_class, positive = 1)

  data.frame(
    ACC = MLmetrics::Accuracy(y_pred = y_pred_class, y_true = y_true),
    AUC = MLmetrics::AUC(y_pred = y_pred_raw, y_true = y_true),
    PRAUC = MLmetrics::PRAUC(y_pred = y_pred_raw, y_true = y_true),
    TSS = sens + spec - 1,
    F1.1 = MLmetrics::F1_Score(y_true = y_true, y_pred = y_pred_class, positive = 1),
    Sens.1 = sens,
    Spec.1 = spec,
    Prec.1 = MLmetrics::Precision(y_true = y_true, y_pred = y_pred_class, positive = 1),
    Recall.1 = MLmetrics::Recall(y_true = y_true, y_pred = y_pred_class, positive = 1),
    MCC = compute_mcc(y_true, y_pred_class)
  )
}

calc_spec_sens_threshold <- function(y, p) {
  ev <- dismo::evaluate(p = p[y == 1], a = p[y == 0])
  as.numeric(dismo::threshold(ev, "spec_sens"))
}

make_variable_splits_equal <- function(x, vars, n = 1000) {
  out <- lapply(vars, function(v) {
    rng <- range(x[[v]], na.rm = TRUE)
    seq(from = rng[1], to = rng[2], length.out = n)
  })
  names(out) <- vars
  out
}

make_ensemble_predict_function <- function(vars) {
  force(vars)
  function(model, newdata) {
    newdata <- newdata[, vars, drop = FALSE]
    prf <- get_ranger_predictions(model$rf, newdata)
    pme <- as.numeric(predict(model$maxent, newdata, type = "cloglog"))
    (prf + pme) / 2
  }
}

summarize_pdp <- function(pdp_all, n_repeats) {
  pdp_mean <- aggregate(value.y ~ variable + value.x, data = pdp_all, FUN = mean)
  pdp_sd <- aggregate(value.y ~ variable + value.x, data = pdp_all, FUN = sd)

  names(pdp_mean) <- c("variable", "value.x", "y.mean")
  names(pdp_sd) <- c("variable", "value.x", "y.sd")

  pdp_summary <- merge(pdp_mean, pdp_sd, by = c("variable", "value.x"), all = TRUE)
  pdp_summary <- pdp_summary[order(pdp_summary$variable, pdp_summary$value.x), ]

  tcrit <- stats::qt(0.975, df = n_repeats - 1)
  pdp_summary$se <- pdp_summary$y.sd / sqrt(n_repeats)
  pdp_summary$ci_low <- pmax(0, pdp_summary$y.mean - tcrit * pdp_summary$se)
  pdp_summary$ci_high <- pmin(1, pdp_summary$y.mean + tcrit * pdp_summary$se)

  pdp_summary
}

summarize_variable_importance <- function(vi_all) {
  vi_mean <- aggregate(dropout_loss ~ variable, data = vi_all, FUN = mean)
  vi_sd <- aggregate(dropout_loss ~ variable, data = vi_all, FUN = sd)
  vi_n <- aggregate(dropout_loss ~ variable, data = vi_all, FUN = function(x) sum(!is.na(x)))

  names(vi_mean)[2] <- "importance_mean"
  names(vi_sd)[2] <- "importance_sd"
  names(vi_n)[2] <- "n_repeats"

  vi_summary <- merge(vi_mean, vi_sd, by = "variable", all = TRUE)
  vi_summary <- merge(vi_summary, vi_n, by = "variable", all = TRUE)
  vi_summary$importance_se <- vi_summary$importance_sd / sqrt(vi_summary$n_repeats)
  vi_summary <- vi_summary[order(-vi_summary$importance_mean), ]

  rownames(vi_summary) <- NULL
  vi_summary
}

plot_variable_importance <- function(vi_summary, file, width, height, res) {
  png(file, width = width, height = height, units = "in", res = res)
  oldpar <- par(no.readonly = TRUE)
  on.exit({
    par(oldpar)
    dev.off()
  }, add = TRUE)

  par(mar = c(5, 4, 2, 2))
  ylim <- c(0, max(vi_summary$importance_mean + vi_summary$importance_se, na.rm = TRUE) * 1.15)
  bp <- barplot(
    vi_summary$importance_mean,
    las = 1,
    ylim = ylim,
    axes = FALSE,
    ylab = "Importance (dropout loss)"
  )
  grid()
  barplot(vi_summary$importance_mean, add = TRUE, col = adjustcolor("gray50", alpha.f = 0.8))
  arrows(
    x0 = bp[, 1],
    y0 = vi_summary$importance_mean - vi_summary$importance_se,
    y1 = vi_summary$importance_mean + vi_summary$importance_se,
    code = 3,
    length = 0.08,
    angle = 90,
    lwd = 1.5
  )
  axis(2, las = 1)
  text(
    x = bp[, 1],
    y = par()$usr[3] - (par()$usr[4] - par()$usr[3]) / 30,
    labels = vi_summary$variable,
    xpd = TRUE,
    cex = 0.85,
    srt = 45,
    adj = 1
  )
}

plot_pdp_by_variable <- function(pdp_all, pdp_summary, presence_env, vars, output_dir,
                                 width, height, res) {
  for (v in vars) {
    db_pdp <- pdp_all[pdp_all$variable == v, , drop = FALSE]
    db_mean <- pdp_summary[pdp_summary$variable == v, , drop = FALSE]

    if (nrow(db_pdp) == 0 || nrow(db_mean) == 0) next

    ylimit <- range(c(db_mean$ci_low, db_mean$ci_high, db_pdp$value.y), na.rm = TRUE)
    file <- file.path(output_dir, paste0("PDP_", v, ".png"))

    png(file, width = width, height = height, units = "in", res = res)
    oldpar <- par(no.readonly = TRUE)
    on.exit({
      par(oldpar)
      dev.off()
    }, add = TRUE)

    par(mar = c(4, 4, 2, 2))
    plot(
      db_pdp$value.x,
      db_pdp$value.y,
      type = "n",
      las = 1,
      ylim = ylimit,
      xlab = v,
      ylab = "Habitat suitability"
    )
    grid()

    if (!is.null(presence_env) && v %in% names(presence_env)) {
      p_range <- range(presence_env[[v]], na.rm = TRUE)
      rect(
        p_range[1], par()$usr[3],
        p_range[2], par()$usr[4],
        col = adjustcolor("gray70", alpha.f = 0.25),
        border = NA
      )
    }

    polygon(
      x = c(db_mean$value.x, rev(db_mean$value.x)),
      y = c(db_mean$ci_high, rev(db_mean$ci_low)),
      border = NA,
      col = adjustcolor("gray50", alpha.f = 0.25)
    )

    for (k in sort(unique(db_pdp$repeat.num))) {
      tmp <- db_pdp[db_pdp$repeat.num == k, , drop = FALSE]
      tmp <- tmp[order(tmp$value.x), ]
      lines(tmp$value.x, tmp$value.y, lwd = 0.5, lty = 1, col = "gray40")
    }

    lines(db_mean$value.x, db_mean$y.mean, lwd = 3)

    par(oldpar)
    dev.off()
    on.exit(NULL, add = FALSE)
  }
}

################################################################################
# 3. LOAD SCRIPT 1 OUTPUTS
################################################################################

message("Loading Script 1 outputs...")

training_data <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$training_data_file))
rf_output <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$rf_output_file))
maxent_output <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$maxent_output_file))

occurrence_environment_path <- file.path(cfg$sdm_input_dir, cfg$occurrence_environment_file)
occurrence_environment <- if (file.exists(occurrence_environment_path)) readRDS(occurrence_environment_path) else NULL

selected_vars <- training_data$selected_vars
original_db <- training_data$original_db
test_sets <- training_data$test_sets

rf_models <- rf_output$models
maxent_models <- maxent_output$models

if (length(rf_models) != length(maxent_models)) {
  stop("RF and MaxEnt model lists have different lengths.", call. = FALSE)
}

n_repeats <- min(cfg$n_repeats, length(rf_models), length(maxent_models), length(test_sets))

if (!all(selected_vars %in% names(original_db))) {
  stop("Some selected variables are missing from original_db.", call. = FALSE)
}

################################################################################
# 4. REPEATED ENSEMBLE PERFORMANCE
################################################################################

message("Calculating repeated ensemble performance...")

ensemble_eval_repeated <- vector("list", n_repeats)
ensemble_test_predictions <- vector("list", n_repeats)

for (k in seq_len(n_repeats)) {
  test_db <- test_sets[[k]]
  pred_raw <- predict_ensemble(rf_models[[k]], maxent_models[[k]], test_db, selected_vars)
  threshold_k <- calc_spec_sens_threshold(test_db$OC, pred_raw)
  pred_class <- ifelse(pred_raw >= threshold_k, 1, 0)

  metrics <- calc_metrics(test_db$OC, pred_raw, pred_class)

  ensemble_eval_repeated[[k]] <- cbind(
    data.frame(
      repeat.num = k,
      threshold = threshold_k,
      N.test = nrow(test_db),
      N.test.OC0 = sum(test_db$OC == 0),
      N.test.OC1 = sum(test_db$OC == 1)
    ),
    metrics
  )

  ensemble_test_predictions[[k]] <- cbind(
    data.frame(repeat.num = k, OC = test_db$OC, Pred.raw = pred_raw, Pred.OC = pred_class),
    test_db[, selected_vars, drop = FALSE]
  )
}

ensemble_eval_repeated <- do.call(rbind, ensemble_eval_repeated)
ensemble_test_predictions <- do.call(rbind, ensemble_test_predictions)

write.csv(
  ensemble_eval_repeated,
  file.path(cfg$output_dir, "ensemble_repeated_test_performance.csv"),
  row.names = FALSE
)

write.csv(
  ensemble_test_predictions,
  file.path(cfg$output_dir, "ensemble_repeated_test_predictions.csv"),
  row.names = FALSE
)

################################################################################
# 5. FINAL ENSEMBLE THRESHOLD FROM POOLED OOF TEST PREDICTIONS
################################################################################

message("Calculating final ensemble threshold from pooled test predictions...")

# Assign row IDs by unique environmental rows. This follows the original workflow
# where repeated test predictions are pooled at the unique-row level.
unique_env <- unique(ensemble_test_predictions[, selected_vars, drop = FALSE])
unique_env$row.ID <- seq_len(nrow(unique_env))

oof_tbl <- merge(
  ensemble_test_predictions,
  unique_env,
  by = selected_vars,
  all.x = TRUE
)

y_by_id <- aggregate(OC ~ row.ID, data = oof_tbl, FUN = function(x) x[1])
p_by_id <- aggregate(Pred.raw ~ row.ID, data = oof_tbl, FUN = function(x) mean(x, na.rm = TRUE))

oof_final <- merge(y_by_id, p_by_id, by = "row.ID", all = TRUE)
names(oof_final)[names(oof_final) == "Pred.raw"] <- "p_oof"

final_threshold <- calc_spec_sens_threshold(oof_final$OC, oof_final$p_oof)
final_metrics <- calc_metrics(
  y_true = oof_final$OC,
  y_pred_raw = oof_final$p_oof,
  y_pred_class = ifelse(oof_final$p_oof >= final_threshold, 1, 0)
)

species_info <- if (!is.null(training_data$species_info)) training_data$species_info else data.frame(species = NA_character_)
final_performance <- cbind(
  species_info[rep(1, nrow(final_metrics)), , drop = FALSE],
  data.frame(threshold_ensemble = final_threshold),
  final_metrics
)

write.csv(
  oof_final,
  file.path(cfg$output_dir, "ensemble_oof_predictions_for_final_threshold.csv"),
  row.names = FALSE
)

write.csv(
  final_performance,
  file.path(cfg$output_dir, "ensemble_final_threshold_performance.csv"),
  row.names = FALSE
)

################################################################################
# 6. DALEX EXPLAINERS FOR ENSEMBLE MODELS
################################################################################

message("Building DALEX explainers for ensemble models...")

bgX <- original_db[, selected_vars, drop = FALSE]
bgY <- original_db$OC

ensemble_predict_function <- make_ensemble_predict_function(selected_vars)

explainer_list <- vector("list", n_repeats)

for (k in seq_len(n_repeats)) {
  message("  Explainer repeat ", k, " / ", n_repeats)

  model_obj <- list(
    rf = rf_models[[k]],
    maxent = maxent_models[[k]]
  )

  test_db <- test_sets[[k]]
  testX <- test_db[, selected_vars, drop = FALSE]
  testY <- test_db$OC

  expl_test <- DALEX::explain(
    model = model_obj,
    data = testX,
    y = testY,
    predict_function = ensemble_predict_function,
    type = "classification",
    label = paste0("ensemble_test_repeat_", k),
    verbose = FALSE
  )

  expl_pdp <- DALEX::explain(
    model = model_obj,
    data = bgX,
    y = bgY,
    predict_function = ensemble_predict_function,
    type = "classification",
    label = paste0("ensemble_pdp_repeat_", k),
    verbose = FALSE
  )

  explainer_list[[k]] <- list(expl_test = expl_test, expl_pdp = expl_pdp)
}

################################################################################
# 7. VARIABLE IMPORTANCE
################################################################################

message("Calculating permutation variable importance...")

vi_list <- vector("list", n_repeats)

for (k in seq_len(n_repeats)) {
  set.seed(cfg$seed + k)

  loss_fun <- switch(
    cfg$variable_importance_loss,
    one_minus_auc = DALEX::loss_one_minus_auc,
    DALEX::loss_one_minus_auc
  )

  vi <- DALEX::model_parts(
    explainer = explainer_list[[k]]$expl_test,
    loss_function = loss_fun,
    type = "difference",
    N = NULL
  )

  vi_df <- as.data.frame(vi)
  vi_df$repeat.num <- k
  vi_list[[k]] <- vi_df
}

vi_all <- do.call(rbind, vi_list)
vi_all <- vi_all[vi_all$permutation == 0, , drop = FALSE]
vi_all <- vi_all[!vi_all$variable %in% c("_baseline_", "_full_model_"), , drop = FALSE]
vi_summary <- summarize_variable_importance(vi_all)

write.csv(
  vi_all,
  file.path(cfg$output_dir, "ensemble_variable_importance_all_repeats.csv"),
  row.names = FALSE
)

write.csv(
  vi_summary,
  file.path(cfg$output_dir, "ensemble_variable_importance_summary.csv"),
  row.names = FALSE
)

################################################################################
# 8. PARTIAL DEPENDENCE PROFILES USING COMMON GRID
################################################################################

message("Calculating PDPs using a common grid...")

variable_splits <- make_variable_splits_equal(bgX, selected_vars, n = cfg$pdp_grid_n)
pdp_list <- vector("list", n_repeats)

for (k in seq_len(n_repeats)) {
  message("  PDP repeat ", k, " / ", n_repeats)
  set.seed(6808 + k)

  pdp <- DALEX::model_profile(
    explainer = explainer_list[[k]]$expl_pdp,
    variables = selected_vars,
    type = "partial",
    variable_splits = variable_splits,
    N = cfg$pdp_sample_n
  )

  pdp_list[[k]] <- data.frame(
    variable = pdp$agr_profiles$`_vname_`,
    value.x = pdp$agr_profiles$`_x_`,
    value.y = pdp$agr_profiles$`_yhat_`,
    repeat.num = k
  )
}

pdp_all <- do.call(rbind, pdp_list)
pdp_summary <- summarize_pdp(pdp_all, n_repeats)

write.csv(
  pdp_all,
  file.path(cfg$output_dir, "ensemble_pdp_all_repeats.csv"),
  row.names = FALSE
)

write.csv(
  pdp_summary,
  file.path(cfg$output_dir, "ensemble_pdp_summary.csv"),
  row.names = FALSE
)

################################################################################
# 9. OPTIONAL FIGURES
################################################################################

if (isTRUE(cfg$save_figures)) {
  message("Saving figures...")

  fig_dir <- file.path(cfg$output_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  plot_variable_importance(
    vi_summary = vi_summary,
    file = file.path(fig_dir, "variable_importance.png"),
    width = cfg$figure_width,
    height = cfg$figure_height,
    res = cfg$figure_res
  )

  presence_env <- occurrence_environment
  plot_pdp_by_variable(
    pdp_all = pdp_all,
    pdp_summary = pdp_summary,
    presence_env = presence_env,
    vars = selected_vars,
    output_dir = fig_dir,
    width = cfg$figure_width,
    height = cfg$figure_height,
    res = cfg$figure_res
  )
}

################################################################################
# 10. SAVE COMPACT RDS OUTPUT
################################################################################

saveRDS(
  list(
    selected_vars = selected_vars,
    ensemble_repeated_performance = ensemble_eval_repeated,
    ensemble_final_threshold = final_threshold,
    ensemble_final_performance = final_performance,
    ensemble_variable_importance_all = vi_all,
    ensemble_variable_importance_summary = vi_summary,
    ensemble_pdp_all = pdp_all,
    ensemble_pdp_summary = pdp_summary
  ),
  file.path(cfg$output_dir, "02_sdm_evaluation_varimp_pdp_outputs.rds")
)

message("Completed. Outputs saved in: ", normalizePath(cfg$output_dir))

################################################################################
# End of script
################################################################################
