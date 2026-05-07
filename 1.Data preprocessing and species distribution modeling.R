################################################################################
# 1. Data preprocessing and species distribution modeling
#
# Main steps:
#   1) Load environmental raster layers (CHELSA bioclimatic variables + elevation)
#   2) Load and filter occurrence records
#   3) Extract environmental values for occurrence records
#   4) Generate and filter background / pseudo-absence points
#   5) Select predictors after multicollinearity filtering
#   6) Split data into repeated train/test sets
#   7) Optimize and fit MaxEnt and RF models
#   8) Save model objects, evaluation metrics, and predictions
#
# Notes for public release:
#   - Replace the paths in the CONFIG section with your local/project paths.
#   - Input occurrence data should contain at least: species, lon, lat, year,
#     basisofrecord. If your file already contains cleaned records, set
#     cfg$occurrence_has_gbif_fields <- FALSE.
#   - MaxEnt uses background samples; RF treats the same non-presence points as
#     pseudo-absences, following the workflow described in the manuscript.
################################################################################

rm(list = ls())

################################################################################
# 0. CONFIGURATION
################################################################################

cfg <- list(
  # Project directories
  project_dir = getwd(),
  chelsa_baseline_dir = "data/raw/CHELSA_1981-2010",
  elevation_file = "data/raw/wc2.1_30s_elev/wc2.1_30s_elev.tif",
  occurrence_file = "data/raw/sagra_femorata_occurrences.csv",
  output_dir = "outputs/01_sdm",

  # Occurrence filtering
  occurrence_has_gbif_fields = TRUE,
  min_year = 1981,
  accepted_basis = c("HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION", "OCCURRENCE"),
  species_name = "Sagra femorata",

  # Background / pseudo-absence generation
  n_background_initial = 15000,
  n_background_final = 10000,
  spatial_weight_in_buffer = 0.30,
  latitude_min = -60,
  latitude_max = 80,
  min_distance_m = 5000,
  varela_bins = 25,

  # Modeling
  n_repeats = 10,
  test_fraction = 0.20,
  k_fold = 5,
  seed = 910520,
  methods = c("RF", "MaxEnt"),
  n_cores = max(1, parallel::detectCores() - 3),

  # Bayesian optimization settings
  bayes_init_points = 5,
  bayes_iter_n = 5,
  bayes_iter_k = 5
)

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# 1. PACKAGES
################################################################################

required_packages <- c(
  "terra", "sf", "dplyr", "megaSDM", "usdm", "maxnet", "ranger", "dismo",
  "caret", "MLmetrics", "ParBayesianOptimization", "doParallel", "parallel",
  "glmnet", "foreach"
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
  library(terra)
  library(sf)
  library(dplyr)
  library(megaSDM)
  library(usdm)
  library(maxnet)
  library(ranger)
  library(dismo)
  library(caret)
  library(MLmetrics)
  library(ParBayesianOptimization)
  library(doParallel)
})

################################################################################
# 2. HELPER FUNCTIONS
################################################################################

standardize_chelsa_names <- function(x) {
  nm <- names(x)
  bio_id <- vapply(strsplit(nm, "_"), function(z) {
    hit <- grep("^bio[0-9]+$", z, value = TRUE)
    if (length(hit) == 0) return(NA_character_)
    hit[1]
  }, character(1))
  bio_id[is.na(bio_id)] <- nm[is.na(bio_id)]
  names(x) <- bio_id
  x
}

load_environment_layers <- function(chelsa_dir, elevation_file) {
  chelsa_files <- list.files(chelsa_dir, pattern = "bio.*\\.tif$", full.names = TRUE, ignore.case = TRUE)
  if (length(chelsa_files) == 0) {
    stop("No CHELSA bioclimatic .tif files were found in: ", chelsa_dir, call. = FALSE)
  }

  env_chelsa <- terra::rast(chelsa_files)
  env_chelsa <- standardize_chelsa_names(env_chelsa)

  env_elev <- terra::rast(elevation_file)
  env_elev <- terra::crop(env_elev, env_chelsa)
  names(env_elev) <- "elev"

  env_maps <- c(env_elev, env_chelsa)
  env_maps
}

load_occurrence_data <- function(file, cfg) {
  if (!file.exists(file)) stop("Occurrence file not found: ", file, call. = FALSE)

  dat <- read.csv(file, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
  names(dat) <- tolower(names(dat))

  # Accept common longitude/latitude variants.
  if (!"lon" %in% names(dat)) {
    lon_candidates <- c("longitude", "decimal_longitude", "decimallongitude")
    hit <- lon_candidates[lon_candidates %in% names(dat)][1]
    if (!is.na(hit)) names(dat)[names(dat) == hit] <- "lon"
  }
  if (!"lat" %in% names(dat)) {
    lat_candidates <- c("latitude", "decimal_latitude", "decimallatitude")
    hit <- lat_candidates[lat_candidates %in% names(dat)][1]
    if (!is.na(hit)) names(dat)[names(dat) == hit] <- "lat"
  }
  if (!all(c("lon", "lat") %in% names(dat))) {
    stop("Occurrence file must contain longitude and latitude columns.", call. = FALSE)
  }

  if (!"species" %in% names(dat)) dat$species <- cfg$species_name
  if (!"siteid" %in% names(dat)) dat$siteid <- paste0("OBS", seq_len(nrow(dat)))

  if (isTRUE(cfg$occurrence_has_gbif_fields)) {
    if ("year" %in% names(dat)) dat <- dat[dat$year >= cfg$min_year | is.na(dat$year), , drop = FALSE]
    if ("basisofrecord" %in% names(dat)) dat <- dat[dat$basisofrecord %in% cfg$accepted_basis, , drop = FALSE]
  }

  dat <- dat[is.finite(dat$lon) & is.finite(dat$lat), , drop = FALSE]
  dat <- dat[dat$lon >= -180 & dat$lon <= 180 & dat$lat >= -90 & dat$lat <= 90, , drop = FALSE]
  dat <- dat[, c("siteid", "species", "lon", "lat")]
  names(dat) <- c("siteID", "species", "lon", "lat")
  dat
}

extract_environment_at_points <- function(occ_df, env_maps) {
  env_values <- terra::extract(env_maps, occ_df[, c("lon", "lat")])
  out <- cbind(occ_df, env_values[, -1, drop = FALSE])
  out <- out[complete.cases(out), , drop = FALSE]
  rownames(out) <- NULL
  out
}

make_buffer_from_occurrences <- function(occ_env_df, output_shp) {
  coordinates <- terra::vect(occ_env_df, geom = c("lon", "lat"), crs = "EPSG:4326")
  distance_mat <- as.matrix(terra::distance(coordinates))

  min_dist <- numeric(ncol(distance_mat))
  for (i in seq_len(ncol(distance_mat))) {
    dz <- distance_mat[distance_mat[, i] > 0, i]
    min_dist[i] <- min(dz)
  }

  buffer_width <- 2 * stats::quantile(min_dist, 0.95, na.rm = TRUE)
  buffer_poly <- terra::buffer(coordinates, buffer_width)
  buffer_poly <- terra::aggregate(buffer_poly)
  buffer_poly <- terra::project(buffer_poly, terra::crs(coordinates))
  terra::writeVector(buffer_poly, output_shp, filetype = "ESRI Shapefile", overwrite = TRUE)

  list(buffer = buffer_poly, buffer_width = buffer_width)
}

# Environmental filtering based on Varela et al. style environmental bins.
# Adapted from the workflow used for this study.
varela_sample <- function(env_occur, no_bins = 25, pca = TRUE, pca_axes = "auto") {
  clim_occur <- env_occur[stats::complete.cases(env_occur), , drop = FALSE]

  if (isTRUE(pca)) {
    pca_env <- stats::prcomp(clim_occur[, 3:ncol(clim_occur), drop = FALSE], scale. = TRUE)
    pca_imp <- summary(pca_env)$importance

    if (is.numeric(pca_axes)) {
      number_axes <- pca_axes
    } else {
      number_axes <- max(2, min(which(pca_imp[3, ] > 0.95)))
    }

    env_occur <- data.frame(clim_occur[, 1:2, drop = FALSE], pca_env$x[, 1:number_axes, drop = FALSE])
  }

  out_ptz <- env_occur[, 1:2, drop = FALSE]
  for (i in 3:ncol(env_occur)) {
    k <- env_occur[!is.na(env_occur[, i]), i]
    rg <- range(k)
    res <- (rg[2] - rg[1]) / no_bins
    d <- (env_occur[, i] - rg[1]) / res
    f <- ceiling(d)
    f[f == 0] <- 1
    out_ptz[[names(env_occur)[i]]] <- f
  }

  sub_ptz <- dplyr::distinct(out_ptz[, -c(1, 2), drop = FALSE])
  sub_ptz$grp <- seq_len(nrow(sub_ptz))
  out_ptz <- suppressMessages(dplyr::left_join(out_ptz, sub_ptz))

  final_out <- data.frame(x = numeric(), y = numeric())
  for (grp_id in seq_len(nrow(sub_ptz))) {
    grp_members <- out_ptz[out_ptz$grp == grp_id, c(1, 2), drop = FALSE]
    final_out <- rbind(final_out, grp_members[sample(seq_len(nrow(grp_members)), 1), , drop = FALSE])
  }

  final_out <- data.frame(x = final_out[, 1], y = final_out[, 2])
  merge(final_out, clim_occur, by = c("x", "y"), all.x = TRUE)
}

thin_points_batch_terra <- function(df, lng_col = "lon", lat_col = "lat", threshold,
                                    batch_size = 100L, shuffle = TRUE, seed = 910520,
                                    verbose = FALSE, precompute_within = TRUE) {
  stopifnot(is.data.frame(df))
  if (!all(c(lng_col, lat_col) %in% names(df))) {
    stop("Coordinate columns were not found in df.", call. = FALSE)
  }

  tt <- df
  if (shuffle) {
    if (!is.null(seed)) set.seed(seed)
    tt <- tt[sample(nrow(tt)), , drop = FALSE]
    rownames(tt) <- NULL
  }

  coords <- as.matrix(tt[, c(lng_col, lat_col), drop = FALSE])
  n <- nrow(coords)
  keep_idx <- rep(FALSE, n)
  kept_sv <- NULL

  mk_vect <- function(xy) terra::vect(data.frame(x = xy[, 1], y = xy[, 2]), geom = c("x", "y"), crs = "EPSG:4326")

  for (start in seq(1L, n, by = batch_size)) {
    end <- min(start + batch_size - 1L, n)
    idx_batch <- start:end
    a_xy <- coords[idx_batch, , drop = FALSE]
    a_sv <- mk_vect(a_xy)

    ok_global <- rep(TRUE, nrow(a_xy))
    if (!is.null(kept_sv)) {
      d_kept <- terra::distance(a_sv, kept_sv)
      if (!is.matrix(d_kept)) d_kept <- matrix(d_kept, nrow = nrow(a_xy))
      ok_global <- apply(d_kept, 1L, min) > threshold
    }

    if (any(ok_global)) {
      if (precompute_within) {
        d_within <- terra::distance(a_sv, a_sv)
        if (!is.matrix(d_within)) d_within <- matrix(d_within, nrow = nrow(a_sv))
        diag(d_within) <- NA_real_
      }

      local_kept <- integer(0)
      for (j in which(ok_global)) {
        ok_local <- TRUE
        if (length(local_kept)) {
          if (precompute_within) {
            if (min(d_within[j, local_kept], na.rm = TRUE) <= threshold) ok_local <- FALSE
          } else {
            d_tmp <- terra::distance(a_sv[j], a_sv[local_kept])
            if (min(d_tmp) <= threshold) ok_local <- FALSE
          }
        }

        if (!ok_local) next
        keep_idx[idx_batch[j]] <- TRUE
        local_kept <- c(local_kept, j)
        kept_sv <- if (is.null(kept_sv)) a_sv[j] else rbind(kept_sv, a_sv[j])
        if (verbose) message("Kept row: ", idx_batch[j])
      }
    }
  }

  res <- tt[keep_idx, , drop = FALSE]
  rownames(res) <- NULL
  res
}

evaluate_sdm_score <- function(y_true, y_pred, y_pred_class) {
  tss <- tryCatch({
    MLmetrics::Sensitivity(y_true = y_true, y_pred = y_pred_class, positive = 1) +
      MLmetrics::Specificity(y_true = y_true, y_pred = y_pred_class, positive = 1) - 1
  }, error = function(e) NA_real_)

  prauc <- tryCatch({
    MLmetrics::PRAUC(y_pred = y_pred, y_true = y_true)
  }, error = function(e) NA_real_)

  if (is.na(tss)) tss <- 0
  if (is.na(prauc)) prauc <- 0
  data.frame(TSS = tss, PRAUC = prauc)
}

safe_minmax <- function(x) {
  rng <- max(x, na.rm = TRUE) - min(x, na.rm = TRUE)
  if (!is.finite(rng) || rng == 0) return(rep(1, length(x)))
  (x - min(x, na.rm = TRUE)) / rng
}

compute_mcc <- function(y_true, y_pred_class) {
  tp <- sum(y_pred_class == 1 & y_true == 1)
  tn <- sum(y_pred_class == 0 & y_true == 0)
  fp <- sum(y_pred_class == 1 & y_true == 0)
  fn <- sum(y_pred_class == 0 & y_true == 1)
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

predict_sdm <- function(model, newdata, method) {
  if (method == "MaxEnt") {
    as.numeric(predict(model, newdata, type = "cloglog"))
  } else if (method == "RF") {
    as.numeric(predict(model, newdata)$predictions)
  } else {
    stop("Unknown method: ", method, call. = FALSE)
  }
}

################################################################################
# 3. LOAD ENVIRONMENTAL AND OCCURRENCE DATA
################################################################################

message("Loading environmental layers...")
env_maps <- load_environment_layers(cfg$chelsa_baseline_dir, cfg$elevation_file)

message("Loading occurrence data...")
occ <- load_occurrence_data(cfg$occurrence_file, cfg)

message("Extracting environmental values at occurrence points...")
occ_env <- extract_environment_at_points(occ, env_maps)
saveRDS(occ_env, file.path(cfg$output_dir, "01_occurrence_environment.rds"))

################################################################################
# 4. GENERATE BACKGROUND / PSEUDO-ABSENCE POINTS
################################################################################

message("Generating occurrence buffer...")
buffer_shp <- file.path(cfg$output_dir, "occurrence_buffer.shp")
buffer_info <- make_buffer_from_occurrences(occ_env, buffer_shp)
message("Buffer width (m): ", round(buffer_info$buffer_width, 2))

message("Generating background points using megaSDM::BackgroundPoints...")
bg_output_dir <- file.path(cfg$output_dir, paste0("background_n", cfg$n_background_initial))
dir.create(bg_output_dir, recursive = TRUE, showWarnings = FALSE)

megaSDM::BackgroundPoints(
  spplist = c("sf"),
  envdata = env_maps,
  output = bg_output_dir,
  spatial_weights = cfg$spatial_weight_in_buffer,
  nbg = cfg$n_background_initial,
  buffers = terra::vect(buffer_shp),
  method = "Varela",
  ncores = cfg$n_cores
)

bg_file <- file.path(bg_output_dir, "sf_background.csv")
if (!file.exists(bg_file)) stop("Background file was not generated: ", bg_file, call. = FALSE)

bg <- read.csv(bg_file, stringsAsFactors = FALSE)
bg <- bg[, -1, drop = FALSE]
names(bg)[1:2] <- c("lon", "lat")
bg <- bg[bg$lat >= cfg$latitude_min & bg$lat <= cfg$latitude_max, , drop = FALSE]

message("Applying environmental filtering to background points...")
env_a <- bg[, !(names(bg) %in% c("lon", "lat")), drop = FALSE]
env_p <- occ_env[, !(names(occ_env) %in% c("siteID", "species", "lon", "lat")), drop = FALSE]

# Remove zero-variance variables for PCA-based filtering.
nonzero_vars <- names(env_p)[vapply(env_p, var, numeric(1), na.rm = TRUE) != 0]
nonzero_vars <- intersect(nonzero_vars, names(env_a)[vapply(env_a, var, numeric(1), na.rm = TRUE) != 0])

env_a2 <- env_a[, nonzero_vars, drop = FALSE]
env_p2 <- env_p[, nonzero_vars, drop = FALSE]

set.seed(6808)
bg_selected_rows <- varela_sample(
  env_occur = data.frame(x = seq_len(nrow(env_a2)), y = seq_len(nrow(env_a2)), env_a2),
  no_bins = cfg$varela_bins,
  pca = TRUE,
  pca_axes = "auto"
)
background_filtered <- bg[bg_selected_rows$x, , drop = FALSE]

message("Removing background points within ", cfg$min_distance_m, " m of presence points...")
presence_sf <- sf::st_as_sf(occ_env, coords = c("lon", "lat"), crs = 4326)
background_sf <- sf::st_as_sf(background_filtered, coords = c("lon", "lat"), crs = 4326)
near_presence <- sf::st_is_within_distance(background_sf, presence_sf, dist = cfg$min_distance_m)
background_filtered <- background_filtered[lengths(near_presence) == 0, , drop = FALSE]

message("Applying distance-based thinning to background points...")
background_final <- thin_points_batch_terra(
  df = background_filtered,
  lng_col = "lon",
  lat_col = "lat",
  threshold = cfg$min_distance_m,
  batch_size = 100,
  shuffle = TRUE,
  seed = cfg$seed,
  verbose = FALSE
)

if (nrow(background_final) > cfg$n_background_final) {
  set.seed(123)
  background_final <- background_final[sample(seq_len(nrow(background_final)), cfg$n_background_final), , drop = FALSE]
}

message("Applying environmental filtering to presence records...")
set.seed(919)
presence_selected_rows <- varela_sample(
  env_occur = data.frame(x = seq_len(nrow(env_p2)), y = seq_len(nrow(env_p2)), env_p2),
  no_bins = cfg$varela_bins,
  pca = TRUE,
  pca_axes = "auto"
)
presence_final <- occ_env[presence_selected_rows$x, , drop = FALSE]

species_info <- data.frame(
  species = cfg$species_name,
  N_presence = nrow(presence_final),
  N_background = nrow(background_final)
)

saveRDS(
  list(presence = presence_final, background = background_final, species_info = species_info),
  file.path(cfg$output_dir, "02_presence_background_dataset.rds")
)

################################################################################
# 5. VARIABLE SELECTION AND REPEATED TRAIN/TEST SPLIT
################################################################################

message("Selecting predictors and creating repeated train/test datasets...")

db_p <- presence_final[, !(names(presence_final) %in% c("siteID", "species", "lon", "lat")), drop = FALSE]
db_a <- background_final[, !(names(background_final) %in% c("lon", "lat")), drop = FALSE]

db_p$OC <- 1
db_a$OC <- 0
sdm_db <- rbind(db_p, db_a)

candidate_env <- sdm_db[, !(names(sdm_db) %in% "OC"), drop = FALSE]
vif_cor <- usdm::vifcor(candidate_env, th = 0.9)
env_after_cor <- usdm::exclude(candidate_env, vif_cor)
vif_step <- usdm::vifstep(env_after_cor, th = 5)
env_after_vif <- usdm::exclude(env_after_cor, vif_step)

selected_vars <- names(env_after_vif)
sdm_formula <- as.formula(paste("OC ~", paste(selected_vars, collapse = " + ")))
sdm_db <- cbind(OC = sdm_db$OC, sdm_db[, selected_vars, drop = FALSE])

train_sets <- vector("list", cfg$n_repeats)
test_sets <- vector("list", cfg$n_repeats)

for (k in seq_len(cfg$n_repeats)) {
  p_rows <- which(sdm_db$OC == 1)
  a_rows <- which(sdm_db$OC == 0)

  set.seed(cfg$seed + k^2)
  test_p <- sample(p_rows, size = floor(cfg$test_fraction * length(p_rows)))
  test_a <- sample(a_rows, size = floor(cfg$test_fraction * length(a_rows)))
  test_rows <- c(test_p, test_a)

  test_sets[[k]] <- sdm_db[test_rows, , drop = FALSE]
  train_sets[[k]] <- sdm_db[-test_rows, , drop = FALSE]
}

saveRDS(
  list(
    species_info = species_info,
    original_db = sdm_db,
    selected_vars = selected_vars,
    sdm_formula = sdm_formula,
    train_sets = train_sets,
    test_sets = test_sets,
    vif_cor = vif_cor,
    vif_step = vif_step
  ),
  file.path(cfg$output_dir, "03_sdm_training_data.rds")
)

################################################################################
# 6. MODEL FITTING: MAXENT AND RANDOM FOREST
################################################################################

message("Starting SDM modeling...")

all_evaluation <- list()
all_prediction <- list()

for (method in cfg$methods) {
  message("Method: ", method)
  model_list <- vector("list", cfg$n_repeats)
  method_evaluation <- list()
  method_prediction <- list()
  method_parameters <- list()

  for (k in seq_len(cfg$n_repeats)) {
    message("  Repeat ", k, " / ", cfg$n_repeats)
    train_db <- train_sets[[k]]
    test_db <- test_sets[[k]]

    set.seed(cfg$seed * cfg$k_fold + k)
    folds <- caret::createFolds(train_db$OC, k = cfg$k_fold, returnTrain = FALSE)

    max_bounds <- list(regmult = c(0.1, 2.0))
    rf_bounds <- list(
      num.trees = c(500L, 1500L),
      mtry = c(1L, max(1L, length(selected_vars))),
      min.node.size = c(3L, 7L)
    )

    max_opt_fun <- function(regmult) {
      tss_vec <- numeric(cfg$k_fold)
      prauc_vec <- numeric(cfg$k_fold)

      for (ii in seq_len(cfg$k_fold)) {
        train_fold <- train_db[-folds[[ii]], , drop = FALSE]
        valid_fold <- train_db[folds[[ii]], , drop = FALSE]

        set.seed(cfg$seed + k * ii)
        model <- maxnet::maxnet(
          p = train_fold$OC,
          data = train_fold[, selected_vars, drop = FALSE],
          f = maxnet::maxnet.formula(train_fold$OC, train_fold[, selected_vars, drop = FALSE], classes = "default"),
          regmult = regmult,
          addsamplestobackground = TRUE
        )

        pred_raw <- as.numeric(predict(model, valid_fold[, selected_vars, drop = FALSE], type = "cloglog"))
        eval_obj <- dismo::evaluate(p = pred_raw[valid_fold$OC == 1], a = pred_raw[valid_fold$OC == 0])
        thr <- dismo::threshold(eval_obj, "spec_sens")
        pred_class <- ifelse(pred_raw >= thr, 1, 0)
        score <- evaluate_sdm_score(valid_fold$OC, pred_raw, pred_class)
        tss_vec[ii] <- score$TSS
        prauc_vec[ii] <- score$PRAUC
      }

      total_score <- safe_minmax(tss_vec) + safe_minmax(prauc_vec)
      list(Score = mean(total_score))
    }

    rf_opt_fun <- function(num.trees, mtry, min.node.size) {
      tss_vec <- numeric(cfg$k_fold)
      prauc_vec <- numeric(cfg$k_fold)

      for (ii in seq_len(cfg$k_fold)) {
        train_fold <- train_db[-folds[[ii]], , drop = FALSE]
        valid_fold <- train_db[folds[[ii]], , drop = FALSE]

        set.seed(cfg$seed + k * ii)
        model <- ranger::ranger(
          formula = sdm_formula,
          data = train_fold,
          num.trees = as.integer(num.trees),
          mtry = as.integer(mtry),
          min.node.size = as.integer(min.node.size)
        )

        pred_raw <- as.numeric(predict(model, valid_fold)$predictions)
        eval_obj <- dismo::evaluate(p = pred_raw[valid_fold$OC == 1], a = pred_raw[valid_fold$OC == 0])
        thr <- dismo::threshold(eval_obj, "spec_sens")
        pred_class <- ifelse(pred_raw >= thr, 1, 0)
        score <- evaluate_sdm_score(valid_fold$OC, pred_raw, pred_class)
        tss_vec[ii] <- score$TSS
        prauc_vec[ii] <- score$PRAUC
      }

      total_score <- safe_minmax(tss_vec) + safe_minmax(prauc_vec)
      list(Score = mean(total_score))
    }

    opt_obj <- NULL
    cl <- NULL
    try({
      cl <- parallel::makeCluster(cfg$n_cores)
      doParallel::registerDoParallel(cl)
      parallel::clusterExport(
        cl,
        varlist = c(
          "cfg", "train_db", "selected_vars", "folds", "sdm_formula",
          "evaluate_sdm_score", "safe_minmax", "max_opt_fun", "rf_opt_fun"
        ),
        envir = environment()
      )

      if (method == "MaxEnt") {
        parallel::clusterEvalQ(cl, { library(maxnet); library(glmnet); library(dismo); library(MLmetrics) })
        opt_obj <- ParBayesianOptimization::bayesOpt(
          FUN = max_opt_fun,
          bounds = max_bounds,
          initPoints = cfg$bayes_init_points,
          iters.n = cfg$bayes_iter_n,
          iters.k = cfg$bayes_iter_k,
          parallel = TRUE,
          verbose = 0
        )
      }

      if (method == "RF") {
        parallel::clusterEvalQ(cl, { library(ranger); library(dismo); library(MLmetrics) })
        opt_obj <- ParBayesianOptimization::bayesOpt(
          FUN = rf_opt_fun,
          bounds = rf_bounds,
          initPoints = cfg$bayes_init_points,
          iters.n = cfg$bayes_iter_n,
          iters.k = cfg$bayes_iter_k,
          parallel = TRUE,
          verbose = 0
        )
      }
    }, silent = TRUE)

    if (!is.null(cl)) parallel::stopCluster(cl)
    foreach::registerDoSEQ()

    if (!is.null(opt_obj)) {
      selected_pars <- as.data.frame(as.list(ParBayesianOptimization::getBestPars(opt_obj)))
      selected_pars$error <- NA_character_
    } else if (method == "MaxEnt") {
      selected_pars <- data.frame(regmult = 1, error = "Bayesian optimization failed")
    } else {
      selected_pars <- data.frame(
        num.trees = 500,
        mtry = max(1, floor(sqrt(length(selected_vars)))),
        min.node.size = 5,
        error = "Bayesian optimization failed"
      )
    }

    # Fit final model using the selected parameters.
    if (method == "MaxEnt") {
      set.seed(cfg$seed + k)
      final_model <- maxnet::maxnet(
        p = train_db$OC,
        data = train_db[, selected_vars, drop = FALSE],
        f = maxnet::maxnet.formula(train_db$OC, train_db[, selected_vars, drop = FALSE], classes = "default"),
        regmult = selected_pars$regmult,
        addsamplestobackground = TRUE
      )
    } else {
      set.seed(cfg$seed + k)
      final_model <- ranger::ranger(
        formula = sdm_formula,
        data = train_db,
        num.trees = as.integer(selected_pars$num.trees),
        mtry = as.integer(selected_pars$mtry),
        min.node.size = as.integer(selected_pars$min.node.size)
      )
    }

    model_list[[k]] <- final_model

    train_pred_raw <- predict_sdm(final_model, train_db, method)
    test_pred_raw <- predict_sdm(final_model, test_db, method)

    eval_train <- dismo::evaluate(p = train_pred_raw[train_db$OC == 1], a = train_pred_raw[train_db$OC == 0])
    eval_test <- dismo::evaluate(p = test_pred_raw[test_db$OC == 1], a = test_pred_raw[test_db$OC == 0])
    threshold_train <- dismo::threshold(eval_train, "spec_sens")
    threshold_test <- dismo::threshold(eval_test, "spec_sens")

    # Use test threshold for both train and test classification, as in the study workflow.
    train_pred_class <- ifelse(train_pred_raw >= threshold_test, 1, 0)
    test_pred_class <- ifelse(test_pred_raw >= threshold_test, 1, 0)

    train_metrics <- calc_metrics(train_db$OC, train_pred_raw, train_pred_class)
    test_metrics <- calc_metrics(test_db$OC, test_pred_raw, test_pred_class)
    names(train_metrics) <- paste0("train.", names(train_metrics))
    names(test_metrics) <- paste0("test.", names(test_metrics))

    info_row <- data.frame(
      species = cfg$species_name,
      SDM.method = method,
      repeat = k,
      Error.OC = selected_pars$error,
      N.train = nrow(train_db),
      N.train.OC0 = sum(train_db$OC == 0),
      N.train.OC1 = sum(train_db$OC == 1),
      N.test = nrow(test_db),
      N.test.OC0 = sum(test_db$OC == 0),
      N.test.OC1 = sum(test_db$OC == 1),
      Threshold.train = threshold_train,
      Threshold.test = threshold_test
    )

    method_evaluation[[k]] <- cbind(info_row, train_metrics, test_metrics)

    pred_db <- rbind(
      data.frame(data.type = "train", train_db, Pred.raw = train_pred_raw, Pred.OC = train_pred_class),
      data.frame(data.type = "test", test_db, Pred.raw = test_pred_raw, Pred.OC = test_pred_class)
    )
    pred_db <- cbind(
      data.frame(species = cfg$species_name, SDM.method = method, repeat = k,
                 Threshold.train = threshold_train, Threshold.test = threshold_test),
      pred_db
    )
    method_prediction[[k]] <- pred_db

    method_parameters[[k]] <- cbind(
      data.frame(species = cfg$species_name, SDM.method = method, repeat = k),
      selected_pars
    )
  }

  method_evaluation_df <- do.call(rbind, method_evaluation)
  method_prediction_df <- do.call(rbind, method_prediction)
  method_parameters_df <- do.call(rbind, method_parameters)

  saveRDS(
    list(
      species_info = species_info,
      selected_vars = selected_vars,
      sdm_formula = sdm_formula,
      models = model_list,
      evaluation = method_evaluation_df,
      predictions = method_prediction_df,
      parameters = method_parameters_df
    ),
    file.path(cfg$output_dir, paste0("04_sdm_output_", method, ".rds"))
  )

  all_evaluation[[method]] <- method_evaluation_df
  all_prediction[[method]] <- method_prediction_df
}

################################################################################
# 7. SAVE COMBINED OUTPUT TABLES
################################################################################

combined_evaluation <- do.call(rbind, all_evaluation)
combined_prediction <- do.call(rbind, all_prediction)

write.csv(combined_evaluation, file.path(cfg$output_dir, "SDM_evaluation.csv"), row.names = FALSE)
write.csv(combined_prediction, file.path(cfg$output_dir, "SDM_predictions.csv"), row.names = FALSE)

message("Completed. Outputs saved in: ", normalizePath(cfg$output_dir))

################################################################################
# End of script
################################################################################
