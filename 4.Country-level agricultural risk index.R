################################################################################
# 4. Country-level agricultural risk index
#
# This script is designed to run AFTER:
#   1.Data preprocessing and species distribution modeling.R
#   2_SDM_Evaluation_variable_importance_and_PDP.R
#
# Main steps:
#   1) Load trained RF and MaxEnt ensemble models from Script 1
#   2) Load the final ensemble threshold from Script 2
#   3) Project ensemble suitability maps for baseline/future environmental rasters
#      or read precomputed ensemble suitability maps
#   4) Overlay suitability maps with GADM country boundaries and GLAD cropland tiles
#   5) Calculate country-level occurrence area and cropland-weighted habitat hazard
#   6) Calculate host-crop production and upland-cropland proportion from FAOSTAT
#   7) Calculate the country-level agricultural risk index
#
# Notes for public release:
#   - Replace file paths in CONFIG with local/project paths.
#   - GLAD cropland class value is set to 244 by default, following the original
#     study workflow.
#   - Projection rasters must contain the same predictor variables used in Script 1.
#   - If you already created ensemble suitability maps, set cfg$use_precomputed_maps
#     to TRUE and place GeoTIFF files in cfg$precomputed_sdm_map_dir.
################################################################################

rm(list = ls())

################################################################################
# 0. CONFIGURATION
################################################################################

cfg <- list(
  project_dir = getwd(),

  # Outputs from previous scripts
  sdm_input_dir = "outputs/01_sdm",
  sdm_eval_dir  = "outputs/02_sdm_evaluation_varimp_pdp",
  output_dir    = "outputs/04_country_agricultural_risk",

  # Script 1 outputs
  training_data_file = "03_sdm_training_data.rds",
  rf_output_file     = "04_sdm_output_RF.rds",
  maxent_output_file = "04_sdm_output_MaxEnt.rds",

  # Script 2 output containing threshold_ensemble
  final_threshold_file = "ensemble_final_threshold_performance.csv",

  # Country boundary data
  gadm_file  = "data/raw/GADM/gadm_410-levels.gpkg",
  gadm_layer = "ADM_0",
  country_id_col = "GID_0",
  country_name_col = "COUNTRY",

  # GLAD 2020 land-cover tiles
  glad_dir = "data/raw/GLAD_2020",
  glad_file_pattern = "\\.tif$",
  glad_cropland_value = 244,

  # Suitability map option
  # FALSE: project RF/MaxEnt models to environmental rasters below.
  # TRUE : read existing ensemble suitability maps from precomputed_sdm_map_dir.
  use_precomputed_maps = FALSE,
  precomputed_sdm_map_dir = "outputs/04_country_agricultural_risk/01_ensemble_suitability_maps",
  precomputed_sdm_map_pattern = "\\.tif$",

  # Environmental projection raster inputs.
  # Use one directory per projection period/scenario. Each directory should contain
  # GeoTIFF layers named so selected predictor names can be detected (e.g., elev,
  # bio2, bio3, bio8, bio9, bio13, bio14, bio15, bio18, bio19).
  # The list names become projection IDs in outputs.
  projection_dirs = list(
    baseline = "data/processed/projection_rasters/baseline",
    nf_ssp126 = "data/processed/projection_rasters/near_future_ssp126",
    nf_ssp370 = "data/processed/projection_rasters/near_future_ssp370",
    nf_ssp585 = "data/processed/projection_rasters/near_future_ssp585",
    ff_ssp126 = "data/processed/projection_rasters/far_future_ssp126",
    ff_ssp370 = "data/processed/projection_rasters/far_future_ssp370",
    ff_ssp585 = "data/processed/projection_rasters/far_future_ssp585"
  ),
  projection_file_pattern = "\\.tif$",

  # FAOSTAT crop production / harvested-area table.
  # A single FAOSTAT Crops and Livestock Products CSV containing both Production
  # and Area harvested elements is recommended.
  faostat_crop_file = "data/raw/FAOSTAT_crops_livestock_products.csv",
  # Optional CSV with columns country_id and faostat_country_id or country_name
  # Use this when FAOSTAT country codes/names do not match GADM GID_0.
  country_crosswalk_file = NULL,
  production_years = 2020:2024,

  # Host crop items used for the production component. Edit these to match the
  # exact item names in your FAOSTAT file if needed.
  host_crop_items = c(
    "Soya beans", "Beans, dry", "Beans, green", "Pulses, nes",
    "Peas, dry", "Peas, green", "Lentils, dry", "Vetches", "Yams"
  ),

  # Crop items treated as paddy crops for upland-cropland proportion.
  rice_crop_items = c("Rice", "Rice, paddy", "Rice, paddy (rice milled equivalent)"),

  # Area-harvested rows used for total harvested area. If NULL, all FAOSTAT rows
  # with Element containing "Area harvested" are used.
  harvested_area_items = NULL,

  # Map projection metadata for cleaner outputs.
  projection_metadata = data.frame(
    projection_id = c("baseline", "nf_ssp126", "nf_ssp370", "nf_ssp585", "ff_ssp126", "ff_ssp370", "ff_ssp585"),
    period = c("baseline", "near_future", "near_future", "near_future", "far_future", "far_future", "far_future"),
    scenario = c("baseline", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5"),
    stringsAsFactors = FALSE
  ),

  # Processing options
  save_projected_maps = TRUE,
  projected_map_dir = "outputs/04_country_agricultural_risk/01_ensemble_suitability_maps",
  process_all_glad_tiles = TRUE,
  max_glad_tiles = Inf,     # Useful for testing; set to Inf for full run.
  overwrite_projected_maps = FALSE,

  # Habitat class thresholds
  high_suitability_threshold = 0.5,

  # Risk-index scaling.
  # "none" uses R = log1p(P_c) * H_c * U_c.
  # "zscore" standardizes risk scores within each projection.
  # "minmax" rescales risk scores to 0-1 within each projection.
  risk_score_scaling = "none"
)

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cfg$projected_map_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# 1. PACKAGES
################################################################################

required_packages <- c("terra", "dplyr", "tidyr", "stringr", "maxnet", "ranger")
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
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(maxnet)
  library(ranger)
})

################################################################################
# 2. HELPER FUNCTIONS
################################################################################

read_required_rds <- function(path) {
  if (!file.exists(path)) stop("Required input file not found: ", path, call. = FALSE)
  readRDS(path)
}

read_required_csv <- function(path) {
  if (!file.exists(path)) stop("Required input file not found: ", path, call. = FALSE)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

safe_name <- function(x) {
  x <- gsub("\\.tif$", "", basename(x), ignore.case = TRUE)
  x <- gsub("^1ens_", "", x)
  x <- gsub("^2final_", "", x)
  x
}

find_column <- function(df, candidates, required = TRUE) {
  nm <- names(df)
  hit <- candidates[candidates %in% nm]
  if (length(hit) > 0) return(hit[1])
  hit_lower <- match(tolower(candidates), tolower(nm))
  hit_lower <- hit_lower[!is.na(hit_lower)]
  if (length(hit_lower) > 0) return(nm[hit_lower[1]])
  if (required) stop("None of these columns were found: ", paste(candidates, collapse = ", "), call. = FALSE)
  NA_character_
}

extract_threshold <- function(threshold_file) {
  df <- read_required_csv(threshold_file)
  threshold_col <- find_column(df, c("threshold_ensemble", "thr1.Ensemble", "threshold", "Threshold"), required = TRUE)
  as.numeric(df[[threshold_col]][1])
}

load_projection_stack <- function(dir_path, selected_vars, pattern = "\\.tif$") {
  if (!dir.exists(dir_path)) stop("Projection raster directory not found: ", dir_path, call. = FALSE)
  files <- list.files(dir_path, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
  if (length(files) == 0) stop("No projection raster files found in: ", dir_path, call. = FALSE)

  r <- terra::rast(files)
  names(r) <- safe_name(files)

  # Try to standardize CHELSA-style names if layer names contain bio IDs.
  layer_names <- names(r)
  detected <- vapply(layer_names, function(z) {
    hit <- regmatches(z, regexpr("bio[0-9]+", z, ignore.case = TRUE))
    if (length(hit) == 0 || hit == "") z else tolower(hit)
  }, character(1))
  detected[grepl("elev|elevation", layer_names, ignore.case = TRUE)] <- "elev"
  names(r) <- detected

  missing <- setdiff(selected_vars, names(r))
  if (length(missing) > 0) {
    stop(
      "Projection raster stack in ", dir_path,
      " is missing selected predictor(s): ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  r[[selected_vars]]
}

predict_rf_raster <- function(model, env_stack) {
  terra::predict(
    env_stack,
    model,
    fun = function(model, data) as.numeric(predict(model, data)$predictions),
    na.rm = TRUE
  )
}

predict_maxent_raster <- function(model, env_stack) {
  terra::predict(
    env_stack,
    model,
    fun = function(model, data) as.numeric(predict(model, data, type = "cloglog")),
    na.rm = TRUE
  )
}

project_ensemble_map <- function(projection_id, env_stack, rf_models, maxent_models, out_file = NULL,
                                 overwrite = FALSE) {
  if (!is.null(out_file) && file.exists(out_file) && !overwrite) {
    message("Using existing projected map: ", out_file)
    return(terra::rast(out_file))
  }

  if (length(rf_models) != length(maxent_models)) {
    stop("RF and MaxEnt model lists have different lengths.", call. = FALSE)
  }

  message("Projecting ensemble suitability map: ", projection_id)
  repeat_maps <- vector("list", length(rf_models))

  for (i in seq_along(rf_models)) {
    message("  model repeat ", i, " / ", length(rf_models))
    prf <- predict_rf_raster(rf_models[[i]], env_stack)
    pme <- predict_maxent_raster(maxent_models[[i]], env_stack)
    repeat_maps[[i]] <- (prf + pme) / 2
    names(repeat_maps[[i]]) <- paste0(projection_id, "_repeat", i)
    rm(prf, pme)
    gc(FALSE)
  }

  ens <- mean(terra::rast(repeat_maps), na.rm = TRUE)
  names(ens) <- projection_id

  if (!is.null(out_file)) {
    terra::writeRaster(ens, out_file, overwrite = TRUE)
  }
  ens
}

load_or_project_sdm_maps <- function(cfg, selected_vars, rf_models, maxent_models) {
  if (isTRUE(cfg$use_precomputed_maps)) {
    files <- list.files(cfg$precomputed_sdm_map_dir, pattern = cfg$precomputed_sdm_map_pattern,
                        full.names = TRUE, ignore.case = TRUE)
    if (length(files) == 0) stop("No precomputed SDM maps found in: ", cfg$precomputed_sdm_map_dir, call. = FALSE)
    maps <- terra::rast(files)
    names(maps) <- safe_name(files)
    return(maps)
  }

  map_list <- vector("list", length(cfg$projection_dirs))
  names(map_list) <- names(cfg$projection_dirs)

  for (projection_id in names(cfg$projection_dirs)) {
    env_stack <- load_projection_stack(cfg$projection_dirs[[projection_id]], selected_vars, cfg$projection_file_pattern)
    out_file <- file.path(cfg$projected_map_dir, paste0(projection_id, ".tif"))
    map_list[[projection_id]] <- project_ensemble_map(
      projection_id = projection_id,
      env_stack = env_stack,
      rf_models = rf_models,
      maxent_models = maxent_models,
      out_file = if (cfg$save_projected_maps) out_file else NULL,
      overwrite = cfg$overwrite_projected_maps
    )
  }

  maps <- terra::rast(map_list)
  names(maps) <- names(map_list)
  maps
}

summarize_total_area_by_country <- function(sdm_tile, boundary_tile, threshold) {
  if (terra::nlyr(sdm_tile) == 0 || length(boundary_tile) == 0) return(NULL)

  r_area <- terra::cellSize(sdm_tile[[1]], unit = "km")
  names(r_area) <- "cell_area_km2"
  var_cols <- names(sdm_tile)
  area_col <- names(r_area)[1]

  thresholded <- terra::ifel(sdm_tile < threshold, 0, sdm_tile)
  r <- c(thresholded, r_area)

  e <- terra::extract(r, boundary_tile, cells = TRUE, weights = TRUE)
  if (is.null(e) || nrow(e) == 0) return(NULL)
  e$country_id <- boundary_tile[[cfg$country_id_col]][e$ID]
  e <- e[complete.cases(e[, var_cols, drop = FALSE]), , drop = FALSE]
  if (nrow(e) == 0) return(NULL)

  e$overlap_area_km2 <- e$weight * e[[area_col]]
  x <- as.matrix(e[, var_cols, drop = FALSE])
  prod <- x * e$overlap_area_km2
  prod <- cbind(prod, overlap_area_km2 = e$overlap_area_km2)
  summed <- rowsum(prod, group = e$country_id, na.rm = TRUE)

  out <- as.data.frame(summed)
  out$country_id <- rownames(out)
  rownames(out) <- NULL
  out <- tidyr::pivot_longer(out, cols = all_of(var_cols), names_to = "projection_id", values_to = "area_weighted_suitability_sum")
  out
}

summarize_habitat_classes_by_country <- function(sdm_tile, boundary_tile, threshold, high_threshold = 0.5) {
  if (terra::nlyr(sdm_tile) == 0 || length(boundary_tile) == 0) return(NULL)

  r_area <- terra::cellSize(sdm_tile[[1]], unit = "km")
  names(r_area) <- "cell_area_km2"
  area_col <- names(r_area)[1]

  out_list <- lapply(seq_len(terra::nlyr(sdm_tile)), function(k) {
    lyr <- sdm_tile[[k]]
    projection_id <- names(sdm_tile)[k]

    suitable <- terra::ifel(lyr >= threshold, 1, 0)
    low <- terra::ifel(lyr >= threshold & lyr < high_threshold, 1, 0)
    high <- terra::ifel(lyr >= high_threshold, 1, 0)
    names(suitable) <- "suitable"
    names(low) <- "low_suitability"
    names(high) <- "high_suitability"

    r <- c(suitable, low, high, r_area)
    e <- terra::extract(r, boundary_tile, cells = TRUE, weights = TRUE)
    if (is.null(e) || nrow(e) == 0) return(NULL)
    e$country_id <- boundary_tile[[cfg$country_id_col]][e$ID]
    e$overlap_area_km2 <- e$weight * e[[area_col]]

    class_cols <- c("suitable", "low_suitability", "high_suitability")
    for (cc in class_cols) {
      e[[paste0(cc, "_area_km2")]] <- e[[cc]] * e$overlap_area_km2
    }

    agg <- aggregate(
      e[, paste0(class_cols, "_area_km2"), drop = FALSE],
      by = list(country_id = e$country_id),
      FUN = sum,
      na.rm = TRUE
    )
    agg$total_overlap_area_km2 <- aggregate(e$overlap_area_km2, by = list(country_id = e$country_id), FUN = sum, na.rm = TRUE)$x
    agg$projection_id <- projection_id
    agg
  })

  do.call(rbind, out_list)
}

summarize_cropland_hazard_by_country <- function(sdm_tile, glad_tile, boundary_tile, threshold, cropland_value = 244) {
  if (terra::nlyr(sdm_tile) == 0 || length(boundary_tile) == 0) return(NULL)

  r_area <- terra::cellSize(sdm_tile[[1]], unit = "km")
  names(r_area) <- "cell_area_km2"
  var_cols <- names(sdm_tile)
  area_col <- names(r_area)[1]

  crop_binary <- terra::ifel(glad_tile == cropland_value, 1, 0)
  crop_fraction <- terra::resample(crop_binary, sdm_tile[[1]], method = "average")
  crop_area <- crop_fraction * r_area
  names(crop_area) <- "crop_area_km2"

  thresholded <- terra::ifel(sdm_tile < threshold, 0, sdm_tile)
  r <- c(thresholded, crop_area, r_area)

  e <- terra::extract(r, boundary_tile, cells = TRUE, weights = TRUE)
  if (is.null(e) || nrow(e) == 0) return(NULL)
  e$country_id <- boundary_tile[[cfg$country_id_col]][e$ID]
  e <- e[complete.cases(e[, var_cols, drop = FALSE]), , drop = FALSE]
  if (nrow(e) == 0) return(NULL)

  e$overlap_crop_area_km2 <- e$weight * e$crop_area_km2
  e$overlap_area_km2 <- e$weight * e[[area_col]]

  x <- as.matrix(e[, var_cols, drop = FALSE])
  prod <- x * e$overlap_crop_area_km2
  prod <- cbind(
    prod,
    overlap_crop_area_km2 = e$overlap_crop_area_km2,
    overlap_area_km2 = e$overlap_area_km2
  )

  summed <- rowsum(prod, group = e$country_id, na.rm = TRUE)
  out <- as.data.frame(summed)
  out$country_id <- rownames(out)
  rownames(out) <- NULL

  out <- tidyr::pivot_longer(out, cols = all_of(var_cols), names_to = "projection_id", values_to = "cropland_weighted_suitability_sum")
  out$cropland_weighted_hazard <- ifelse(
    out$overlap_crop_area_km2 > 0,
    out$cropland_weighted_suitability_sum / out$overlap_crop_area_km2,
    NA_real_
  )

  out
}

process_glad_tiles <- function(sdm_maps, country_boundary, glad_files, cfg, threshold) {
  total_area_all <- list()
  class_area_all <- list()
  crop_hazard_all <- list()

  if (!is.finite(cfg$max_glad_tiles)) {
    files_to_run <- glad_files
  } else {
    files_to_run <- glad_files[seq_len(min(length(glad_files), cfg$max_glad_tiles))]
  }

  for (i in seq_along(files_to_run)) {
    glad_file <- files_to_run[i]
    message("Processing GLAD tile ", i, " / ", length(files_to_run), ": ", basename(glad_file))
    glad_tile <- terra::rast(glad_file)

    boundary_tile <- terra::crop(country_boundary, terra::ext(glad_tile))
    if (length(boundary_tile) == 0) next

    sdm_tile <- terra::crop(sdm_maps, glad_tile)
    if (is.null(sdm_tile) || terra::ncell(sdm_tile) == 0) next

    total_area <- summarize_total_area_by_country(sdm_tile, boundary_tile, threshold)
    class_area <- summarize_habitat_classes_by_country(sdm_tile, boundary_tile, threshold, cfg$high_suitability_threshold)
    crop_hazard <- summarize_cropland_hazard_by_country(sdm_tile, glad_tile, boundary_tile, threshold, cfg$glad_cropland_value)

    if (!is.null(total_area)) {
      total_area$glad_tile <- basename(glad_file)
      total_area_all[[length(total_area_all) + 1]] <- total_area
    }
    if (!is.null(class_area)) {
      class_area$glad_tile <- basename(glad_file)
      class_area_all[[length(class_area_all) + 1]] <- class_area
    }
    if (!is.null(crop_hazard)) {
      crop_hazard$glad_tile <- basename(glad_file)
      crop_hazard_all[[length(crop_hazard_all) + 1]] <- crop_hazard
    }

    rm(glad_tile, boundary_tile, sdm_tile, total_area, class_area, crop_hazard)
    gc(FALSE)
  }

  list(
    total_area_raw = do.call(rbind, total_area_all),
    class_area_raw = do.call(rbind, class_area_all),
    cropland_hazard_raw = do.call(rbind, crop_hazard_all)
  )
}

aggregate_tile_outputs <- function(tile_outputs) {
  total_area <- tile_outputs$total_area_raw %>%
    group_by(country_id, projection_id) %>%
    summarise(
      area_weighted_suitability_sum = sum(area_weighted_suitability_sum, na.rm = TRUE),
      overlap_area_km2 = sum(overlap_area_km2, na.rm = TRUE),
      area_weighted_suitability_mean = ifelse(overlap_area_km2 > 0, area_weighted_suitability_sum / overlap_area_km2, NA_real_),
      .groups = "drop"
    )

  class_area <- tile_outputs$class_area_raw %>%
    group_by(country_id, projection_id) %>%
    summarise(
      suitable_area_km2 = sum(suitable_area_km2, na.rm = TRUE),
      low_suitability_area_km2 = sum(low_suitability_area_km2, na.rm = TRUE),
      high_suitability_area_km2 = sum(high_suitability_area_km2, na.rm = TRUE),
      total_overlap_area_km2 = sum(total_overlap_area_km2, na.rm = TRUE),
      suitable_area_percent = ifelse(total_overlap_area_km2 > 0, 100 * suitable_area_km2 / total_overlap_area_km2, NA_real_),
      low_suitability_area_percent = ifelse(total_overlap_area_km2 > 0, 100 * low_suitability_area_km2 / total_overlap_area_km2, NA_real_),
      high_suitability_area_percent = ifelse(total_overlap_area_km2 > 0, 100 * high_suitability_area_km2 / total_overlap_area_km2, NA_real_),
      .groups = "drop"
    )

  cropland_hazard <- tile_outputs$cropland_hazard_raw %>%
    group_by(country_id, projection_id) %>%
    summarise(
      cropland_weighted_suitability_sum = sum(cropland_weighted_suitability_sum, na.rm = TRUE),
      overlap_crop_area_km2 = sum(overlap_crop_area_km2, na.rm = TRUE),
      overlap_area_km2 = sum(overlap_area_km2, na.rm = TRUE),
      cropland_weighted_hazard = ifelse(overlap_crop_area_km2 > 0,
                                        cropland_weighted_suitability_sum / overlap_crop_area_km2,
                                        NA_real_),
      .groups = "drop"
    )

  list(total_area = total_area, class_area = class_area, cropland_hazard = cropland_hazard)
}

standardize_faostat <- function(faostat_file) {
  x <- read_required_csv(faostat_file)
  area_col <- find_column(x, c("Area", "Area Name", "Country", "country"), required = TRUE)
  iso_col <- find_column(x, c("ISO3 Code", "ISO3", "Area Code (ISO3)", "GID_0", "country_id"), required = FALSE)
  item_col <- find_column(x, c("Item", "Item Name", "item"), required = TRUE)
  element_col <- find_column(x, c("Element", "element"), required = TRUE)
  year_col <- find_column(x, c("Year", "year"), required = TRUE)
  unit_col <- find_column(x, c("Unit", "unit"), required = FALSE)
  value_col <- find_column(x, c("Value", "value"), required = TRUE)

  out <- data.frame(
    country_name = x[[area_col]],
    country_id = if (!is.na(iso_col)) x[[iso_col]] else x[[area_col]],
    item = x[[item_col]],
    element = x[[element_col]],
    year = as.integer(x[[year_col]]),
    unit = if (!is.na(unit_col)) x[[unit_col]] else NA_character_,
    value = suppressWarnings(as.numeric(x[[value_col]])),
    stringsAsFactors = FALSE
  )
  out
}

apply_country_crosswalk <- function(fao, crosswalk_file) {
  if (is.null(crosswalk_file) || is.na(crosswalk_file) || !nzchar(crosswalk_file)) return(fao)
  if (!file.exists(crosswalk_file)) stop("country_crosswalk_file not found: ", crosswalk_file, call. = FALSE)

  cw <- read.csv(crosswalk_file, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"country_id" %in% names(cw)) stop("Crosswalk must contain a country_id column matching GADM GID_0.", call. = FALSE)

  if ("faostat_country_id" %in% names(cw)) {
    fao <- fao %>%
      left_join(cw[, c("faostat_country_id", "country_id"), drop = FALSE],
                by = c("country_id" = "faostat_country_id"), suffix = c("", ".gadm")) %>%
      mutate(country_id = ifelse(!is.na(country_id.gadm), country_id.gadm, country_id)) %>%
      select(-country_id.gadm)
  } else if ("country_name" %in% names(cw)) {
    fao <- fao %>%
      left_join(cw[, c("country_name", "country_id"), drop = FALSE],
                by = "country_name", suffix = c("", ".gadm")) %>%
      mutate(country_id = ifelse(!is.na(country_id.gadm), country_id.gadm, country_id)) %>%
      select(-country_id.gadm)
  } else {
    stop("Crosswalk must contain either faostat_country_id or country_name.", call. = FALSE)
  }

  fao
}

calculate_host_production <- function(fao, host_crop_items, years) {
  prod <- fao %>%
    filter(year %in% years, item %in% host_crop_items, grepl("Production", element, ignore.case = TRUE)) %>%
    mutate(value_1000_t = ifelse(grepl("ton", unit, ignore.case = TRUE), value / 1000, value)) %>%
    group_by(country_id, country_name, year) %>%
    summarise(host_production_1000_t = sum(value_1000_t, na.rm = TRUE), .groups = "drop") %>%
    group_by(country_id, country_name) %>%
    summarise(
      host_production_1000_t_mean = mean(host_production_1000_t, na.rm = TRUE),
      host_production_log1p = log1p(host_production_1000_t_mean),
      .groups = "drop"
    )
  prod
}

calculate_upland_proportion <- function(fao, rice_crop_items, harvested_area_items, years) {
  area_rows <- fao %>%
    filter(year %in% years, grepl("Area harvested", element, ignore.case = TRUE))

  if (!is.null(harvested_area_items)) {
    area_rows <- area_rows %>% filter(item %in% harvested_area_items)
  }

  area_rows <- area_rows %>%
    mutate(area_ha = value)

  total_area <- area_rows %>%
    group_by(country_id, country_name, year) %>%
    summarise(total_harvested_area_ha = sum(area_ha, na.rm = TRUE), .groups = "drop")

  rice_area <- area_rows %>%
    filter(item %in% rice_crop_items | grepl("rice", item, ignore.case = TRUE)) %>%
    group_by(country_id, country_name, year) %>%
    summarise(rice_harvested_area_ha = sum(area_ha, na.rm = TRUE), .groups = "drop")

  upland <- total_area %>%
    left_join(rice_area, by = c("country_id", "country_name", "year")) %>%
    mutate(
      rice_harvested_area_ha = ifelse(is.na(rice_harvested_area_ha), 0, rice_harvested_area_ha),
      upland_proportion = ifelse(total_harvested_area_ha > 0,
                                 pmax(0, pmin(1, 1 - rice_harvested_area_ha / total_harvested_area_ha)),
                                 NA_real_)
    ) %>%
    group_by(country_id, country_name) %>%
    summarise(
      total_harvested_area_ha_mean = mean(total_harvested_area_ha, na.rm = TRUE),
      rice_harvested_area_ha_mean = mean(rice_harvested_area_ha, na.rm = TRUE),
      upland_proportion = mean(upland_proportion, na.rm = TRUE),
      .groups = "drop"
    )

  upland
}

scale_risk_score <- function(df, scaling) {
  scaling <- match.arg(scaling, c("none", "zscore", "minmax"))
  if (scaling == "none") {
    df$risk_score <- df$risk_score_raw
    return(df)
  }

  df <- df %>%
    group_by(projection_id) %>%
    mutate(
      risk_score = dplyr::case_when(
        scaling == "zscore" ~ as.numeric(scale(risk_score_raw)),
        scaling == "minmax" ~ {
          rng <- max(risk_score_raw, na.rm = TRUE) - min(risk_score_raw, na.rm = TRUE)
          ifelse(is.finite(rng) & rng > 0,
                 (risk_score_raw - min(risk_score_raw, na.rm = TRUE)) / rng,
                 NA_real_)
        },
        TRUE ~ risk_score_raw
      )
    ) %>%
    ungroup()
  df
}

################################################################################
# 3. LOAD PREVIOUS SCRIPT OUTPUTS
################################################################################

message("Loading outputs from Scripts 1 and 2...")

training_data <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$training_data_file))
rf_output <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$rf_output_file))
maxent_output <- read_required_rds(file.path(cfg$sdm_input_dir, cfg$maxent_output_file))
final_threshold <- extract_threshold(file.path(cfg$sdm_eval_dir, cfg$final_threshold_file))

selected_vars <- training_data$selected_vars
rf_models <- rf_output$models
maxent_models <- maxent_output$models

message("Final ensemble threshold: ", final_threshold)
message("Selected predictors: ", paste(selected_vars, collapse = ", "))

################################################################################
# 4. CREATE OR LOAD ENSEMBLE SUITABILITY MAPS
################################################################################

sdm_maps <- load_or_project_sdm_maps(cfg, selected_vars, rf_models, maxent_models)

# Add projection metadata-compatible names when precomputed maps have matching names.
projection_ids <- names(sdm_maps)
metadata <- cfg$projection_metadata %>%
  filter(projection_id %in% projection_ids)

################################################################################
# 5. COUNTRY-LEVEL HABITAT AREA AND CROPLAND-WEIGHTED HAZARD
################################################################################

message("Loading country boundaries and GLAD tiles...")
country_boundary <- terra::vect(cfg$gadm_file, layer = cfg$gadm_layer)
if (!cfg$country_id_col %in% names(country_boundary)) {
  stop("country_id_col not found in GADM boundary: ", cfg$country_id_col, call. = FALSE)
}

country_lookup <- as.data.frame(country_boundary)[, intersect(c(cfg$country_id_col, cfg$country_name_col), names(country_boundary)), drop = FALSE]
names(country_lookup)[names(country_lookup) == cfg$country_id_col] <- "country_id"
if (cfg$country_name_col %in% names(country_boundary)) {
  names(country_lookup)[names(country_lookup) == cfg$country_name_col] <- "country_name_gadm"
}
country_lookup <- unique(country_lookup)

glad_files <- sort(list.files(cfg$glad_dir, pattern = cfg$glad_file_pattern, full.names = TRUE, ignore.case = TRUE))
if (length(glad_files) == 0) stop("No GLAD .tif files found in: ", cfg$glad_dir, call. = FALSE)

message("Calculating country-level habitat area and cropland-weighted hazard...")
tile_outputs <- process_glad_tiles(sdm_maps, country_boundary, glad_files, cfg, final_threshold)
aggregated <- aggregate_tile_outputs(tile_outputs)

write.csv(tile_outputs$total_area_raw, file.path(cfg$output_dir, "country_total_area_weighted_suitability_by_tile.csv"), row.names = FALSE)
write.csv(tile_outputs$class_area_raw, file.path(cfg$output_dir, "country_habitat_class_area_by_tile.csv"), row.names = FALSE)
write.csv(tile_outputs$cropland_hazard_raw, file.path(cfg$output_dir, "country_cropland_hazard_by_tile.csv"), row.names = FALSE)

country_total_area <- aggregated$total_area %>% left_join(metadata, by = "projection_id")
country_class_area <- aggregated$class_area %>% left_join(metadata, by = "projection_id")
country_cropland_hazard <- aggregated$cropland_hazard %>% left_join(metadata, by = "projection_id")

write.csv(country_total_area, file.path(cfg$output_dir, "country_total_area_weighted_suitability.csv"), row.names = FALSE)
write.csv(country_class_area, file.path(cfg$output_dir, "country_habitat_class_area.csv"), row.names = FALSE)
write.csv(country_cropland_hazard, file.path(cfg$output_dir, "country_cropland_weighted_hazard.csv"), row.names = FALSE)

################################################################################
# 6. FAOSTAT PRODUCTION AND UPLAND-CROPLAND COMPONENTS
################################################################################

message("Calculating FAOSTAT-based crop production and upland-cropland components...")
fao <- standardize_faostat(cfg$faostat_crop_file)
fao <- apply_country_crosswalk(fao, cfg$country_crosswalk_file)
host_production <- calculate_host_production(fao, cfg$host_crop_items, cfg$production_years)
upland_proportion <- calculate_upland_proportion(fao, cfg$rice_crop_items, cfg$harvested_area_items, cfg$production_years)

write.csv(host_production, file.path(cfg$output_dir, "country_host_crop_production_component.csv"), row.names = FALSE)
write.csv(upland_proportion, file.path(cfg$output_dir, "country_upland_cropland_component.csv"), row.names = FALSE)

################################################################################
# 7. COUNTRY-LEVEL AGRICULTURAL RISK INDEX
################################################################################

message("Calculating country-level agricultural risk index...")

risk_index <- country_cropland_hazard %>%
  left_join(country_class_area, by = c("country_id", "projection_id", "period", "scenario")) %>%
  left_join(country_lookup, by = "country_id") %>%
  left_join(host_production, by = "country_id", suffix = c("", ".host")) %>%
  left_join(upland_proportion, by = "country_id", suffix = c("", ".upland")) %>%
  mutate(
    host_production_1000_t_mean = ifelse(is.na(host_production_1000_t_mean), 0, host_production_1000_t_mean),
    host_production_log1p = ifelse(is.na(host_production_log1p), 0, host_production_log1p),
    upland_proportion = ifelse(is.na(upland_proportion), NA_real_, upland_proportion),
    risk_score_raw = host_production_log1p * cropland_weighted_hazard * upland_proportion
  )

risk_index <- scale_risk_score(risk_index, cfg$risk_score_scaling)

risk_index <- risk_index %>%
  arrange(projection_id, desc(risk_score))

write.csv(risk_index, file.path(cfg$output_dir, "country_level_agricultural_risk_index.csv"), row.names = FALSE)

################################################################################
# 8. OPTIONAL CROP-SPECIFIC RISK INDEX
################################################################################

message("Calculating crop-specific risk index...")

crop_specific_production <- fao %>%
  filter(year %in% cfg$production_years, item %in% cfg$host_crop_items, grepl("Production", element, ignore.case = TRUE)) %>%
  mutate(value_1000_t = ifelse(grepl("ton", unit, ignore.case = TRUE), value / 1000, value)) %>%
  group_by(country_id, country_name, item, year) %>%
  summarise(crop_production_1000_t = sum(value_1000_t, na.rm = TRUE), .groups = "drop") %>%
  group_by(country_id, country_name, item) %>%
  summarise(
    crop_production_1000_t_mean = mean(crop_production_1000_t, na.rm = TRUE),
    crop_production_log1p = log1p(crop_production_1000_t_mean),
    .groups = "drop"
  )

crop_specific_risk <- country_cropland_hazard %>%
  left_join(country_class_area, by = c("country_id", "projection_id", "period", "scenario")) %>%
  left_join(country_lookup, by = "country_id") %>%
  left_join(crop_specific_production, by = "country_id", suffix = c("", ".crop")) %>%
  left_join(upland_proportion, by = "country_id", suffix = c("", ".upland")) %>%
  mutate(
    crop_production_1000_t_mean = ifelse(is.na(crop_production_1000_t_mean), 0, crop_production_1000_t_mean),
    crop_production_log1p = ifelse(is.na(crop_production_log1p), 0, crop_production_log1p),
    crop_risk_score_raw = crop_production_log1p * cropland_weighted_hazard * upland_proportion
  ) %>%
  arrange(projection_id, item, desc(crop_risk_score_raw))

write.csv(crop_specific_risk, file.path(cfg$output_dir, "country_level_crop_specific_risk_index.csv"), row.names = FALSE)

################################################################################
# 9. SAVE COMPACT RDS OUTPUT AND RUN SUMMARY
################################################################################

saveRDS(
  list(
    final_threshold = final_threshold,
    selected_vars = selected_vars,
    country_total_area = country_total_area,
    country_class_area = country_class_area,
    country_cropland_hazard = country_cropland_hazard,
    host_production = host_production,
    upland_proportion = upland_proportion,
    risk_index = risk_index,
    crop_specific_risk = crop_specific_risk,
    config = cfg
  ),
  file.path(cfg$output_dir, "country_level_agricultural_risk_outputs.rds")
)

run_summary <- data.frame(
  item = c(
    "final_threshold",
    "n_projection_maps",
    "n_glad_tiles",
    "n_countries_in_risk_table",
    "risk_score_scaling",
    "output_dir"
  ),
  value = c(
    final_threshold,
    terra::nlyr(sdm_maps),
    length(glad_files),
    length(unique(risk_index$country_id)),
    cfg$risk_score_scaling,
    normalizePath(cfg$output_dir, mustWork = FALSE)
  )
)
write.csv(run_summary, file.path(cfg$output_dir, "run_summary.csv"), row.names = FALSE)

message("Completed country-level agricultural risk analysis. Outputs saved in: ",
        normalizePath(cfg$output_dir, mustWork = FALSE))

################################################################################
# End of script
################################################################################
