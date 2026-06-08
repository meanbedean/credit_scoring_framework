suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

ks_static_deciles <- function(df,
                              score_col   = "combined_risk",
                              target_col  = "downgrade_target",
                              segment_col = "dev_oot_flag",
                              train_label = "train",
                              n_bins      = 10) {

  # 1. BREAKS FROM TRAIN
  train_scores <- df %>%
    dplyr::filter(.data[[segment_col]] == train_label) %>%
    dplyr::pull(.data[[score_col]])

  if (length(train_scores) == 0) {
    stop("No rows found for train_label = '", train_label, "'.")
  }

  breaks <- quantile(train_scores,
                     probs = seq(0, 1, length.out = n_bins + 1),
                     na.rm = TRUE) %>% unname()
  breaks[1]              <- -Inf
  breaks[length(breaks)] <-  Inf

  # 2. Build a lookup of "decile -> TRAIN range" (the canonical bin definition)
  train_range <- tibble::tibble(
    decile      = 1:n_bins,
    lower_break = breaks[1:n_bins],
    upper_break = breaks[2:(n_bins + 1)]
  ) %>%
    dplyr::mutate(
      train_range = sprintf("(%s, %s]",
                            ifelse(is.infinite(lower_break), "-Inf",
                                   formatC(lower_break, format = "f", digits = 4)),
                            ifelse(is.infinite(upper_break), "Inf",
                                   formatC(upper_break, format = "f", digits = 4)))
    )

  cat("Decile breaks built from TRAIN scores:\n")
  print(train_range)

  # 3. APPLY BREAKS
  df <- df %>%
    dplyr::mutate(
      decile = cut(.data[[score_col]],
                   breaks         = breaks,
                   labels         = 1:n_bins,
                   include.lowest = TRUE,
                   right          = TRUE) %>% as.integer()
    )

  # 4. AGGREGATE — now also capture the ACTUAL min/max observed in each
  #    (decile x segment) cell so you can see TRAIN's canonical bin AND
  #    how each other segment's scores spread within those same bins
  agg <- df %>%
    dplyr::group_by(.data[[segment_col]], decile) %>%
    dplyr::summarise(
      n                = dplyr::n(),
      events           = sum(.data[[target_col]] == 1, na.rm = TRUE),
      nonevents        = sum(.data[[target_col]] == 0, na.rm = TRUE),
      bad_rate         = mean(.data[[target_col]], na.rm = TRUE),
      score_min        = min(.data[[score_col]], na.rm = TRUE),
      score_max        = max(.data[[score_col]], na.rm = TRUE),
      score_avg        = mean(.data[[score_col]], na.rm = TRUE),
      .groups          = "drop"
    ) %>%
    dplyr::arrange(.data[[segment_col]], dplyr::desc(decile))

  # 5. CUMULATIVE % AND KS
  agg <- agg %>%
    dplyr::group_by(.data[[segment_col]]) %>%
    dplyr::mutate(
      cum_events       = cumsum(events),
      cum_nonevents    = cumsum(nonevents),
      cum_event_pct    = 100 * cum_events    / sum(events),
      cum_nonevent_pct = 100 * cum_nonevents / sum(nonevents),
      ks               = abs(cum_event_pct - cum_nonevent_pct),
      pop_pct          = 100 * n / sum(n)
    ) %>%
    dplyr::ungroup()

  # 6. ATTACH TRAIN BIN RANGE + OBSERVED RANGE FOR EACH ROW
  agg <- agg %>%
    dplyr::left_join(train_range %>% dplyr::select(decile, train_range),
                     by = "decile") %>%
    dplyr::mutate(
      observed_range = sprintf("[%s, %s]",
                               formatC(score_min, format = "f", digits = 4),
                               formatC(score_max, format = "f", digits = 4))
    ) %>%
    # Reorder columns so the ranges are easy to read
    dplyr::select(dplyr::all_of(segment_col), decile,
                  train_range, observed_range,
                  n, pop_pct, events, nonevents, bad_rate,
                  score_min, score_max, score_avg,
                  cum_event_pct, cum_nonevent_pct, ks)

  # 7. SUMMARY
  summary_ks <- agg %>%
    dplyr::group_by(.data[[segment_col]]) %>%
    dplyr::summarise(
      ks_max          = max(ks),
      ks_peak_decile  = decile[which.max(ks)],
      total_events    = sum(events),
      total_n         = sum(n),
      event_rate      = total_events / total_n,
      .groups         = "drop"
    )

  list(detail = agg, summary = summary_ks, train_range = train_range,
       breaks = breaks)
}


# ---------- Usage ----------
res <- ks_static_deciles(
  df          = scored_card,
  score_col   = "combined_risk",
  target_col  = "downgrade_target",
  segment_col = "dev_oot_flag",
  train_label = "train",
  n_bins      = 10
)

print(res$summary)
print(res$detail, n = 40, width = Inf)
print(res$train_range)
