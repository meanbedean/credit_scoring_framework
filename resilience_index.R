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

  # 1. BUILD DECILE BREAKS ON TRAIN ONLY  (static cuts)
  train_scores <- df %>%
    dplyr::filter(.data[[segment_col]] == train_label) %>%
    dplyr::pull(.data[[score_col]])

  if (length(train_scores) == 0) {
    stop("No rows found for train_label = '", train_label, "' in column '", segment_col, "'.")
  }

  breaks <- quantile(train_scores,
                     probs = seq(0, 1, length.out = n_bins + 1),
                     na.rm = TRUE) %>% unname()

  # Open the tails so test / oot scores outside the train range still bin cleanly
  breaks[1]               <- -Inf
  breaks[length(breaks)]  <-  Inf

  cat("Decile breaks built from TRAIN scores (static cuts applied to all segments):\n")
  print(setNames(breaks, paste0("p", seq(0, 100, by = 10))))

  # 2. APPLY THE SAME BREAKS TO EVERY SEGMENT
  df <- df %>%
    dplyr::mutate(
      decile = cut(.data[[score_col]],
                   breaks         = breaks,
                   labels         = 1:n_bins,
                   include.lowest = TRUE,
                   right          = TRUE) %>% as.integer()
    )

  # 3. PER-SEGMENT, PER-DECILE STATS
  agg <- df %>%
    dplyr::group_by(.data[[segment_col]], decile) %>%
    dplyr::summarise(
      n              = dplyr::n(),
      events         = sum(.data[[target_col]] == 1, na.rm = TRUE),
      nonevents      = sum(.data[[target_col]] == 0, na.rm = TRUE),
      bad_rate       = mean(.data[[target_col]], na.rm = TRUE),
      avg_score      = mean(.data[[score_col]], na.rm = TRUE),
      .groups        = "drop"
    ) %>%
    dplyr::arrange(.data[[segment_col]], dplyr::desc(decile))   # 10 first = highest risk

  # 4. CUMULATIVE PERCENTS + KS, COMPUTED PER SEGMENT
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

  # 5. SUMMARY (one row per segment)
  summary_ks <- agg %>%
    dplyr::group_by(.data[[segment_col]]) %>%
    dplyr::summarise(
      ks_max         = max(ks),
      ks_peak_decile = decile[which.max(ks)],
      total_events   = sum(events),
      total_n        = sum(n),
      event_rate     = total_events / total_n,
      .groups        = "drop"
    )

  list(detail = agg, summary = summary_ks, breaks = breaks)
}


# ---------- Usage ----------
res <- ks_static_deciles(
  df          = scored_card,
  score_col   = "combined_risk",        # or risk_index_baseline / resilience_index_baseline
  target_col  = "downgrade_target",
  segment_col = "dev_oot_flag",
  train_label = "train",                # change if your train flag is "TRAIN" or "Train"
  n_bins      = 10
)

print(res$summary)
print(res$detail, n = 40)

# KS curve across deciles, one line per segment
ggplot(res$detail, aes(x = decile, y = ks, colour = dev_oot_flag)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_reverse(breaks = 1:10) +
  labs(title = "KS by decile (static deciles from TRAIN)",
       x = "Decile (10 = highest combined_risk)",
       y = "KS") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
