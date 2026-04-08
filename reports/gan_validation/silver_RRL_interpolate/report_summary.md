# GAN Validity Report: silver_RRL_interpolate

## Summary
- dataset: silver_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2026-05-08
- future_rows: 267
- expected_future_rows: 267
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.001878
- max_vol_ratio_gap: 0.001879
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.066454
- max_ks_statistic: 0.079502
- avg_acf_gap_lag1: 0.134727
- max_acf_gap_lag1: 0.213906
- corr_matrix_mae: 0.130957
- corr_matrix_max_gap: 0.470776
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 1.0
- share_features_acf_gap_lag1_lt_0p15: 0.5

## Highest KS Distance Features
- Silver_Futures: ks_statistic=0.0795, ks_pvalue=0.0889, vol_ratio=1.0019, acf_gap_1=0.0811
- US30: ks_statistic=0.0787, ks_pvalue=0.0944, vol_ratio=1.0019, acf_gap_1=0.2139
- SnP500: ks_statistic=0.0756, ks_pvalue=0.1192, vol_ratio=1.0019, acf_gap_1=0.1749
- Gold_Futures: ks_statistic=0.0639, ks_pvalue=0.2641, vol_ratio=1.0019, acf_gap_1=0.0661
- NASDAQ_100: ks_statistic=0.0614, ks_pvalue=0.3082, vol_ratio=1.0019, acf_gap_1=0.1546
