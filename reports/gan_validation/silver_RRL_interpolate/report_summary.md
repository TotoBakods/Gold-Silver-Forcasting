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
- max_vol_ratio_gap: 0.001878
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.056838
- max_ks_statistic: 0.075414
- avg_acf_gap_lag1: 0.058316
- max_acf_gap_lag1: 0.162952
- corr_matrix_mae: 0.240753
- corr_matrix_max_gap: 0.571587
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 1.0
- share_features_acf_gap_lag1_lt_0p15: 0.833333

## Highest KS Distance Features
- US30: ks_statistic=0.0754, ks_pvalue=0.1211, vol_ratio=1.0019, acf_gap_1=0.0198
- SnP500: ks_statistic=0.0673, ks_pvalue=0.2132, vol_ratio=1.0019, acf_gap_1=0.0073
- Gold_Futures: ks_statistic=0.0603, ks_pvalue=0.3292, vol_ratio=1.0019, acf_gap_1=0.0918
- Silver_Futures: ks_statistic=0.0542, ks_pvalue=0.4601, vol_ratio=1.0019, acf_gap_1=0.0308
- NASDAQ_100: ks_statistic=0.0497, ks_pvalue=0.5704, vol_ratio=1.0019, acf_gap_1=0.0373
