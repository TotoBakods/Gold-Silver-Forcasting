# GAN Validity Report: silver_RRL_interpolate

## Summary
- dataset: silver_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2027-12-31
- future_rows: 697
- expected_future_rows: 697
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.000717
- max_vol_ratio_gap: 0.000718
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.191484
- max_ks_statistic: 0.21489
- avg_acf_gap_lag1: 0.095161
- max_acf_gap_lag1: 0.133716
- corr_matrix_mae: 1e-06
- corr_matrix_max_gap: 6e-06
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.0
- share_features_acf_gap_lag1_lt_0p15: 1.0

## Highest KS Distance Features
- Silver_Futures: ks_statistic=0.2149, ks_pvalue=0.0000, vol_ratio=1.0007, acf_gap_1=0.0912
- US30: ks_statistic=0.1995, ks_pvalue=0.0000, vol_ratio=1.0007, acf_gap_1=0.1337
- Gold_Futures: ks_statistic=0.1969, ks_pvalue=0.0000, vol_ratio=1.0007, acf_gap_1=0.0760
- SnP500: ks_statistic=0.1968, ks_pvalue=0.0000, vol_ratio=1.0007, acf_gap_1=0.1173
- NASDAQ_100: ks_statistic=0.1752, ks_pvalue=0.0000, vol_ratio=1.0007, acf_gap_1=0.0646
