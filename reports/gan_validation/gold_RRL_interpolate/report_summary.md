# GAN Validity Report: gold_RRL_interpolate

## Summary
- dataset: gold_RRL_interpolate
- source_last_date: 2025-04-30
- generated_last_date: 2026-05-08
- future_rows: 267
- expected_future_rows: 267
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.001878
- max_vol_ratio_gap: 0.001878
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.205009
- max_ks_statistic: 0.43138
- avg_acf_gap_lag1: 0.152296
- max_acf_gap_lag1: 0.508316
- corr_matrix_mae: 0.290898
- corr_matrix_max_gap: 0.741295
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.0
- share_features_acf_gap_lag1_lt_0p15: 0.75

## Highest KS Distance Features
- Federal_Funds_Rate: ks_statistic=0.4314, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.0821
- Employment_Pop_Ratio: ks_statistic=0.4169, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.5083
- gepu: ks_statistic=0.2439, ks_pvalue=0.0000, vol_ratio=1.0019, acf_gap_1=0.2451
- Gold_Futures: ks_statistic=0.1242, ks_pvalue=0.0010, vol_ratio=1.0019, acf_gap_1=0.0438
- Silver_Futures: ks_statistic=0.1199, ks_pvalue=0.0017, vol_ratio=1.0019, acf_gap_1=0.0629
