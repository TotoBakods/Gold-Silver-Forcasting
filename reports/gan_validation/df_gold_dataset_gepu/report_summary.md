# GAN Validity Report: df_gold_dataset_gepu

## Summary
- dataset: df_gold_dataset_gepu
- source_last_date: 2025-11-26
- generated_last_date: 2026-05-08
- future_rows: 117
- expected_future_rows: 117
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 0.996774
- max_vol_ratio_gap: 0.047196
- avg_mean_gap_abs: 0.129501
- avg_ks_statistic: 0.104654
- max_ks_statistic: 0.164163
- avg_acf_gap_lag1: 0.074833
- max_acf_gap_lag1: 0.221926
- corr_matrix_mae: 0.051665
- corr_matrix_max_gap: 0.152604
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.571429
- share_features_acf_gap_lag1_lt_0p15: 0.857143

## Highest KS Distance Features
- gepu: ks_statistic=0.1642, ks_pvalue=0.0041, vol_ratio=0.9528, acf_gap_1=0.0176
- Silver_Futures: ks_statistic=0.1391, ks_pvalue=0.0233, vol_ratio=1.0043, acf_gap_1=0.2219
- Gold_Futures: ks_statistic=0.1338, ks_pvalue=0.0325, vol_ratio=1.0043, acf_gap_1=0.1163
- Crude_Oil_Futures: ks_statistic=0.1112, ks_pvalue=0.1153, vol_ratio=1.0043, acf_gap_1=0.0382
- UST10Y_Treasury_Yield: ks_statistic=0.0652, ks_pvalue=0.7002, vol_ratio=1.0043, acf_gap_1=0.0699
