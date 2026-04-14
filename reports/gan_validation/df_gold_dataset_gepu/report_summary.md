# GAN Validity Report: df_gold_dataset_gepu

## Summary
- dataset: df_gold_dataset_gepu
- source_last_date: 2025-11-26
- generated_last_date: 2027-12-31
- future_rows: 547
- expected_future_rows: 547
- duplicate_dates: 0
- null_cells: 0
- avg_vol_ratio: 1.000914
- max_vol_ratio_gap: 0.000915
- avg_mean_gap_abs: 0.0
- avg_ks_statistic: 0.22854
- max_ks_statistic: 0.524989
- avg_acf_gap_lag1: 0.077959
- max_acf_gap_lag1: 0.227423
- corr_matrix_mae: 1e-06
- corr_matrix_max_gap: 1.1e-05
- share_features_vol_ratio_pass_0p8_to_1p2: 1.0
- share_features_ks_pvalue_gt_0p05: 0.142857
- share_features_acf_gap_lag1_lt_0p15: 0.857143

## Highest KS Distance Features
- DFF: ks_statistic=0.5250, ks_pvalue=0.0000, vol_ratio=1.0009, acf_gap_1=0.0231
- Gold_Futures: ks_statistic=0.3020, ks_pvalue=0.0000, vol_ratio=1.0009, acf_gap_1=0.0529
- Silver_Futures: ks_statistic=0.2828, ks_pvalue=0.0000, vol_ratio=1.0009, acf_gap_1=0.0625
- Crude_Oil_Futures: ks_statistic=0.2011, ks_pvalue=0.0000, vol_ratio=1.0009, acf_gap_1=0.1306
- UST10Y_Treasury_Yield: ks_statistic=0.1714, ks_pvalue=0.0000, vol_ratio=1.0009, acf_gap_1=0.2274
