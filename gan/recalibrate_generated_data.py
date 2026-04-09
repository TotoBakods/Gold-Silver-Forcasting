import os
import pandas as pd
import numpy as np

from dataset_catalog import (
    ensure_prepared_source,
    get_enabled_dataset_configs,
    get_extended_output_path,
    get_output_base_name,
)


def classify_columns(df):
    price_keywords = ["Futures", "US30", "SnP500", "NASDAQ_100"]
    price_cols = [c for c in df.columns if c != "Date" and any(k in c for k in price_keywords)]
    other_cols = [c for c in df.columns if c not in price_cols and c != "Date"]
    return price_cols, other_cols


def to_stationary(df, price_cols, other_cols):
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in other_cols:
        stat_df[col] = df[col].diff()
    return stat_df.dropna().reset_index(drop=True)


def calibrate_generated_stationary(gen_stat, hist_stat_df, feature_names):
    calibrated = gen_stat.copy()
    hist_means = hist_stat_df[feature_names].mean().to_numpy()
    hist_stds = hist_stat_df[feature_names].std().to_numpy()
    gen_means = calibrated.mean(axis=0)
    gen_stds = calibrated.std(axis=0)

    for idx in range(len(feature_names)):
        hist_mean = hist_means[idx]
        hist_std = hist_stds[idx]
        gen_mean = gen_means[idx]
        gen_std = gen_stds[idx]

        if hist_std <= 1e-12:
            calibrated[:, idx] = hist_mean
            continue

        normalized = calibrated[:, idx] - gen_mean
        if gen_std > 1e-12:
            normalized = normalized / gen_std
        else:
            normalized = np.zeros_like(normalized)

        calibrated[:, idx] = normalized * hist_std + hist_mean

    return calibrated


def reconstruct_future(source_df, future_df, feature_names, price_cols):
    current_vals = source_df[feature_names].iloc[-1].astype(float).to_numpy(copy=True)
    rows = []
    for _, row in future_df.iterrows():
        stat_values = row[feature_names].astype(float).to_numpy()
        for idx, col in enumerate(feature_names):
            if col in price_cols:
                current_vals[idx] = current_vals[idx] * np.exp(stat_values[idx])
            else:
                current_vals[idx] = current_vals[idx] + stat_values[idx]
        rows.append(current_vals.copy())
    rebuilt = pd.DataFrame(rows, columns=feature_names)
    rebuilt.insert(0, "Date", future_df["Date"].to_list())
    return rebuilt


def recalibrate_file(dataset_config):
    source_path, _ = ensure_prepared_source(dataset_config)
    extended_path = get_extended_output_path(dataset_config)

    if not extended_path.exists():
        print(f"Skipping {get_output_base_name(dataset_config)}: missing {extended_path.name}")
        return

    source_df = pd.read_csv(source_path)
    extended_df = pd.read_csv(extended_path)
    source_df["Date"] = pd.to_datetime(source_df["Date"])
    extended_df["Date"] = pd.to_datetime(extended_df["Date"])

    price_cols, other_cols = classify_columns(source_df)
    feature_names = price_cols + other_cols
    source_df = source_df.sort_values("Date").reset_index(drop=True)
    extended_df = extended_df.sort_values("Date").reset_index(drop=True)
    future_df = extended_df[extended_df["Date"] > source_df["Date"].max()].copy()

    if future_df.empty:
        print(f"Skipping {get_output_base_name(dataset_config)}: no generated rows to recalibrate")
        return

    hist_stat_df = to_stationary(source_df, price_cols, other_cols)
    bridge_df = pd.concat([source_df.tail(1), future_df], ignore_index=True)
    future_stat_df = to_stationary(bridge_df, price_cols, other_cols)

    if future_stat_df.empty:
        print(f"Skipping {get_output_base_name(dataset_config)}: not enough future rows to recalibrate")
        return

    calibrated_stat = calibrate_generated_stationary(
        future_stat_df[feature_names].to_numpy(),
        hist_stat_df,
        feature_names,
    )
    calibrated_future_stat_df = pd.DataFrame(calibrated_stat, columns=feature_names)
    calibrated_future_stat_df.insert(0, "Date", future_stat_df["Date"].to_list())
    rebuilt_future = reconstruct_future(
        source_df,
        calibrated_future_stat_df,
        feature_names,
        price_cols,
    )

    final_df = pd.concat([source_df, rebuilt_future], ignore_index=True)
    final_df.to_csv(extended_path, index=False)
    print(f"Recalibrated {extended_path.name}")


def main():
    for dataset_config in get_enabled_dataset_configs():
        recalibrate_file(dataset_config)


if __name__ == "__main__":
    main()
