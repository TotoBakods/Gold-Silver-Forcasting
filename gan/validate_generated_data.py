import os
import sys
import pandas as pd
import numpy as np

from dataset_catalog import (
    TARGET_END_DATE,
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


def stationary_series(series, is_price):
    if is_price:
        return np.log(series / series.shift(1)).dropna()
    return series.diff().dropna()


def validate_file(dataset_config):
    source_path, _ = ensure_prepared_source(dataset_config)
    extended_path = get_extended_output_path(dataset_config)

    if not extended_path.exists():
        return {
            "file": get_output_base_name(dataset_config),
            "status": "missing",
            "errors": [f"Extended file not found: {extended_path.name}"],
            "checks": [],
        }

    source_df = pd.read_csv(source_path)
    extended_df = pd.read_csv(extended_path)

    for df in (source_df, extended_df):
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    price_cols, other_cols = classify_columns(source_df)
    all_value_cols = price_cols + other_cols
    future_df = extended_df[extended_df["Date"] > source_df["Date"].max()].copy()

    checks = []
    errors = []

    expected_future_dates = pd.date_range(
        start=source_df["Date"].max() + pd.Timedelta(days=1),
        end=TARGET_END_DATE,
        freq="B",
    )

    checks.append(("source_last_date", str(source_df["Date"].max().date())))
    checks.append(("extended_last_date", str(extended_df["Date"].max().date())))
    checks.append(("future_rows", int(len(future_df))))
    checks.append(("expected_future_rows", int(len(expected_future_dates))))
    checks.append(("duplicate_dates", int(extended_df["Date"].duplicated().sum())))
    checks.append(("null_cells", int(extended_df.isna().sum().sum())))

    if extended_df["Date"].max() != TARGET_END_DATE:
        errors.append(f"Extended file ends at {extended_df['Date'].max().date()}, not {TARGET_END_DATE.date()}")

    if len(future_df) != len(expected_future_dates):
        errors.append(
            f"Generated {len(future_df)} future rows, expected {len(expected_future_dates)} business-day rows"
        )

    if not extended_df["Date"].is_monotonic_increasing:
        errors.append("Dates are not strictly increasing")

    if extended_df["Date"].duplicated().any():
        errors.append("Duplicate dates found")

    if extended_df.isna().any().any():
        errors.append("Null values found")

    if not future_df.empty and list(future_df["Date"]) != list(expected_future_dates):
        errors.append("Future dates do not match the expected business-day calendar")

    nonpositive_price_cols = [col for col in price_cols if (extended_df[col] <= 0).any()]
    if nonpositive_price_cols:
        errors.append(f"Non-positive price values found in: {', '.join(nonpositive_price_cols)}")

    for col in all_value_cols:
        hist = source_df[col].astype(float)
        gen = future_df[col].astype(float)
        if gen.empty:
            errors.append(f"No generated rows available for {col}")
            continue

        hist_stat = stationary_series(hist, col in price_cols)
        gen_stat = stationary_series(gen, col in price_cols)

        if gen_stat.empty:
            errors.append(f"Not enough generated rows to compute stationary stats for {col}")
            continue

        hist_std = float(hist_stat.std())
        gen_std = float(gen_stat.std())
        hist_mean = float(hist_stat.mean())
        gen_mean = float(gen_stat.mean())
        vol_ratio = gen_std / (hist_std + 1e-12)
        mean_gap = abs(gen_mean - hist_mean)

        checks.append((f"{col}_vol_ratio", round(vol_ratio, 4)))
        checks.append((f"{col}_mean_gap", round(mean_gap, 6)))

        if hist_std > 0 and not 0.5 <= vol_ratio <= 1.5:
            errors.append(
                f"{col} generated volatility ratio {vol_ratio:.3f} is outside the acceptable range [0.5, 1.5]"
            )

        if mean_gap > max(3 * hist_std, 1e-6):
            errors.append(
                f"{col} generated mean shift {mean_gap:.6f} is too large relative to historical volatility {hist_std:.6f}"
            )

    return {
        "file": get_output_base_name(dataset_config),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "checks": checks,
    }


def main():
    overall_ok = True
    for dataset_config in get_enabled_dataset_configs():
        result = validate_file(dataset_config)
        print(f"\n[{result['file']}] {result['status'].upper()}")
        for name, value in result["checks"]:
            print(f"  {name}: {value}")
        if result["errors"]:
            overall_ok = False
            print("  errors:")
            for error in result["errors"]:
                print(f"    - {error}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
