import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parent

GOLD_INPUT = ROOT / "df_gold_dataset_USA_EPU_APRIL_01_2015_to_APRIL_14_2026.csv"
SILVER_INPUT = ROOT / "df_silver_dataset_APRIL_01_2015_to_APRIL_14_2026.csv"
SILVER_TEMPLATE = ROOT / "silver_RRL_interpolate.csv"

GOLD_PREPARED = ROOT / "df_gold_dataset_usa_epu_prepared.csv"
GOLD_TRAIN = ROOT / "df_gold_dataset_usa_epu_train_2015_2026.csv"
GOLD_TEST = ROOT / "df_gold_dataset_usa_epu_test_2026_04_01_2026_04_13.csv"

SILVER_PREPARED = ROOT / "df_silver_dataset_aligned_prepared.csv"
SILVER_TRAIN = ROOT / "df_silver_dataset_train_2015_2026.csv"
SILVER_TEST = ROOT / "df_silver_dataset_test_2026_04_01_2026_04_13.csv"

TRAIN_START = pd.Timestamp("2015-04-01")
TRAIN_END = pd.Timestamp("2026-03-31")
TEST_START = pd.Timestamp("2026-04-01")
TEST_END = pd.Timestamp("2026-04-13")


def _last_valid_value(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def _normalize_dates(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _split_by_date(df, train_start, train_end, test_start, test_end):
    train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)
    test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def prepare_gold():
    if not GOLD_INPUT.exists():
        raise FileNotFoundError(f"Missing gold input: {GOLD_INPUT}")

    df = pd.read_csv(GOLD_INPUT)
    df = _normalize_dates(df)

    if "USA_EPU" in df.columns and "gepu" not in df.columns:
        df = df.rename(columns={"USA_EPU": "gepu"})

    value_cols = [col for col in df.columns if col != "Date"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target_col = "Gold_Futures"
    weekend_mask = df[target_col].notna() & (df["Date"].dt.dayofweek >= 5)
    if weekend_mask.any():
        df.loc[weekend_mask, "Date"] = df.loc[weekend_mask, "Date"].map(
            lambda dt: (dt - BDay(1)).normalize()
        )

    collapsed = df.groupby("Date", as_index=False).agg({col: _last_valid_value for col in value_cols})

    valid_target_dates = collapsed.loc[collapsed[target_col].notna(), "Date"]
    if valid_target_dates.empty:
        raise ValueError("Gold input does not contain any non-null Gold_Futures values.")

    first_target_date = valid_target_dates.min()
    last_target_date = valid_target_dates.max()
    business_index = pd.date_range(first_target_date, last_target_date, freq="B")

    aligned = pd.DataFrame(index=business_index)
    aligned.index.name = "Date"
    aligned = aligned.join(collapsed.set_index("Date")[value_cols], how="left")

    filled = aligned.interpolate(method="time", limit_direction="both").ffill().bfill()
    filled = filled.reset_index()

    filled.to_csv(GOLD_PREPARED, index=False)

    train_df, test_df = _split_by_date(filled, TRAIN_START, TRAIN_END, TEST_START, TEST_END)
    train_df.to_csv(GOLD_TRAIN, index=False)
    test_df.to_csv(GOLD_TEST, index=False)

    print(f"Gold prepared rows: {len(filled)}")
    print(f"Gold train rows: {len(train_df)} -> {GOLD_TRAIN.name}")
    print(f"Gold test rows: {len(test_df)} -> {GOLD_TEST.name}")


def prepare_silver():
    if not SILVER_INPUT.exists():
        raise FileNotFoundError(f"Missing silver input: {SILVER_INPUT}")
    if not SILVER_TEMPLATE.exists():
        raise FileNotFoundError(f"Missing silver template: {SILVER_TEMPLATE}")

    template = pd.read_csv(SILVER_TEMPLATE)
    template = _normalize_dates(template)
    template_dates = pd.DatetimeIndex(template["Date"].drop_duplicates())

    raw = pd.read_csv(SILVER_INPUT)
    raw = _normalize_dates(raw)

    value_cols = [col for col in raw.columns if col != "Date"]
    for col in value_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.groupby("Date", as_index=False).agg({col: _last_valid_value for col in value_cols})

    calendar_dates = template_dates
    template_end = template_dates.max()
    raw_max = raw["Date"].max()
    if raw_max > template_end:
        try:
            import exchange_calendars as xcals

            calendar = xcals.get_calendar("COMEX")
            cal_sessions = pd.DatetimeIndex(
                calendar.sessions_in_range(template_end + pd.Timedelta(days=1), raw_max)
            ).normalize()
            calendar_dates = calendar_dates.union(cal_sessions)
        except Exception:
            future_raw = raw[raw["Date"] > template_end]["Date"]
            future_raw = future_raw[future_raw.dt.dayofweek < 5]
            calendar_dates = calendar_dates.union(pd.DatetimeIndex(future_raw))

    raw = raw.set_index("Date").reindex(calendar_dates)

    for col in value_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.interpolate(method="time", limit_direction="both").ffill().bfill()
    raw = raw.reset_index()
    raw.rename(columns={"index": "Date"}, inplace=True)

    ordered_cols = ["Date"] + [col for col in template.columns if col != "Date"]
    for col in ordered_cols:
        if col not in raw.columns:
            raw[col] = np.nan
    raw = raw[ordered_cols]

    raw.to_csv(SILVER_PREPARED, index=False)

    train_df, test_df = _split_by_date(raw, TRAIN_START, TRAIN_END, TEST_START, TEST_END)
    train_df.to_csv(SILVER_TRAIN, index=False)
    test_df.to_csv(SILVER_TEST, index=False)

    print(f"Silver prepared rows: {len(raw)}")
    print(f"Silver train rows: {len(train_df)} -> {SILVER_TRAIN.name}")
    print(f"Silver test rows: {len(test_df)} -> {SILVER_TEST.name}")


if __name__ == "__main__":
    prepare_gold()
    prepare_silver()
