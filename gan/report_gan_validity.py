import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, skew, kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports", "gan_validation")
TARGET_END_DATE = pd.Timestamp("2026-05-08")
SOURCE_FILES = ["gold_RRL_interpolate.csv", "silver_RRL_interpolate.csv"]


def classify_columns(df):
    price_keywords = ["Futures", "US30", "SnP500", "NASDAQ_100"]
    price_cols = [c for c in df.columns if c != "Date" and any(k in c for k in price_keywords)]
    other_cols = [c for c in df.columns if c not in price_cols and c != "Date"]
    return price_cols, other_cols


def stationary_frame(df, price_cols, other_cols):
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in other_cols:
        stat_df[col] = df[col].diff()
    return stat_df.dropna().reset_index(drop=True)


def autocorr_at_lags(series, max_lag=5):
    clean = pd.Series(series).dropna()
    values = {}
    for lag in range(1, max_lag + 1):
        values[f"acf_{lag}"] = float(clean.autocorr(lag=lag)) if len(clean) > lag else np.nan
    return values


def format_summary_value(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def save_line_panel(df_hist, df_future, columns, output_path, title):
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, columns):
        recent_hist = df_hist[["Date", col]].tail(252)
        ax.plot(recent_hist["Date"], recent_hist[col], label="Recent historical", color="#1f77b4")
        ax.plot(df_future["Date"], df_future[col], label="Generated future", color="#ff7f0e", linestyle="--")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)

    for ax in axes[len(columns):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_distribution_panel(hist_stat_df, future_stat_df, columns, output_path, title):
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, columns):
        hist_vals = hist_stat_df[col].dropna().to_numpy()
        future_vals = future_stat_df[col].dropna().to_numpy()
        bins = 40
        ax.hist(hist_vals, bins=bins, density=True, alpha=0.5, label="Historical", color="#1f77b4")
        ax.hist(future_vals, bins=bins, density=True, alpha=0.5, label="Generated", color="#ff7f0e")
        ax.set_title(f"{col} stationary distribution")
        ax.grid(alpha=0.2)

    for ax in axes[len(columns):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmaps(hist_corr, future_corr, output_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    matrices = [(hist_corr, "Historical stationary correlation"), (future_corr, "Generated stationary correlation")]

    for ax, (corr_df, subtitle) in zip(axes, matrices):
        image = ax.imshow(corr_df.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(subtitle)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_yticklabels(corr_df.index)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.88, wspace=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_acf_panel(hist_stat_df, future_stat_df, columns, output_path, title):
    max_lag = 5
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    x = np.arange(1, max_lag + 1)

    for ax, col in zip(axes, columns):
        hist_acf = [hist_stat_df[col].autocorr(lag) for lag in x]
        future_acf = [future_stat_df[col].autocorr(lag) for lag in x]
        width = 0.35
        ax.bar(x - width / 2, hist_acf, width=width, label="Historical", color="#1f77b4")
        ax.bar(x + width / 2, future_acf, width=width, label="Generated", color="#ff7f0e")
        ax.set_title(f"{col} stationary autocorrelation")
        ax.set_xticks(x)
        ax.set_xlabel("Lag")
        ax.grid(alpha=0.2)

    for ax in axes[len(columns):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def report_file(source_name):
    source_path = os.path.join(PROJECT_ROOT, source_name)
    extended_path = os.path.join(PROJECT_ROOT, f"{os.path.splitext(source_name)[0]}_extended.csv")
    if not os.path.exists(extended_path):
        raise FileNotFoundError(f"Missing generated file: {extended_path}")

    source_df = pd.read_csv(source_path)
    extended_df = pd.read_csv(extended_path)
    for df in (source_df, extended_df):
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    price_cols, other_cols = classify_columns(source_df)
    feature_cols = price_cols + other_cols
    last_hist_date = source_df["Date"].max()
    future_df = extended_df[extended_df["Date"] > last_hist_date].copy()
    if future_df.empty:
        raise ValueError(f"No generated future rows found in {extended_path}")

    hist_stat_df = stationary_frame(source_df, price_cols, other_cols)
    bridge_future = pd.concat([source_df.tail(1), future_df], ignore_index=True)
    future_stat_df = stationary_frame(bridge_future, price_cols, other_cols)

    dataset_name = os.path.splitext(source_name)[0]
    report_dir = os.path.join(REPORTS_ROOT, dataset_name)
    plots_dir = os.path.join(report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    feature_rows = []
    for col in feature_cols:
        hist_series = source_df[col].astype(float)
        future_series = future_df[col].astype(float)
        hist_stat = hist_stat_df[col].astype(float)
        future_stat = future_stat_df[col].astype(float)

        ks_stat, ks_pvalue = ks_2samp(hist_stat, future_stat)
        hist_acf = autocorr_at_lags(hist_stat, max_lag=5)
        future_acf = autocorr_at_lags(future_stat, max_lag=5)

        row = {
            "feature": col,
            "series_type": "price" if col in price_cols else "difference",
            "hist_level_min": float(hist_series.min()),
            "hist_level_max": float(hist_series.max()),
            "future_level_min": float(future_series.min()),
            "future_level_max": float(future_series.max()),
            "hist_stat_mean": float(hist_stat.mean()),
            "future_stat_mean": float(future_stat.mean()),
            "mean_gap_abs": float(abs(hist_stat.mean() - future_stat.mean())),
            "hist_stat_std": float(hist_stat.std()),
            "future_stat_std": float(future_stat.std()),
            "vol_ratio": float(future_stat.std() / (hist_stat.std() + 1e-12)),
            "hist_stat_skew": float(skew(hist_stat, bias=False)),
            "future_stat_skew": float(skew(future_stat, bias=False)),
            "hist_stat_kurtosis": float(kurtosis(hist_stat, fisher=True, bias=False)),
            "future_stat_kurtosis": float(kurtosis(future_stat, fisher=True, bias=False)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }

        for lag in range(1, 6):
            row[f"hist_acf_{lag}"] = hist_acf[f"acf_{lag}"]
            row[f"future_acf_{lag}"] = future_acf[f"acf_{lag}"]
            row[f"acf_gap_{lag}"] = abs(hist_acf[f"acf_{lag}"] - future_acf[f"acf_{lag}"])

        feature_rows.append(row)

    feature_metrics = pd.DataFrame(feature_rows)
    hist_corr = hist_stat_df[feature_cols].corr()
    future_corr = future_stat_df[feature_cols].corr()
    corr_gap = (hist_corr - future_corr).abs()
    corr_gap_long = corr_gap.stack().reset_index()
    corr_gap_long.columns = ["feature_a", "feature_b", "abs_corr_gap"]
    corr_gap_long = corr_gap_long[corr_gap_long["feature_a"] < corr_gap_long["feature_b"]].sort_values(
        "abs_corr_gap", ascending=False
    )

    expected_future_dates = pd.date_range(
        start=last_hist_date + pd.Timedelta(days=1),
        end=TARGET_END_DATE,
        freq="B",
    )
    summary = {
        "dataset": dataset_name,
        "source_last_date": str(last_hist_date.date()),
        "generated_last_date": str(extended_df["Date"].max().date()),
        "future_rows": int(len(future_df)),
        "expected_future_rows": int(len(expected_future_dates)),
        "duplicate_dates": int(extended_df["Date"].duplicated().sum()),
        "null_cells": int(extended_df.isna().sum().sum()),
        "avg_vol_ratio": float(feature_metrics["vol_ratio"].mean()),
        "max_vol_ratio_gap": float((feature_metrics["vol_ratio"] - 1.0).abs().max()),
        "avg_mean_gap_abs": float(feature_metrics["mean_gap_abs"].mean()),
        "avg_ks_statistic": float(feature_metrics["ks_statistic"].mean()),
        "max_ks_statistic": float(feature_metrics["ks_statistic"].max()),
        "avg_acf_gap_lag1": float(feature_metrics["acf_gap_1"].mean()),
        "max_acf_gap_lag1": float(feature_metrics["acf_gap_1"].max()),
        "corr_matrix_mae": float(corr_gap_long["abs_corr_gap"].mean()),
        "corr_matrix_max_gap": float(corr_gap_long["abs_corr_gap"].max()) if not corr_gap_long.empty else 0.0,
        "share_features_vol_ratio_pass_0p8_to_1p2": float(((feature_metrics["vol_ratio"] >= 0.8) & (feature_metrics["vol_ratio"] <= 1.2)).mean()),
        "share_features_ks_pvalue_gt_0p05": float((feature_metrics["ks_pvalue"] > 0.05).mean()),
        "share_features_acf_gap_lag1_lt_0p15": float((feature_metrics["acf_gap_1"] < 0.15).mean()),
    }

    summary_df = pd.DataFrame(
        [{"metric": key, "value": format_summary_value(value)} for key, value in summary.items()]
    )
    summary_df.to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)
    feature_metrics.to_csv(os.path.join(report_dir, "feature_metrics.csv"), index=False)
    corr_gap_long.to_csv(os.path.join(report_dir, "correlation_gap.csv"), index=False)

    report_payload = {
        "summary": {key: format_summary_value(value) for key, value in summary.items()},
        "top_correlation_gaps": corr_gap_long.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(report_dir, "report_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    top_ks = feature_metrics.nlargest(min(5, len(feature_metrics)), "ks_statistic")[
        ["feature", "ks_statistic", "ks_pvalue", "vol_ratio", "acf_gap_1"]
    ]
    md_lines = [
        f"# GAN Validity Report: {dataset_name}",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        md_lines.append(f"- {key}: {format_summary_value(value)}")
    md_lines.extend(["", "## Highest KS Distance Features"])
    for _, row in top_ks.iterrows():
        md_lines.append(
            f"- {row['feature']}: ks_statistic={row['ks_statistic']:.4f}, "
            f"ks_pvalue={row['ks_pvalue']:.4f}, vol_ratio={row['vol_ratio']:.4f}, "
            f"acf_gap_1={row['acf_gap_1']:.4f}"
        )
    with open(os.path.join(report_dir, "report_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    save_line_panel(
        source_df,
        future_df,
        feature_cols,
        os.path.join(plots_dir, "recent_history_vs_generated.png"),
        f"{dataset_name}: recent history vs generated future",
    )
    save_distribution_panel(
        hist_stat_df,
        future_stat_df,
        feature_cols,
        os.path.join(plots_dir, "stationary_distribution_comparison.png"),
        f"{dataset_name}: stationary distribution comparison",
    )
    save_heatmaps(
        hist_corr,
        future_corr,
        os.path.join(plots_dir, "stationary_correlation_heatmaps.png"),
        f"{dataset_name}: stationary correlation structure",
    )
    save_acf_panel(
        hist_stat_df,
        future_stat_df,
        feature_cols,
        os.path.join(plots_dir, "stationary_autocorrelation.png"),
        f"{dataset_name}: stationary autocorrelation comparison",
    )

    print(f"Saved GAN validity report to {report_dir}")


def main():
    os.makedirs(REPORTS_ROOT, exist_ok=True)
    for source_name in SOURCE_FILES:
        report_file(source_name)


if __name__ == "__main__":
    main()
