import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports" / "gan_diagnostics"

DEFAULT_START = "2026-04-01"
DEFAULT_END = "2026-04-13"

ASSET_CONFIGS = {
    "gold": {
        "test_csv": "df_gold_dataset_usa_epu_test_2026_04_01_2026_04_13.csv",
        "synthetic_csv": "gold_apr2026_train_extended.csv",
    },
    "silver": {
        "test_csv": "df_silver_dataset_test_2026_04_01_2026_04_13.csv",
        "synthetic_csv": "silver_apr2026_train_extended.csv",
    },
}


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


def acf_values(series, max_lag=5):
    clean = pd.Series(series).dropna()
    values = []
    for lag in range(1, max_lag + 1):
        if len(clean) <= lag:
            values.append(np.nan)
        else:
            values.append(float(clean.autocorr(lag=lag)))
    return values


def pacf_values(series, max_lag=5):
    clean = pd.Series(series).dropna().to_numpy(dtype=float)
    values = []
    if clean.size == 0:
        return [np.nan] * max_lag
    clean = (clean - clean.mean()) / (clean.std() + 1e-12)
    for lag in range(1, max_lag + 1):
        if clean.size <= lag:
            values.append(np.nan)
            continue
        y = clean[lag:]
        X = np.column_stack([clean[lag - i - 1 : -i - 1] for i in range(lag)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        values.append(float(coeffs[-1]))
    return values


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def plot_wasserstein(distances, output_path):
    labels = list(distances.keys())
    values = [distances[k] for k in labels]
    plt.figure(figsize=(max(8, len(labels) * 0.6), 4))
    plt.bar(labels, values, color="#ff7f0e")
    plt.title("Wasserstein distance per feature (stationary)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_hist_overlays(real_df, synth_df, columns, output_path):
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, columns):
        real_vals = real_df[col].dropna().to_numpy()
        synth_vals = synth_df[col].dropna().to_numpy()
        bins = 30
        ax.hist(real_vals, bins=bins, density=True, alpha=0.5, label="Real", color="#1f77b4")
        ax.hist(synth_vals, bins=bins, density=True, alpha=0.5, label="Synthetic", color="#ff7f0e")
        ax.set_title(f"{col} stationary")
        ax.grid(alpha=0.2)

    for ax in axes[len(columns):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Stationary distribution overlays")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lag_panel(real_df, synth_df, columns, output_path, title, lag_func):
    max_lag = 5
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    x = np.arange(1, max_lag + 1)

    for ax, col in zip(axes, columns):
        real_acf = lag_func(real_df[col], max_lag=max_lag)
        synth_acf = lag_func(synth_df[col], max_lag=max_lag)
        width = 0.35
        ax.bar(x - width / 2, real_acf, width=width, label="Real", color="#1f77b4")
        ax.bar(x + width / 2, synth_acf, width=width, label="Synthetic", color="#ff7f0e")
        ax.set_title(col)
        ax.set_xticks(x)
        ax.set_xlabel("Lag")
        ax.grid(alpha=0.2)

    for ax in axes[len(columns):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmaps(real_corr, synth_corr, output_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    matrices = [(real_corr, "Real"), (synth_corr, "Synthetic")]
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


def plot_pca_tsne(X_scaled, labels, output_dir):
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 6))
    for label_value, label_name, color in [(0, "Real", "#1f77b4"), (1, "Synthetic", "#ff7f0e")]:
        mask = labels == label_value
        plt.scatter(pca_points[mask, 0], pca_points[mask, 1], label=label_name, alpha=0.8, color=color)
    plt.title("PCA projection (stationary)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scatter.png", dpi=180)
    plt.close()

    tsne_output = output_dir / "tsne_scatter.png"
    try:
        tsne = TSNE(n_components=2, perplexity=5, learning_rate="auto", init="pca", random_state=42)
        tsne_points = tsne.fit_transform(X_scaled)
        plt.figure(figsize=(7, 6))
        for label_value, label_name, color in [(0, "Real", "#1f77b4"), (1, "Synthetic", "#ff7f0e")]:
            mask = labels == label_value
            plt.scatter(tsne_points[mask, 0], tsne_points[mask, 1], label=label_name, alpha=0.8, color=color)
        plt.title("t-SNE projection (stationary)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(tsne_output, dpi=180)
        plt.close()
    except Exception as exc:
        print(f"t-SNE skipped: {exc}")


def plot_classifier_scores(real_scores, synth_scores, output_path, accuracy):
    plt.figure(figsize=(8, 4))
    plt.hist(real_scores, bins=20, alpha=0.5, label="Real", color="#1f77b4")
    plt.hist(synth_scores, bins=20, alpha=0.5, label="Synthetic", color="#ff7f0e")
    plt.axvline(0.5, color="#666", linestyle="--", label="Decision 0.5")
    plt.title(f"Classifier scores (accuracy={accuracy:.2%})")
    plt.xlabel("Predicted probability of Real")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_diagnostics(asset_name, test_csv, synthetic_csv, start_date, end_date):
    report_dir = REPORTS_ROOT / asset_name
    _ensure_dir(report_dir)

    test_df = pd.read_csv(test_csv)
    synth_df = pd.read_csv(synthetic_csv)

    for df in (test_df, synth_df):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df.dropna(subset=["Date"], inplace=True)
        df.sort_values("Date", inplace=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    test_df = test_df[(test_df["Date"] >= start) & (test_df["Date"] <= end)].copy()
    synth_df = synth_df[(synth_df["Date"] >= start) & (synth_df["Date"] <= end)].copy()

    if test_df.empty or synth_df.empty:
        raise ValueError(f"Missing rows for {asset_name} in the {start_date} to {end_date} window.")

    common_cols = [col for col in test_df.columns if col in synth_df.columns and col != "Date"]
    test_df = test_df[["Date"] + common_cols]
    synth_df = synth_df[["Date"] + common_cols]

    price_cols, other_cols = classify_columns(test_df)
    stat_real = stationary_frame(test_df, price_cols, other_cols)
    stat_synth = stationary_frame(synth_df, price_cols, other_cols)

    feature_cols = [col for col in stat_real.columns if col != "Date"]
    stat_real = stat_real[feature_cols]
    stat_synth = stat_synth[feature_cols]

    stat_real = stat_real.replace([np.inf, -np.inf], np.nan).dropna()
    stat_synth = stat_synth.replace([np.inf, -np.inf], np.nan).dropna()

    wasserstein = {
        col: float(wasserstein_distance(stat_real[col], stat_synth[col])) for col in feature_cols
    }
    pd.DataFrame(
        {"feature": list(wasserstein.keys()), "wasserstein": list(wasserstein.values())}
    ).to_csv(report_dir / "wasserstein_distances.csv", index=False)

    plot_wasserstein(wasserstein, report_dir / "wasserstein_distance.png")
    plot_hist_overlays(stat_real, stat_synth, feature_cols, report_dir / "hist_overlays.png")
    plot_lag_panel(stat_real, stat_synth, feature_cols, report_dir / "acf_comparison.png", "ACF comparison", acf_values)
    plot_lag_panel(stat_real, stat_synth, feature_cols, report_dir / "pacf_comparison.png", "PACF comparison", pacf_values)

    real_corr = stat_real.corr()
    synth_corr = stat_synth.corr()
    plot_correlation_heatmaps(real_corr, synth_corr, report_dir / "correlation_heatmaps.png", "Correlation heatmaps")

    scaler = StandardScaler()
    X = np.vstack([stat_real.values, stat_synth.values])
    X_scaled = scaler.fit_transform(X)
    labels = np.concatenate([np.zeros(len(stat_real)), np.ones(len(stat_synth))])

    plot_pca_tsne(X_scaled, labels, report_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))

    real_scores = probs[: len(stat_real)]
    synth_scores = probs[len(stat_real) :]
    plot_classifier_scores(real_scores, synth_scores, report_dir / "classifier_scores.png", accuracy)

    summary = {
        "asset": asset_name,
        "start_date": start_date,
        "end_date": end_date,
        "wasserstein_mean": float(np.mean(list(wasserstein.values()))),
        "classifier_accuracy": accuracy,
        "samples_real": int(len(stat_real)),
        "samples_synth": int(len(stat_synth)),
    }
    with open(report_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved GAN diagnostics for {asset_name} -> {report_dir}")


def main():
    start_date = os.getenv("GAN_DIAG_START", DEFAULT_START)
    end_date = os.getenv("GAN_DIAG_END", DEFAULT_END)

    for asset_name, config in ASSET_CONFIGS.items():
        test_csv = os.getenv(f"GAN_DIAG_{asset_name.upper()}_TEST", config["test_csv"])
        synth_csv = os.getenv(f"GAN_DIAG_{asset_name.upper()}_SYNTH", config["synthetic_csv"])
        run_diagnostics(asset_name, test_csv, synth_csv, start_date, end_date)


if __name__ == "__main__":
    main()
