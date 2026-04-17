import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, skew, kurtosis
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as stats

def calculate_validation_metrics(real_df, gen_df, target_col):
    """
    Calculate metrics comparing real vs generated data for a specific window.
    Drops rows where real data has NaNs.
    """
    # Ensure indices match for boolean indexing
    r_df = real_df.reset_index(drop=True)
    g_df = gen_df.reset_index(drop=True)
    
    mask = r_df[target_col].notna()
    real_clean = r_df[mask]
    gen_clean = g_df[mask]
    
    real_vals = real_clean[target_col].values
    gen_vals = gen_clean[target_col].values
    
    if len(real_vals) < 2:
        return {"error": "Insufficient non-NaN data for validation"}

    # log returns for statistical comparison
    real_returns = np.log(real_clean[target_col] / real_clean[target_col].shift(1)).dropna()
    gen_returns = np.log(gen_clean[target_col] / gen_clean[target_col].shift(1)).dropna()
    
    # 1. Point-wise metrics (Path Fidelity)
    rmse = np.sqrt(mean_squared_error(real_vals, gen_vals))
    mae = mean_absolute_error(real_vals, gen_vals)
    mape = np.mean(np.abs((real_vals - gen_vals) / (real_vals + 1e-9))) * 100
    
    # 2. Distributional metrics (Return Similarity)
    ks_stat, ks_p = ks_2samp(real_returns, gen_returns)
    real_std = real_returns.std()
    gen_std = gen_returns.std()
    vol_ratio = gen_std / (real_std + 1e-9)
    
    # 3. Shape metrics
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape_pct": float(mape),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p),
        "vol_ratio": float(vol_ratio),
        "real_skew": float(skew(real_returns)),
        "gen_skew": float(skew(gen_returns)),
        "real_kurt": float(kurtosis(real_returns)),
        "gen_kurt": float(kurtosis(gen_returns)),
    }
    
    # 4. Directional Accuracy
    real_dir = np.sign(np.diff(real_vals))
    gen_dir = np.sign(np.diff(gen_vals))
    valid_mask = (real_dir != 0)
    metrics["directional_accuracy"] = float(np.mean(real_dir[valid_mask] == gen_dir[valid_mask])) if any(valid_mask) else 0.0
    
    # 5. Advanced Fidelity Metrics
    # ACF-MSE
    real_acf_vals = acf(real_returns, nlags=min(10, len(real_returns)-1))
    gen_acf_vals = acf(gen_returns, nlags=min(10, len(gen_returns)-1))
    metrics["acf_mse"] = float(mean_squared_error(real_acf_vals, gen_acf_vals))
    
    # MMD (Maximum Mean Discrepancy) - using RBF kernel
    metrics["mmd"] = float(calculate_mmd(real_returns.values.reshape(-1, 1), gen_returns.values.reshape(-1, 1)))
    
    # JSD (Jensen-Shannon Divergence) on returns distribution
    metrics["jsd"] = float(calculate_jsd(real_returns, gen_returns))
    
    # 6. Financial Stylized Facts
    metrics["real_arch_effect"] = float(calculate_arch_effect(real_returns))
    metrics["gen_arch_effect"] = float(calculate_arch_effect(gen_returns))
    metrics["mdd_ratio"] = float(calculate_mdd(gen_vals) / (calculate_mdd(real_vals) + 1e-9))
    
    return metrics

def plot_path_comparison(real_df, gen_df, target_col, asset_name, output_dir):
    """
    Plots the price paths of real and synthetic data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(real_df['Date'], real_df[target_col], label='Real Data', color='blue', marker='o', linewidth=2)
    plt.plot(gen_df['Date'], gen_df[target_col], label='GAN Generated', color='orange', linestyle='-', marker='x', linewidth=2)
    plt.title(f"{asset_name} Real vs GAN Generated Path")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{asset_name.lower()}_path_comparison.png"), dpi=300)
    plt.close()

def sanitize_and_align(r, g):
    """Utility to ensure both series are numpy arrays and free of NaNs/Infs."""
    r = np.array(r).flatten()
    g = np.array(g).flatten()
    return r[np.isfinite(r)], g[np.isfinite(g)]

def plot_returns_dist(real_returns, gen_returns, asset_name, output_dir):
    """
    Plots comparison of log-return histograms.
    """
    real_returns, gen_returns = sanitize_and_align(real_returns, gen_returns)
    if len(real_returns) == 0 or len(gen_returns) == 0:
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(real_returns, bins=20, alpha=0.5, label='Real Returns', color='blue', density=True)
    plt.hist(gen_returns, bins=20, alpha=0.5, label='Gen Returns', color='orange', density=True)
    plt.title(f"{asset_name} Log-Return Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, f"{asset_name.lower()}_returns_dist.png"), dpi=300)
    plt.close()

def plot_acf_comparison(real_returns, gen_returns, asset_name, output_dir):
    """
    Plots Autocorrelation Function comparison.
    """
    real_returns, gen_returns = sanitize_and_align(real_returns, gen_returns)
    if len(real_returns) > 5 and len(gen_returns) > 5:
        plt.figure(figsize=(10, 6))
        real_acf = acf(real_returns, nlags=min(10, len(real_returns)-1))
        gen_acf = acf(gen_returns, nlags=min(10, len(gen_returns)-1))
        plt.stem(range(len(real_acf)), real_acf, linefmt='b-', markerfmt='bo', basefmt='r-', label='Real ACF')
        plt.stem(range(len(gen_acf)), gen_acf, linefmt='y-', markerfmt='yx', basefmt='r-', label='Gen ACF')
        plt.title(f"{asset_name} Autocorrelation Comparison")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{asset_name.lower()}_acf_comparison.png"), dpi=300)
        plt.close()

def plot_real_vs_gen(real_df, gen_df, target_col, asset_name, output_dir):
    # Backward compatible wrapper
    plot_path_comparison(real_df, gen_df, target_col, asset_name, output_dir)
    
    # Extract returns on OVERLAP only (so masks align)
    overlap_len = min(len(real_df), len(gen_df))
    r_overlap = real_df.iloc[:overlap_len].reset_index(drop=True)
    g_overlap = gen_df.iloc[:overlap_len].reset_index(drop=True)
    
    mask = r_overlap[target_col].notna()
    real_clean = r_overlap[mask]
    gen_clean = g_overlap[mask]
    
    real_returns = np.log(real_clean[target_col] / real_clean[target_col].shift(1)).dropna()
    gen_returns = np.log(gen_clean[target_col] / gen_clean[target_col].shift(1)).dropna()
    
    plot_returns_dist(real_returns, gen_returns, asset_name, output_dir)
    plot_acf_comparison(real_returns, gen_returns, asset_name, output_dir)

def plot_sequence_diversity(real_data, gen_data, asset_name, output_dir, window_size=15):
    """
    Plots PCA and t-SNE of flattened sequence windows to compare latent-space capture.
    """
    # STRICLY SANITIZE DATA BEFORE WINDOWING
    real_data = np.array(real_data).flatten()
    gen_data = np.array(gen_data).flatten()
    
    real_data = real_data[np.isfinite(real_data)]
    gen_data = gen_data[np.isfinite(gen_data)]
    
    if len(real_data) < window_size or len(gen_data) < window_size:
        print(f"DEBUG: Data too small after sanitization for {asset_name}")
        return

    def extract_windows(data, size):
        windows = []
        for i in range(len(data) - size + 1):
            window = data[i:i+size].flatten()
            if np.all(np.isfinite(window)):
                # Standardize THE WINDOW ITSELF to compare SHAPES (Manifold alignment)
                # (window - mean) / std ensures price level doesn't separate clusters
                w_std = np.std(window)
                if w_std > 1e-8:
                    window = (window - np.mean(window)) / w_std
                else:
                    window = window - np.mean(window)
                windows.append(window)
        return np.array(windows)

    real_windows = extract_windows(real_data, window_size)
    gen_windows = extract_windows(gen_data, window_size)
    
    if len(real_windows) < 2 or len(gen_windows) < 2:
        print(f"DEBUG: Not enough windows for {asset_name}: Real={len(real_windows)}, Gen={len(gen_windows)}")
        return # Not enough data for PCA/t-SNE

    # Sample to balance if necessary
    min_samples = min(len(real_windows), len(gen_windows))
    print(f"DEBUG: Plotting manifold for {asset_name} with {min_samples} samples")
    real_windows = real_windows[:min_samples]
    gen_windows = gen_windows[:min_samples]
    
    combined = np.concatenate([real_windows, gen_windows])
    labels = ["Real"] * min_samples + ["Synthetic"] * min_samples

    # 1. PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(combined)

    # 2. t-SNE
    perp = min(30, max(5, min_samples // 4)) 
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    tsne_results = tsne.fit_transform(combined)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {"Real": "blue", "Synthetic": "orange"}
    for label in ["Real", "Synthetic"]:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax1.scatter(pca_results[idx, 0], pca_results[idx, 1], c=colors[label], label=label, alpha=0.6, s=20)
        ax2.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[label], label=label, alpha=0.6, s=20)

    ax1.set_title(f"{asset_name} PCA (Seq Windows)")
    ax2.set_title(f"{asset_name} t-SNE (Seq Windows)")
    ax1.legend()
    ax2.legend()
    
    # Check scales to confirm normalization worked
    print(f"DEBUG: PCA Scale for {asset_name}: {np.ptp(pca_results, axis=0)}")

    plt.tight_layout()
    # RENAME FILE TO BYPASS CACHE
    save_path = os.path.join(output_dir, f"{asset_name.lower()}_manifold_alignment.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"DEBUG: Saved manifold plot to {save_path}")

def plot_distribution_diagnostics(real_returns, gen_returns, asset_name, output_dir):
    """
    Plots Q-Q Plot and Rolling Volatility comparison.
    """
    real_returns, gen_returns = sanitize_and_align(real_returns, gen_returns)
    if len(real_returns) < 5 or len(gen_returns) < 5:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Q-Q Plot
    stats.probplot(real_returns, dist="norm", plot=ax1)
    stats.probplot(gen_returns, dist="norm", plot=ax1)
    ax1.get_lines()[0].set_color("blue") # Real
    ax1.get_lines()[0].set_label("Real")
    ax1.get_lines()[2].set_color("orange") # Gen
    ax1.get_lines()[2].set_label("Synthetic")
    ax1.set_title(f"{asset_name} Q-Q Plot (Returns)")
    ax1.legend()

    # Rolling Volatility
    # Convert back to Series for rolling
    r_ser = pd.Series(real_returns)
    g_ser = pd.Series(gen_returns)
    window = min(5, len(r_ser) // 2)
    if window > 1:
        real_vol = r_ser.rolling(window=window).std()
        gen_vol = g_ser.rolling(window=window).std()
        ax2.plot(real_vol, label="Real Vol", color="blue")
        ax2.plot(gen_vol, label="Synthetic Vol", color="orange")
        ax2.set_title(f"{asset_name} Rolling Volatility ({window}-day)")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{asset_name.lower()}_dist_diagnostics.png"), dpi=300)
    plt.close()

def plot_financial_diagnostics(real_vals, gen_vals, asset_name, output_dir):
    """
    Plots Cumulative Returns and Volatility Clustering (Squared return ACF).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cumulative Returns
    real_returns = pd.Series(real_vals).pct_change().dropna()
    gen_returns = pd.Series(gen_vals).pct_change().dropna()
    
    if len(real_returns) == 0 or len(gen_returns) == 0:
        return # Cannot plot cumulative returns
    
    ax1.plot((1 + real_returns).cumprod(), label="Real Path", color="blue")
    ax1.plot((1 + gen_returns).cumprod(), label="Synthetic Path", color="orange")
    ax1.set_title(f"{asset_name} Cumulative Returns Comparison")
    ax1.legend()

    # Volatility Clustering (ARCH Effect)
    real_sq_acf = acf(real_returns**2, nlags=min(15, len(real_returns)-1))
    gen_sq_acf = acf(gen_returns**2, nlags=min(15, len(gen_returns)-1))
    
    ax2.stem(range(len(real_sq_acf)), real_sq_acf, linefmt='b-', markerfmt='bo', basefmt='r-', label='Real Sq ACF')
    ax2.stem(range(len(gen_sq_acf)), gen_sq_acf, linefmt='y-', markerfmt='yx', basefmt='r-', label='Gen Sq ACF')
    ax2.set_title(f"{asset_name} Volatility Clustering (Sq Return ACF)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{asset_name.lower()}_financial_diagnostics.png"), dpi=300)
    plt.close()

def log_validation_results(metrics, asset_name, log_path):
    with open(log_path, 'w') as f:
        f.write(f"--- GAN Validation Report: {asset_name} ---\n")
        if "error" in metrics:
            f.write(f"ERROR: {metrics['error']}\n")
        else:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
    print(f"Validation metrics logged to {log_path}")

def calculate_mmd(x, y, gamma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel.
    """
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

def calculate_jsd(p, q, bins=20):
    """
    Jensen-Shannon Divergence using histogram binning.
    """
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    # Add epsilon to avoid log(0)
    p_hist += 1e-9
    q_hist += 1e-9
    return jensenshannon(p_hist, q_hist)

def calculate_arch_effect(returns, lag=1):
    """
    Measure serial correlation of squared returns (volatility clustering).
    """
    sq_returns = returns**2
    if len(sq_returns) <= lag: return 0.0
    return np.corrcoef(sq_returns[:-lag], sq_returns[lag:])[0, 1]

def calculate_mdd(prices):
    """
    Calculate Maximum Drawdown.
    """
    df = pd.Series(prices)
    roll_max = df.cummax()
    drawdown = (df - roll_max) / roll_max
    return abs(drawdown.min())
