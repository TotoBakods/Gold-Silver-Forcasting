import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "gan"))

from gan_core import Generator, make_stationary, reconstruct_future_rows
from validation_utils import (
    calculate_validation_metrics, log_validation_results, 
    plot_real_vs_gen, plot_returns_dist, plot_acf_comparison,
    plot_sequence_diversity, plot_distribution_diagnostics,
    plot_financial_diagnostics
)
from dataset_catalog import get_dataset_config_by_asset, DATASET_CONFIGS

GEN_DAYS = 60
WINDOW_SIZE = 15
NOISE_DIM = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_60d_forecast(asset_name, val_start="2026-04-01", val_end="2026-04-13"):
    print(f"--- 60-Day Diagnostic Report for {asset_name.upper()} ---")
    
    # 1. Setup paths
    asset_dir = PROJECT_ROOT / "gan" / asset_name
    report_dir = asset_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    model_path = asset_dir / f"{asset_name}_val_gen.pth"
    
    # 2. Load Real Data
    config = next(c for c in DATASET_CONFIGS if c['asset'] == asset_name)
    df = pd.read_csv(PROJECT_ROOT / config['source_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    price_cols = [config['target_col']]
    if asset_name == "gold":
        if "Silver_Futures" in df.columns: price_cols.append("Silver_Futures")
        if "Crude_Oil_Futures" in df.columns: price_cols.append("Crude_Oil_Futures")
    elif asset_name == "silver":
        if "Gold_Futures" in df.columns: price_cols.append("Gold_Futures")
        
    rate_cols = [c for c in df.columns if c not in price_cols and c != 'Date']
    all_features = price_cols + rate_cols
    
    train_df = df[df['Date'] <= "2026-03-31"].copy()
    real_test_df = df[(df['Date'] >= val_start) & (df['Date'] <= val_end)].copy()
    
    # 3. Preprocess for initial window
    from sklearn.preprocessing import StandardScaler
    stat_train = make_stationary(train_df, price_cols, rate_cols)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(stat_train[all_features].values)
    
    # 4. Load Model and Generate
    num_features = len(all_features)
    netG = Generator(num_features, 128, num_features, noise_dim=NOISE_DIM).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    last_window = scaled_train[-WINDOW_SIZE:]
    current_window = torch.FloatTensor(last_window).unsqueeze(0).to(device)
    last_known_vals = train_df[all_features].iloc[-1].values
    
    gen_stat_scaled = []
    for _ in range(GEN_DAYS):
        with torch.no_grad():
            noise = torch.randn(1, WINDOW_SIZE, NOISE_DIM, device=device)
            next_stat = netG(current_window, noise)
            gen_stat_scaled.append(next_stat.cpu().numpy()[0, 0, :])
            current_window = torch.cat((current_window[:, 1:, :], next_stat), dim=1)
            
    gen_stat = scaler.inverse_transform(np.array(gen_stat_scaled))
    recon_vals = reconstruct_future_rows(last_known_vals, gen_stat, all_features, price_cols)
    
    # 5. Create DataFrame
    gen_df = pd.DataFrame(recon_vals, columns=all_features)
    
    # Aligned date generation
    full_real_dates = df[df['Date'] >= val_start]['Date'].sort_values().unique()
    dates_for_gen = []
    for i in range(GEN_DAYS):
        if i < len(full_real_dates):
            dates_for_gen.append(full_real_dates[i])
        else:
            last_date = dates_for_gen[-1]
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)
            dates_for_gen.append(next_date)
            
    gen_df.insert(0, 'Date', dates_for_gen[:GEN_DAYS])
    
    # 6. Metrics (Overlap only: April 1-13)
    overlap_len = min(len(real_test_df), len(gen_df))
    metrics = calculate_validation_metrics(real_test_df.iloc[:overlap_len], gen_df.iloc[:overlap_len], config['target_col'])
    log_validation_results(metrics, f"{asset_name} (2-Month Extended Diagnostics)", report_dir / "validation_metrics.txt")
    
    # 7. Plots
    print("Generating comprehensive plots...")
    plot_real_vs_gen(real_test_df, gen_df, config['target_col'], asset_name.capitalize(), report_dir)
    plot_returns_dist(real_test_df[config['target_col']], gen_df[config['target_col']], asset_name, report_dir)
    plot_acf_comparison(real_test_df[config['target_col']], gen_df[config['target_col']], asset_name, report_dir)
    
    # Advanced
    # Ensure we grab the last 300 days of history BEFORE the generation start date
    real_history_df = df[df['Date'] < val_start]
    train_slice = real_history_df[config['target_col']].values[-300:]
    target_gen = gen_df[config['target_col']]
    
    print(f"DEBUG: Manifold Comparison: Real History Size={len(train_slice)}, Gen Size={len(target_gen)}")
    plot_sequence_diversity(train_slice, target_gen.values, asset_name, report_dir)
    
    real_ret = pd.Series(real_test_df[config['target_col']].values).pct_change().dropna()
    gen_ret = pd.Series(target_gen.values).pct_change().dropna()
    plot_distribution_diagnostics(real_ret, gen_ret, asset_name, report_dir)
    plot_financial_diagnostics(real_test_df[config['target_col']].values, target_gen.values, asset_name, report_dir)
    
    # Save CSV
    gen_df.to_csv(report_dir / "synthetic_forecast_60d.csv", index=False)
    print(f"Results saved to {report_dir}")

if __name__ == "__main__":
    # GLOBAL START DATE: MATCH GROUND TRUTH MARCH 1ST
    START_DATE = "2026-03-01"
    END_DATE = "2026-04-14"
    run_60d_forecast("gold", val_start=START_DATE, val_end=END_DATE)
    run_60d_forecast("silver", val_start=START_DATE, val_end=END_DATE)
