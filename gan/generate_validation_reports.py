import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gan.dataset_catalog import get_dataset_config_by_asset, PROJECT_ROOT
from gan.gan_core import Generator, make_stationary, reconstruct_future_rows
from gan.validation_utils import calculate_validation_metrics, plot_real_vs_gen, log_validation_results

def run_validation(asset):
    print(f"--- Generating Validation Reports for {asset} ---")
    output_dir = Path(__file__).resolve().parent / asset
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    config = get_dataset_config_by_asset(asset)
    # Ensure we use the validation dataset
    from gan.dataset_catalog import DATASET_CONFIGS
    config = next(c for c in DATASET_CONFIGS if c['name'] == f"{asset}_2015_2026_val")
    
    df = pd.read_csv(PROJECT_ROOT / config['source_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    TRAIN_CUTOFF = "2026-03-31"
    VAL_START = "2026-04-01"
    VAL_END = "2026-04-13"
    WINDOW_SIZE = 15
    NOISE_DIM = 32
    
    train_df = df[df['Date'] <= TRAIN_CUTOFF].copy()
    real_test_df = df[(df['Date'] >= VAL_START) & (df['Date'] <= VAL_END)].copy()
    
    # 2. Preprocess (same as training)
    price_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100']
    rate_keywords = ['Yield', 'Rate', 'Ratio', 'USD_index', 'gepu', 'gpr_daily']
    
    price_cols = [c for c in df.columns if any(kw in c for kw in price_keywords)]
    rate_cols = [c for c in df.columns if any(kw in c for kw in rate_keywords)]
    leftovers = [c for c in df.columns if c not in price_cols and c != 'Date' and c not in rate_cols]
    rate_cols += leftovers
    all_features = price_cols + rate_cols
    
    stat_train_df = make_stationary(train_df, price_cols, rate_cols)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(stat_train_df[all_features].values)
    
    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = len(all_features)
    netG = Generator(num_features, 128, num_features).to(device)
    
    from gan.dataset_catalog import get_model_output_path
    model_path = get_model_output_path(config)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    netG.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    netG.eval()
    
    # 4. Generate
    last_window = scaler.transform(stat_train_df[all_features].values)[-WINDOW_SIZE:]
    current_window = torch.FloatTensor(last_window).unsqueeze(0).to(device)
    last_known_vals = train_df[all_features].iloc[-1].values
    
    gen_stat_scaled = []
    for _ in range(len(real_test_df)):
        with torch.no_grad():
            noise = torch.randn(1, WINDOW_SIZE, NOISE_DIM, device=device)
            next_stat = netG(current_window, noise)
            gen_stat_scaled.append(next_stat.cpu().numpy()[0, 0, :])
            current_window = torch.cat((current_window[:, 1:, :], next_stat), dim=1)
            
    gen_stat = scaler.inverse_transform(np.array(gen_stat_scaled))
    recon_vals = reconstruct_future_rows(last_known_vals, gen_stat, all_features, price_cols)
    
    gen_test_df = pd.DataFrame(recon_vals, columns=all_features)
    gen_test_df.insert(0, 'Date', real_test_df['Date'].values)
    
    # Save synthetic path for API
    gen_test_df.to_csv(output_dir / "gan_validation_path.csv", index=False)
    print(f"Saved synthetic path to {output_dir / 'gan_validation_path.csv'}")
    
    # 5. Validate & Plot
    metrics = calculate_validation_metrics(real_test_df, gen_test_df, config['target_col'])
    log_validation_results(metrics, f"{asset.capitalize()} (Val: April 2026)", report_dir / "validation_metrics.txt")
    
    # Comprehensive Plotting
    plot_real_vs_gen(real_test_df, gen_test_df, config['target_col'], asset.capitalize(), report_dir)
    
    # Extract returns for advanced diagnostics
    real_returns = np.log(real_test_df[config['target_col']] / real_test_df[config['target_col']].shift(1)).dropna()
    gen_returns = np.log(gen_test_df[config['target_col']] / gen_test_df[config['target_col']].shift(1)).dropna()
    
    from gan.validation_utils import plot_distribution_diagnostics, plot_financial_diagnostics
    plot_distribution_diagnostics(real_returns, gen_returns, asset.capitalize(), report_dir)
    plot_financial_diagnostics(real_test_df[config['target_col']], gen_test_df[config['target_col']], asset.capitalize(), report_dir)
    
    print(f"Validation reports generated in {report_dir}")

if __name__ == "__main__":
    run_validation("gold")
    run_validation("silver")
