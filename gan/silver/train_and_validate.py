import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from tqdm import tqdm

# Add parent directory to sys.path to import from gan/
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset_catalog import get_dataset_config_by_asset, PROJECT_ROOT
from gan_core import (
    Generator, Discriminator, make_stationary, 
    compute_moment_loss, compute_drift_loss, 
    discriminator_hinge_loss, reconstruct_future_rows, set_global_seed
)
from validation_utils import calculate_validation_metrics, plot_real_vs_gen, log_validation_results

# Constants
ASSET = "silver"
TRAIN_CUTOFF = "2026-02-28"
VAL_START = "2026-03-01"
VAL_END = "2026-04-14"
WINDOW_SIZE = 15
NOISE_DIM = 64
BATCH_SIZE = 64
EPOCHS = 2000
LR_G = 0.0002
LR_D = 0.0002
N_CRITIC = 5
GEN_DAYS = 60 # 2 months in advance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Setup directory and logging
    output_dir = Path(__file__).resolve().parent
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=output_dir / 'gan_training_validation.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.info(f"Starting GAN Training & Validation for {ASSET}")
    print(f"Starting GAN Training & Validation for {ASSET}")
    
    # 1. Load Data
    config = get_dataset_config_by_asset(ASSET)
    if config['name'] != "silver_2015_2026_val":
        from dataset_catalog import DATASET_CONFIGS
        config = next(c for c in DATASET_CONFIGS if c['name'] == "silver_2015_2026_val")

    df = pd.read_csv(PROJECT_ROOT / config['source_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Define features
    price_cols = [config['target_col']]
    if "Gold_Futures" in df.columns: price_cols.append("Gold_Futures")
    
    rate_cols = [c for c in df.columns if c not in price_cols and c != 'Date']
    all_features = price_cols + rate_cols
    
    # Split into Train and Real Test
    train_df = df[df['Date'] <= TRAIN_CUTOFF].copy()
    real_test_df = df[(df['Date'] >= VAL_START) & (df['Date'] <= VAL_END)].copy()
    
    if len(real_test_df) == 0:
        print(f"Error: No real data found for validation period {VAL_START} to {VAL_END}")
        return

    # 2. Preprocess
    stat_train_df = make_stationary(train_df, price_cols, rate_cols)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(stat_train_df[all_features].values)
    
    # Sequence building
    X, Y = [], []
    for i in range(len(scaled_train) - WINDOW_SIZE):
        X.append(scaled_train[i:i+WINDOW_SIZE])
        Y.append(scaled_train[i+WINDOW_SIZE])
    
    X_tensor = torch.FloatTensor(np.array(X)).to(device)
    Y_tensor = torch.FloatTensor(np.array(Y)).unsqueeze(1).to(device)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Train
    set_global_seed(123) # Different seed for silver
    num_features = len(all_features)
    netG = Generator(num_features, 128, num_features, noise_dim=NOISE_DIM).to(device)
    netD = Discriminator(num_features, 128).to(device)
    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.9))
    
    logging.info(f"Training parameters: Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR_G={LR_G}, LR_D={LR_D}")
    print("Training...")
    for epoch in tqdm(range(EPOCHS), desc=f"Training {ASSET}"):
        for i, (history, actual_next) in enumerate(dataloader):
            # history, actual_next are already on device
            
            # Train D
            for _ in range(N_CRITIC):
                optD.zero_grad()
                noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
                fake_next = netG(history, noise)
                
                real_score = netD(torch.cat((history, actual_next), dim=1))
                fake_score = netD(torch.cat((history, fake_next.detach()), dim=1))
                d_loss = discriminator_hinge_loss(real_score, fake_score)
                d_loss.backward()
                optD.step()
            
            # Train G
            optG.zero_grad()
            noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
            fake_next = netG(history, noise)
            g_adv_loss = -netD(torch.cat((history, fake_next), dim=1)).mean()
            mse_loss = nn.MSELoss()(fake_next, actual_next)
            moment_loss = compute_moment_loss(fake_next, actual_next)
            drift_loss = compute_drift_loss(history, fake_next, actual_next)
            
            g_loss = g_adv_loss + 0.1 * mse_loss + 10.0 * moment_loss + 1.0 * drift_loss
            g_loss.backward()
            optG.step()
            
        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            msg = f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
            logging.info(msg)
            print(msg)

    # 4. Generate for Test Period
    logging.info("Generating synthetic data for validation period...")
    print("Generating synthetic data for validation period...")
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
    
    # Create Generated DataFrame
    gen_test_df = pd.DataFrame(recon_vals, columns=all_features)
    
    # Generate dates matched to ground truth first, then extend
    full_real_dates = df[df['Date'] >= VAL_START]['Date'].sort_values().unique()
    
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
            
    gen_test_df.insert(0, 'Date', dates_for_gen[:GEN_DAYS])
    
    # Save results
    gen_test_df.to_csv(report_dir / "synthetic_test_data.csv", index=False)
    logging.info(f"Synthetic data saved to {report_dir / 'synthetic_test_data.csv'}")

    # Save the model before validation in case of metrics crash
    torch.save(netG.state_dict(), output_dir / f"{ASSET}_val_gen.pth")
    logging.info(f"Model saved to {output_dir / f'{ASSET}_val_gen.pth'}")
    
    # 5. Validate & Plot
    logging.info("Calculating metrics on overlapping period...")
    print("Calculating metrics on overlapping period...")
    # metrics sliced to real test period (up to April 13)
    overlap_len = min(len(real_test_df), len(gen_test_df))
    metrics = calculate_validation_metrics(real_test_df.iloc[:overlap_len], gen_test_df.iloc[:overlap_len], config['target_col'])
    
    log_validation_results(metrics, f"{ASSET} (Ground Truth Comparison up to 2026-04-13)", report_dir / "validation_metrics.txt")
    
    # Advanced Plotting
    logging.info("Generating advanced diagnostic plots...")
    from validation_utils import (
        plot_returns_dist, plot_acf_comparison, 
        plot_sequence_diversity, plot_distribution_diagnostics, 
        plot_financial_diagnostics
    )
    
    # Base Path Plot
    plot_real_vs_gen(real_test_df, gen_test_df, config['target_col'], ASSET.capitalize(), report_dir)
    
    # Statistical Plots
    target_real = real_test_df[config['target_col']]
    target_gen = gen_test_df[config['target_col']]
    plot_returns_dist(target_real, target_gen, ASSET, report_dir)
    plot_acf_comparison(target_real, target_gen, ASSET, report_dir)
    
    # PCA/t-SNE (Sample training bits for manifold check)
    train_slice = train_df[config['target_col']].values[-300:]
    plot_sequence_diversity(train_slice, target_gen.values, ASSET, report_dir)
    
    # Financial Stylized Facts
    real_ret = pd.Series(target_real.values).pct_change().dropna()
    gen_ret = pd.Series(target_gen.values).pct_change().dropna()
    plot_distribution_diagnostics(real_ret, gen_ret, ASSET, report_dir)
    plot_financial_diagnostics(target_real.values, target_gen.values, ASSET, report_dir)
    
    logging.info(f"Validation complete. Outputs in {report_dir}")
    print(f"Validation complete. Outputs in {report_dir}")

if __name__ == "__main__":
    main()
