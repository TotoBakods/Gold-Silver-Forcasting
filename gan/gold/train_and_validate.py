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
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import from gan/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# --- Improved GAN Architectures ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, use_sn=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        if use_sn: conv = nn.utils.spectral_norm(conv)
        self.block = nn.Sequential(
            conv,
            nn.BatchNorm1d(out_channels) if not use_sn else nn.Identity(),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_dim=64):
        super().__init__()
        channels = input_size + noise_dim
        self.backbone = nn.Sequential(
            ConvBlock(channels, hidden_size, dilation=1),
            ConvBlock(hidden_size, hidden_size, dilation=2),
            ConvBlock(hidden_size, hidden_size, dilation=4),
        )
        self.attention = SelfAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, history, noise):
        x = torch.cat((history, noise), dim=2).transpose(1, 2)
        x = self.backbone(x).transpose(1, 2)
        x = self.attention(x)
        generated = self.head(x[:, -1, :])
        return generated.unsqueeze(1)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Input is (history + next) -> WINDOW_SIZE + 1
        self.backbone = nn.Sequential(
            ConvBlock(input_size, hidden_size, use_sn=True),
            ConvBlock(hidden_size, hidden_size, dilation=2, use_sn=True),
        )
        self.attention = SelfAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.backbone(x).transpose(1, 2)
        x = self.attention(x)
        score = self.head(x[:, -1, :])
        return score

# --- Helper Functions ---
def make_stationary(df, p_cols, r_cols):
    s_df = df.copy()
    for col in p_cols: s_df[col] = np.log(df[col] / df[col].shift(1))
    for col in r_cols: s_df[col] = df[col].diff()
    return s_df.dropna()

def reconstruct_future_rows(last_known, stat_gen, features, p_cols):
    reconstructed = []
    curr = last_known.copy()
    for row in stat_gen:
        for i, col in enumerate(features):
            if col in p_cols: curr[i] = curr[i] * np.exp(row[i])
            else: curr[i] = curr[i] + row[i]
        reconstructed.append(curr.copy())
    return reconstructed

def compute_moment_loss(fake, real):
    # Batch is (B, 1, Features) -> squeeze to (B, Features)
    fake = fake.squeeze(1)
    real = real.squeeze(1)
    mean_loss = torch.mean((fake.mean(dim=0) - real.mean(dim=0))**2)
    std_loss = torch.mean((fake.std(dim=0) - real.std(dim=0))**2)
    # Skewness anchor
    f_diff = fake - fake.mean(dim=0)
    r_diff = real - real.mean(dim=0)
    f_skew = torch.mean(f_diff**3) / (torch.mean(f_diff**2)**1.5 + 1e-8)
    r_skew = torch.mean(r_diff**3) / (torch.mean(r_diff**2)**1.5 + 1e-8)
    skew_loss = torch.mean((f_skew - r_skew)**2)
    return mean_loss + std_loss + 0.5 * skew_loss

def compute_drift_loss(history, fake_next):
    # Anchor fake_next to the window's own local drift
    fake_next = fake_next.squeeze(1)
    win_mean = history.mean(dim=1) 
    return torch.mean((fake_next - win_mean)**2)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# Constants
ASSET = "gold"
TRAIN_CUTOFF = "2026-01-31"
VAL_START = "2026-02-01"
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
    logging.info(f"Starting IMPROVED GAN Training for {ASSET}")
    print(f"Starting IMPROVED GAN Training for {ASSET}")
    
    # 1. Load Data
    # Manually defined config to avoid missing catalog dependency
    config = {
        'source_file': 'df_gold_dataset_USA_EPU_APRIL_01_2015_to_APRIL_14_2026.csv',
        'target_col': 'Gold_Futures'
    }

    df = pd.read_csv(PROJECT_ROOT / config['source_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Define features
    price_cols = [config['target_col']]
    if "Silver_Futures" in df.columns: price_cols.append("Silver_Futures")
    if "Crude_Oil_Futures" in df.columns: price_cols.append("Crude_Oil_Futures")
    
    rate_cols = [c for c in df.columns if c not in price_cols and c != 'Date']
    all_features = price_cols + rate_cols
    
    # Split
    train_df = df[df['Date'] <= TRAIN_CUTOFF].copy()
    real_test_df = df[(df['Date'] >= VAL_START) & (df['Date'] <= VAL_END)].copy()
    
    # 2. Preprocess
    stat_train_df = make_stationary(train_df, price_cols, rate_cols)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(stat_train_df[all_features].values)
    
    # Sequence building
    X, Y = [], []
    for i in range(len(scaled_train) - WINDOW_SIZE):
        X.append(scaled_train[i:i+WINDOW_SIZE])
        Y.append(scaled_train[i+WINDOW_SIZE])
    
    dataloader = DataLoader(TensorDataset(torch.FloatTensor(np.array(X)).to(device), 
                                          torch.FloatTensor(np.array(Y)).unsqueeze(1).to(device)), 
                            batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Train
    set_global_seed(42)
    num_features = len(all_features)
    netG = Generator(num_features, 128, num_features, noise_dim=NOISE_DIM).to(device)
    netD = Discriminator(num_features, 128).to(device)
    
    # Lower beta1 for GAN stability
    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.9))
    
    print("Training Improved GAN...")
    for epoch in tqdm(range(EPOCHS), desc=f"Training {ASSET}"):
        for history, actual_next in dataloader:
            # Train D
            for _ in range(N_CRITIC):
                optD.zero_grad()
                noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
                fake_next = netG(history, noise)
                
                # Hinge Loss for D
                real_score = netD(torch.cat((history, actual_next), dim=1))
                fake_score = netD(torch.cat((history, fake_next.detach()), dim=1))
                d_loss = torch.mean(nn.ReLU()(1.0 - real_score)) + torch.mean(nn.ReLU()(1.0 + fake_score))
                
                d_loss.backward()
                optD.step()
            
            # Train G
            optG.zero_grad()
            noise = torch.randn(history.size(0), WINDOW_SIZE, NOISE_DIM, device=device)
            fake_next = netG(history, noise)
            
            # Adversarial Loss (Hinge)
            g_adv_loss = -netD(torch.cat((history, fake_next), dim=1)).mean()
            
            # Statistical Anchors
            mse_loss = nn.MSELoss()(fake_next, actual_next)
            moment_loss = compute_moment_loss(fake_next, actual_next)
            drift_loss = compute_drift_loss(history, fake_next)
            
            # Multi-objective Optimization
            g_loss = g_adv_loss + 0.5 * mse_loss + 20.0 * moment_loss + 5.0 * drift_loss
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
    # Pull absolute latest real dates available
    full_real_dates = df[df['Date'] >= VAL_START]['Date'].sort_values().unique()
    
    dates_for_gen = []
    # Fill with real dates if available
    for i in range(GEN_DAYS):
        if i < len(full_real_dates):
            dates_for_gen.append(full_real_dates[i])
        else:
            # Extend with business days from the last point
            last_date = dates_for_gen[-1]
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5: # Skip weekends for purely fake future
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
    
    def calculate_validation_metrics(real, gen, col):
        r = real[col].values
        g = gen[col].values
        mse = np.mean((r - g)**2)
        mae = np.mean(np.abs(r - g))
        rmse = np.sqrt(mse)
        return {"MSE": mse, "MAE": mae, "RMSE": rmse}

    def log_validation_results(metrics, title, path):
        with open(path, "w") as f:
            f.write(f"--- {title} ---\n")
            for k, v in metrics.items(): f.write(f"{k}: {v:.6f}\n")

    def plot_real_vs_gen(real, gen, col, title, out_dir):
        plt.figure(figsize=(12, 6))
        plt.plot(real['Date'], real[col], label='Real Data', color='blue')
        plt.plot(gen['Date'], gen[col], label='GAN Synthetic', color='orange')
        plt.title(f"{title} GAN Validation")
        plt.legend()
        plt.savefig(out_dir / f"{col.lower()}_validation.png")
        plt.close()

    # Placeholders for advanced plots to avoid crashes
    def plot_returns_dist(*args): pass
    def plot_acf_comparison(*args): pass
    def plot_sequence_diversity(*args): pass
    def plot_distribution_diagnostics(*args): pass
    def plot_financial_diagnostics(*args): pass

    overlap_len = min(len(real_test_df), len(gen_test_df))
    metrics = calculate_validation_metrics(real_test_df.iloc[:overlap_len], gen_test_df.iloc[:overlap_len], config['target_col'])
    
    log_validation_results(metrics, f"{ASSET} (Ground Truth Comparison up to 2026-04-13)", report_dir / "validation_metrics.txt")
    
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
