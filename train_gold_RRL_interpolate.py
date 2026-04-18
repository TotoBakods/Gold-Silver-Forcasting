import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging
from tqdm import tqdm
from models import CNN_BiLSTM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
LOOKBACK = 30
HIDDEN_DIM = 64
FILTERS = 128
KERNEL_SIZE = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 500
DROPOUT = 0.2
COUNCIL_SIZE = 10

# GAN Hyperparameters (must match gan/generate_gan_data.py)
GAN_WINDOW_SIZE = 15
GAN_NOISE_DIM = 64
GAN_HIDDEN_SIZE = 128
GAN_DROPOUT = 0.10

# Date Constraints
TRAIN_END_DATE = "2026-01-31"
TEST_START_DATE = "2026-02-01"
ACTUAL_DATA_END = "2026-04-13"
FORECAST_END_DATE = "2026-05-31"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CNN_BiLSTM moved to models.py

# --- GAN Generator Architecture (Copied from gan/generate_gan_data.py) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        channels = input_size + GAN_NOISE_DIM
        self.backbone = nn.Sequential(
            ConvBlock(channels, hidden_size, dilation=1),
            ConvBlock(hidden_size, hidden_size, dilation=2),
            ConvBlock(hidden_size, hidden_size, dilation=4),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(GAN_DROPOUT),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, history, noise):
        x = torch.cat((history, noise), dim=2).transpose(1, 2)
        hidden = self.backbone(x).transpose(1, 2)
        generated = self.head(hidden[:, -1, :])
        return generated.unsqueeze(1)

def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        # Predict the return for the step immediately following the window end
        y.append(target[i + lookback - 1])
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_test, y_test, input_dim, seed):
    set_seed(seed)
    model = CNN_BiLSTM(input_dim, HIDDEN_DIM, FILTERS, KERNEL_SIZE, 1, DROPOUT).to(device)
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_y_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Seed {seed}")
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_X_t)
                test_loss = criterion(test_outputs, test_y_t.unsqueeze(1))
                pbar.set_postfix({"Train Loss": f"{epoch_loss/len(train_loader):.4f}", "Test Loss": f"{test_loss.item():.4f}"})
                if epoch % 50 == 0:
                    logger.info(f"Seed {seed} | Epoch {epoch} | Train Loss {epoch_loss/len(train_loader):.6f} | Test Loss {test_loss.item():.6f}")
                
    return model

def add_indicators(df, target_col):
    df = df.copy()
    # Use relative indicators to ensure scale invariance
    df['EMA_Fast'] = (df[target_col].ewm(span=3, adjust=False).mean() / df[target_col]) - 1
    df['EMA_Slow'] = (df[target_col].ewm(span=8, adjust=False).mean() / df[target_col]) - 1
    
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))
    
    exp1 = df[target_col].ewm(span=6, adjust=False).mean()
    exp2 = df[target_col].ewm(span=13, adjust=False).mean()
    df['MACD_Flash'] = (exp1 - exp2) / df[target_col] # Relative MACD
    df['MACD_Signal'] = df['MACD_Flash'].ewm(span=5, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Flash'] - df['MACD_Signal']
    
    df['BB_Mid'] = df[target_col].rolling(window=5).mean()
    df['BB_Std'] = df[target_col].rolling(window=5).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    
    # Volatility and Volume Indicators
    df['ATR'] = (df[target_col].rolling(window=7).max() - df[target_col].rolling(window=7).min()) / df[target_col]
    if 'Volume' in df.columns:
        df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1e-8)
    else:
        df['Vol_Ratio'] = 1.0
        
    df['GS_Ratio'] = df['Gold_Futures'] / (df['Silver_Futures'] + 1e-8) if 'Gold_Futures' in df.columns and 'Silver_Futures' in df.columns else 0.0
        
    return df.ffill().bfill().fillna(0)

def main():
    logger.info("Starting Gold Price Training & Strictly Out-of-Sample Evaluation")
    
    # Load raw data
    raw_df = pd.read_csv("df_gold_dataset_USA_EPU_APRIL_01_2015_to_APRIL_14_2026.csv")
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    # Configuration
    features = ['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'USA_EPU', 'DFF', 'gpr_daily']
    target_col = 'Gold_Futures'
    
    # 1. Clean data and handle NaNs globally BEFORE splitting
    # This ensures technical indicators and continuity are preserved
    raw_df[features] = raw_df[features].ffill().bfill()
    raw_df = raw_df.dropna(subset=features).reset_index(drop=True)
    
    # Add Technical Indicators to full dataset
    df = add_indicators(raw_df, target_col)
    
    tech_cols = ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ATR', 'Vol_Ratio', 'GS_Ratio']
    all_features = features + tech_cols
    
    # Target: Percentage Return for next day
    df['target_return'] = df[target_col].pct_change().shift(-1).fillna(0)
    
    # Strictly Out-of-Sample Split
    train_df = df[df['Date'] <= TRAIN_END_DATE].copy()
    actual_test_df = df[(df['Date'] >= TEST_START_DATE) & (df['Date'] <= ACTUAL_DATA_END)].copy()
    
    # Re-verify continuity
    last_train_p = train_df[target_col].iloc[-1]
    first_test_p = actual_test_df[target_col].iloc[0]
    logger.info(f"Boundary check: Last Train Price={last_train_p:.2f}, First Test Price={first_test_p:.2f}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(train_df[all_features])
    scaler_y.fit(train_df[['target_return']])
    
    train_X_scaled = scaler_X.transform(train_df[all_features])
    train_y_scaled = scaler_y.transform(train_df[['target_return']]).flatten()
    
    # For testing, we include the last LOOKBACK rows of training data to allow predicting from Feb 1
    test_period_df = pd.concat([train_df.tail(LOOKBACK), actual_test_df]).copy()
    test_X_scaled = scaler_X.transform(test_period_df[all_features])
    test_y_scaled = scaler_y.transform(test_period_df[['target_return']]).flatten()
    
    # Create Sequences
    X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, LOOKBACK)
    X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, LOOKBACK)
    
    # Council Training
    SAVE_DIR = "models/gold_RRL_interpolate"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Train Ensemble Council
    seeds = [42, 123, 99, 7, 88, 555, 13, 21, 666, 10]
    models_list = []
    for s in seeds[:COUNCIL_SIZE]:
        model_path = os.path.join(SAVE_DIR, f"gold_model_seed_{s}.pth")
        model = CNN_BiLSTM(len(all_features), HIDDEN_DIM, FILTERS, KERNEL_SIZE, 1, DROPOUT).to(device)
        if os.path.exists(model_path):
            logger.info(f"--- Loading Existing Council Member: Seed {s} ---")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.info(f"--- Training Council Member: Seed {s} ---")
            model = train_model(X_train, y_train, X_test, y_test, len(all_features), s)
            torch.save(model.state_dict(), model_path)
        models_list.append(model)

    # Save shared assets
    with open(os.path.join(SAVE_DIR, "scaler_X.pkl"), "wb") as f: pickle.dump(scaler_X, f)
    with open(os.path.join(SAVE_DIR, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)

    # --- GAN Path Generation & Forecasting Integration ---
    logger.info("--- Starting GAN Integration ---")
    
    # Load GAN model
    gan_path = "gan/gold/gold_val_gen.pth"
    num_features = len(features)
    netG = Generator(num_features, GAN_HIDDEN_SIZE, num_features).to(device)
    netG.load_state_dict(torch.load(gan_path, map_location=device))
    netG.eval()
    
    # GAN requires stationary data for generation. 
    def make_stationary(df, p_cols, r_cols):
        s_df = df.copy()
        for col in p_cols: s_df[col] = np.log(df[col] / df[col].shift(1))
        for col in r_cols: s_df[col] = df[col].diff()
        return s_df.dropna()

    p_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100']
    price_cols = [c for c in features if any(kw in c for kw in p_keywords)]
    rate_cols = [c for c in features if c not in price_cols]
    
    # Prepare last window for GAN
    hist_stat_df = make_stationary(train_df[features], price_cols, rate_cols)
    scaler_gan = StandardScaler()
    scaled_stat = scaler_gan.fit_transform(hist_stat_df.values)
    current_window = torch.FloatTensor(scaled_stat[-GAN_WINDOW_SIZE:]).unsqueeze(0).to(device)
    
    future_dates = pd.date_range(start=TEST_START_DATE, end=FORECAST_END_DATE, freq='B')
    last_known_vals = train_df[features].iloc[-1].values.copy()
    
    # Generate GAN Path
    gan_stat_gen = []
    win = current_window.clone()
    for _ in range(len(future_dates)):
        with torch.no_grad():
            noise = torch.randn(1, GAN_WINDOW_SIZE, GAN_NOISE_DIM, device=device)
            next_stat = netG(win, noise)
            gan_stat_gen.append(next_stat.cpu().numpy()[0, 0, :])
            win = torch.cat((win[:, 1:, :], next_stat), dim=1)
    
    gan_stat = scaler_gan.inverse_transform(np.array(gan_stat_gen))
    # Clip returns to prevent price explosion (e.g., max 5% move per day is much safer for long paths)
    gan_stat = np.clip(gan_stat, -0.05, 0.05)
    
    # Reconstruct Price Path from GAN stationary returns
    gan_prices = []
    curr = last_known_vals.copy()
    for row in gan_stat:
        for i, col in enumerate(features):
            if col in price_cols:
                curr[i] = curr[i] * np.exp(row[i])
            else:
                curr[i] = curr[i] + row[i]
        gan_prices.append(curr.copy())
    
    gan_df = pd.DataFrame(gan_prices, columns=features)
    gan_df = gan_df.ffill().bfill().fillna(0) # Safety
    gan_df['Date'] = future_dates
    gan_df.to_csv(os.path.join(SAVE_DIR, "gan_forecast_data.csv"), index=False)
    
    # Training complete. Reporting is now handled by generate_reports.py
    logger.info("Gold Council Training Complete.")
    logger.info("To generate diagnostic reports, run: python generate_reports.py --asset gold")

if __name__ == "__main__":
    main()
