import pandas as pd
import numpy as np
import torch
import pickle
import os
from sklearn.preprocessing import StandardScaler
from models import CNN_BiLSTM, Generator

# Configuration
ASSET = "silver"
CONFIG = {
    "target_col": "Silver_Futures",
    "raw_csv": "df_silver_dataset_APRIL_01_2015_to_APRIL_14_2026.csv",
    "train_end": "2026-01-31",
    "test_start": "2026-02-01",
    "gan_seed_end": "2026-02-28",
    "forecast_end": "2026-05-31",
    "gan_path": "gan/silver/silver_val_gen.pth",
    "features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index'],
    "gan_window": 15,
    "gan_noise": 64,
    "gan_hidden": 128
}

def make_stationary(df_in, p_cols, r_cols):
    s_df = df_in.copy()
    for col in p_cols: s_df[col] = np.log(df_in[col] / df_in[col].shift(1))
    for col in r_cols: s_df[col] = df_in[col].diff()
    return s_df.dropna()

device = torch.device("cpu")

df = pd.read_csv(CONFIG["raw_csv"])
df['Date'] = pd.to_datetime(df['Date'])
df[CONFIG["features"]] = df[CONFIG["features"]].ffill().bfill()
df = df.dropna(subset=CONFIG["features"]).reset_index(drop=True)

train_df_for_scaler = df[df['Date'] <= CONFIG["gan_seed_end"]].copy()

p_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
price_cols = [c for c in CONFIG["features"] if any(kw in c for kw in p_keywords)]
rate_cols = [c for c in CONFIG["features"] if c not in price_cols]

netG = Generator(len(CONFIG["features"]), CONFIG["gan_hidden"], len(CONFIG["features"]), CONFIG["gan_noise"]).to(device)
netG.load_state_dict(torch.load(CONFIG["gan_path"], map_location=device, weights_only=True))
netG.eval()

hist_stat_df = make_stationary(train_df_for_scaler[CONFIG["features"]], price_cols, rate_cols)
scaler_gan = StandardScaler()
scaler_gan.fit(hist_stat_df.values)

train_df_jan = df[df['Date'] <= CONFIG["train_end"]].copy()
jan_stat_df = make_stationary(train_df_jan[CONFIG["features"]], price_cols, rate_cols)
scaled_train_stat = scaler_gan.transform(jan_stat_df.values)
win = torch.FloatTensor(scaled_train_stat[-CONFIG["gan_window"]:]).unsqueeze(0).to(device)

warmup_df = df[(df['Date'] >= CONFIG["test_start"]) & (df['Date'] <= CONFIG["gan_seed_end"])].reset_index(drop=True)
warmup_with_prev = df[df['Date'] <= CONFIG["gan_seed_end"]].tail(len(warmup_df) + 1)[CONFIG["features"]].copy()
warmup_stat_df = make_stationary(warmup_with_prev, price_cols, rate_cols)
scaled_warmup = scaler_gan.transform(warmup_stat_df.values)

for step_row in scaled_warmup:
    real_step = torch.FloatTensor(step_row).reshape(1, 1, -1).to(device)
    win = torch.cat((win[:, 1:, :], real_step), dim=1)

free_dates = pd.date_range(start=warmup_df['Date'].iloc[-1] + pd.offsets.BDay(1), end=CONFIG["forecast_end"], freq='B')
gan_stat_gen = []
for i in range(len(free_dates)):
    with torch.no_grad():
        noise = torch.randn(1, CONFIG["gan_window"], CONFIG["gan_noise"], device=device)
        next_stat = netG(win, noise)
        next_stat_clipped = torch.clamp(next_stat, -10.0, 10.0)
        gan_stat_gen.append(next_stat_clipped.cpu().numpy()[0, 0, :])
        win = torch.cat((win[:, 1:, :], next_stat_clipped), dim=1)

gan_stat = scaler_gan.inverse_transform(np.array(gan_stat_gen))
gan_stat = np.clip(gan_stat, -0.05, 0.05)

silver_returns = gan_stat[:, CONFIG["features"].index("Silver_Futures")]
print(f"Mean silver return: {np.mean(silver_returns):.6f}")
print(f"Std silver return: {np.std(silver_returns):.6f}")
print(f"Percentage of positive returns: {np.mean(silver_returns > 0) * 100:.2f}%")
