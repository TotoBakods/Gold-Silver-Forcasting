import pandas as pd
import numpy as np
import torch
import pickle
import os
from sklearn.preprocessing import StandardScaler
from models import CNN_BiLSTM, Generator

# Configuration
ASSET = "gold"
CONFIG = {
    "target_col": "Gold_Futures",
    "raw_csv": "df_gold_dataset_USA_EPU_APRIL_01_2015_to_APRIL_14_2026.csv",
    "train_end": "2026-01-31",
    "test_start": "2026-02-01",
    "gan_seed_end": "2026-02-28",
    "forecast_end": "2026-05-31",
    "gan_path": "gan/gold/gold_val_gen.pth",
    "features": ['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'USA_EPU', 'DFF', 'gpr_daily'],
    "gan_window": 15,
    "gan_noise": 64,
    "gan_hidden": 128
}

def make_stationary(df_in, p_cols):
    s_df = df_in.copy()
    for col in p_cols: s_df[col] = np.log(df_in[col] / df_in[col].shift(1))
    return s_df.dropna()

device = torch.device("cpu")

df = pd.read_csv(CONFIG["raw_csv"])
df['Date'] = pd.to_datetime(df['Date'])
df[CONFIG["features"]] = df[CONFIG["features"]].ffill().bfill()
df = df.dropna(subset=CONFIG["features"]).reset_index(drop=True)

train_df = df[df['Date'] <= CONFIG["train_end"]].copy()

p_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
price_cols = [c for c in CONFIG["features"] if any(kw in c for kw in p_keywords)]

netG = Generator(len(CONFIG["features"]), CONFIG["gan_hidden"], len(CONFIG["features"]), CONFIG["gan_noise"]).to(device)
netG.load_state_dict(torch.load(CONFIG["gan_path"], map_location=device, weights_only=True))
netG.eval()

# Phase 1: Warm-up
hist_stat_df = make_stationary(train_df[CONFIG["features"]], price_cols)
scaler_gan = StandardScaler()
scaler_gan.fit(hist_stat_df.values)

scaled_train_stat = scaler_gan.transform(hist_stat_df.values)
win = torch.FloatTensor(scaled_train_stat[-CONFIG["gan_window"]:]).unsqueeze(0).to(device)

warmup_df = df[(df['Date'] >= CONFIG["test_start"]) & (df['Date'] <= CONFIG["gan_seed_end"])].reset_index(drop=True)
warmup_with_prev = df[df['Date'] <= CONFIG["gan_seed_end"]].tail(len(warmup_df) + 1)[CONFIG["features"]].copy()
warmup_stat_df = make_stationary(warmup_with_prev, price_cols)
scaled_warmup = scaler_gan.transform(warmup_stat_df.values)

for step_row in scaled_warmup:
    real_step = torch.FloatTensor(step_row).reshape(1, 1, -1).to(device)
    win = torch.cat((win[:, 1:, :], real_step), dim=1)

# Phase 2: Generation
free_dates = pd.date_range(start=warmup_df['Date'].iloc[-1] + pd.offsets.BDay(1), end=CONFIG["forecast_end"], freq='B')
gan_stat_gen = []
for i in range(len(free_dates)):
    with torch.no_grad():
        noise = torch.randn(1, CONFIG["gan_window"], CONFIG["gan_noise"], device=device)
        next_stat = netG(win, noise)
        gan_stat_gen.append(next_stat.cpu().numpy()[0, 0, :])
        win = torch.cat((win[:, 1:, :], next_stat), dim=1)
        if np.any(np.isnan(next_stat.numpy())):
            print(f"NaN at step {i}")
            break

gan_stat = scaler_gan.inverse_transform(np.array(gan_stat_gen))
print(f"Generated {len(gan_stat)} steps")
print(f"Max return: {np.max(gan_stat):.4f}")
print(f"Min return: {np.min(gan_stat):.4f}")
print(f"Last 5 Gold prices (approx):")
last_vals = warmup_df[CONFIG["features"]].iloc[-1].values.copy()
for i, row in enumerate(gan_stat[:5]):
    for j, col in enumerate(CONFIG["features"]):
        if col in price_cols: last_vals[j] *= np.exp(row[j])
        else: last_vals[j] += row[j]
    print(f"Step {i}: {last_vals[0]:.2f}")
