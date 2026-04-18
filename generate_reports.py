import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from sklearn.preprocessing import StandardScaler
from models import CNN_BiLSTM, Generator

# --- Configuration ---
ASSET_CONFIGS = {
    "gold": {
        "target_col": "Gold_Futures",
        "raw_csv": "df_gold_dataset_USA_EPU_APRIL_01_2015_to_APRIL_14_2026.csv",
        "train_end": "2026-01-31",
        "test_start": "2026-02-01",
        "gan_seed_end": "2026-02-28",  # Matches gan/gold training cutoff — closer regime to test period
        "data_end": "2026-04-13",
        "forecast_end": "2026-05-31",
        "model_dir": "models/gold_RRL_interpolate",
        "gan_path": "gan/gold/gold_val_gen.pth",
        "seeds": [42, 123, 99, 7, 88, 555, 13, 21, 666, 10],
        "features": ['Gold_Futures', 'Silver_Futures', 'Crude_Oil_Futures', 'UST10Y_Treasury_Yield', 'USA_EPU', 'DFF', 'gpr_daily'],
        "tech_cols": ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'MACD_Signal', 'MACD_Hist', 'BB_Width', 'ATR', 'Vol_Ratio', 'GS_Ratio'],
        "n_layers": 1,
        "hidden_dim": 64,
        "filters": 128,
        "kernel_size": 5,
        "lookback": 30,
        "gan_window": 15,
        "gan_noise": 64,
        "gan_hidden": 128,
        "report_name": "gold_forecast_vs_gan_vs_actual.png"
    },
    "silver": {
        "target_col": "Silver_Futures",
        "raw_csv": "df_silver_dataset_APRIL_01_2015_to_APRIL_14_2026.csv",
        "train_end": "2026-01-31",
        "test_start": "2026-02-01",
        "gan_seed_end": "2026-02-28",  # Matches gan/silver training cutoff — closer regime to test period
        "data_end": "2026-04-13",
        "forecast_end": "2026-05-31",
        "model_dir": "models/silver_RRL_interpolate",
        "gan_path": "gan/silver/silver_val_gen.pth",
        "seeds": [42, 123, 99, 7, 88, 555, 13, 21, 666, 10],
        "features": ['Silver_Futures', 'Gold_Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index'],
        "tech_cols": ['EMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'Vol_Ratio', 'GS_Ratio'],
        "n_layers": 2,
        "hidden_dim": 64,
        "filters": 128,
        "kernel_size": 5,
        "lookback": 30,
        "gan_window": 15,
        "gan_noise": 64,
        "gan_hidden": 128,
        "report_name": "silver_forecast_vs_gan_vs_actual.png"
    }
}

def add_indicators(df, target_col):
    df = df.copy()
    # Basic technicals
    df['EMA_Fast'] = (df[target_col].ewm(span=3, adjust=False).mean() / df[target_col]) - 1
    df['EMA_Slow'] = (df[target_col].ewm(span=8, adjust=False).mean() / df[target_col]) - 1
    
    if 'Silver_Futures' in df.columns:
        df['EMA_10'] = (df['Silver_Futures'].ewm(span=10, adjust=False).mean() / df['Silver_Futures']) - 1
        df['EMA_20'] = (df['Silver_Futures'].ewm(span=20, adjust=False).mean() / df['Silver_Futures']) - 1
        
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))
    
    gain14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs14 = gain14 / (loss14 + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs14))

    exp1 = df[target_col].ewm(span=6, adjust=False).mean()
    exp2 = df[target_col].ewm(span=13, adjust=False).mean()
    df['MACD_Flash'] = (exp1 - exp2) / (df[target_col] + 1e-8)
    df['MACD_Signal'] = df['MACD_Flash'].ewm(span=5, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Flash'] - df['MACD_Signal']
    df['MACD'] = df['MACD_Flash']
    
    df['BB_Mid'] = df[target_col].rolling(window=5).mean()
    df['BB_Std'] = df[target_col].rolling(window=5).std()
    df['BB_Width'] = (4 * df['BB_Std']) / (df['BB_Mid'] + 1e-8)
    
    df['ATR'] = (df[target_col].rolling(window=7).max() - df[target_col].rolling(window=7).min()) / df[target_col]
    if 'Volume' in df.columns:
        df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1e-8)
    else:
        df['Vol_Ratio'] = 1.0
        
    df['ROC_2'] = df[target_col].pct_change(periods=2)
    df['ROC_5'] = df[target_col].pct_change(periods=5)
    
    if 'Silver_Futures' in df.columns and 'Gold_Futures' in df.columns:
        df['GS_Ratio'] = df['Gold_Futures'] / (df['Silver_Futures'] + 1e-8)
    else:
        df['GS_Ratio'] = 0.0
        
    return df.ffill().bfill().fillna(0)

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic reports for Gold and Silver models.")
    parser.add_argument("--asset", type=str, choices=["gold", "silver"], required=True, help="Asset to generate report for.")
    args = parser.parse_args()
    
    asset = args.asset
    config = ASSET_CONFIGS[asset]
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    logger.info(f"--- Generating Comprehensive Report for {asset.upper()} ---")
    df = pd.read_csv(config["raw_csv"])
    df['Date'] = pd.to_datetime(df['Date'])
    df[config["features"]] = df[config["features"]].ffill().bfill()
    df = df.dropna(subset=config["features"]).reset_index(drop=True)
    df_with_inds = add_indicators(df, config["target_col"])
    
    train_df = df_with_inds[df_with_inds['Date'] <= config["train_end"]].copy()
    test_df = df_with_inds[(df_with_inds['Date'] >= config["test_start"]) & (df_with_inds['Date'] <= config["data_end"])].copy()
    
    target_col = config["target_col"]
    features = config["features"]
    all_features = config["features"] + config["tech_cols"]
    
    # 2. Load Models & Scalers
    with open(os.path.join(config["model_dir"], "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(config["model_dir"], "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
        
    models = []
    for s in config["seeds"]:
        path = os.path.join(config["model_dir"], f"{asset}_model_seed_{s}.pth")
        if os.path.exists(path):
            model = CNN_BiLSTM(len(all_features), config["hidden_dim"], config["filters"], config["kernel_size"], config["n_layers"]).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            models.append(model)
    logger.info(f"Loaded {len(models)} models for ensemble.")
    
    # 3. GAN Integration
    logger.info("Generating GAN Synthetic Path...")
    netG = Generator(len(features), config["gan_hidden"], len(features), config["gan_noise"]).to(device)
    if os.path.exists(config["gan_path"]):
        netG.load_state_dict(torch.load(config["gan_path"], map_location=device, weights_only=True))
        netG.eval()
    else:
        logger.warning(f"GAN model not found at {config['gan_path']}. Skipping GAN section.")
        netG = None

    # --- GAN Warm-Up + Free Generation ---
    # Phase 1 (Warm-up): seed at Jan 31, feed real Feb 1-Feb 28 data step-by-step
    #   → GAN window lands in the real Feb regime, no wrong starting price level
    # Phase 2 (Free): generate from Mar 1 → forecast_end
    # Combined: gan_df covers Feb 1 → forecast_end, matching the actual forecast start
    gan_df = None
    if netG:
        # Stationary prep for GAN
        def make_stationary(df_in, p_cols):
            s_df = df_in.copy()
            for col in p_cols: s_df[col] = np.log(df_in[col] / df_in[col].shift(1))
            return s_df.dropna()

        p_keywords = ['Futures', 'US30', 'SnP500', 'NASDAQ_100', 'USD_index']
        price_cols = [c for c in features if any(kw in c for kw in p_keywords)]
        rate_cols  = [c for c in features if c not in price_cols]

        # Fit scaler on train only
        hist_stat_df = make_stationary(train_df[features], price_cols)
        scaler_gan = StandardScaler()
        scaler_gan.fit(hist_stat_df.values)

        # --- Phase 1: warm-up window with real Feb data ---
        # Seed: last gan_window rows of stationary train (up to Jan 31)
        scaled_train_stat = scaler_gan.transform(hist_stat_df.values)
        win = torch.FloatTensor(scaled_train_stat[-config["gan_window"]:]).unsqueeze(0).to(device)

        # Warm-up dataframe: real data from test_start through gan_seed_end
        warmup_df = df_with_inds[
            (df_with_inds['Date'] >= config["test_start"]) &
            (df_with_inds['Date'] <= config["gan_seed_end"])
        ].reset_index(drop=True)

        # Build stationary returns for the warm-up period (need one row before for diff)
        warmup_with_prev = df_with_inds[
            df_with_inds['Date'] <= config["gan_seed_end"]
        ].tail(len(warmup_df) + 1)[features].copy()
        warmup_stat_df = make_stationary(warmup_with_prev, price_cols)   # len == len(warmup_df)
        scaled_warmup = scaler_gan.transform(warmup_stat_df.values)

        # Slide real Feb rows through the window without collecting outputs
        for step_row in scaled_warmup:
            real_step = torch.FloatTensor(step_row).reshape(1, 1, -1).to(device)
            win = torch.cat((win[:, 1:, :], real_step), dim=1)

        # Reconstruct warmup prices directly from real data
        warmup_prices = warmup_df[features].values.tolist()
        warmup_dates  = warmup_df['Date'].values

        # --- Phase 2: free generation from Mar 1 → forecast_end ---
        free_dates = pd.date_range(
            start=warmup_df['Date'].iloc[-1] + pd.offsets.BDay(1),
            end=config["forecast_end"], freq='B'
        )
        gan_stat_gen = []
        for _ in range(len(free_dates)):
            with torch.no_grad():
                noise = torch.randn(1, config["gan_window"], config["gan_noise"], device=device)
                next_stat = netG(win, noise)
                gan_stat_gen.append(next_stat.cpu().numpy()[0, 0, :])
                win = torch.cat((win[:, 1:, :], next_stat), dim=1)

        gan_stat = scaler_gan.inverse_transform(np.array(gan_stat_gen))
        gan_stat = np.clip(gan_stat, -0.05, 0.05)  # Safety clip

        # Reconstruct free-gen prices starting from last real warmup price
        last_vals = warmup_df[features].iloc[-1].values.copy()
        free_prices = []
        for row in gan_stat:
            for i, col in enumerate(features):
                if col in price_cols: last_vals[i] *= np.exp(row[i])
                else:                 last_vals[i] += row[i]
            free_prices.append(last_vals.copy())

        # --- Combine Phase 1 + Phase 2 ---
        all_prices = warmup_prices + free_prices
        all_dates  = list(warmup_dates) + list(free_dates)
        gan_df = pd.DataFrame(all_prices, columns=features)
        gan_df['Date'] = all_dates
        gan_df = add_indicators(gan_df, target_col)

        # Update future_dates to full span (used later for GAN-based forecast)
        future_dates = pd.DatetimeIndex(all_dates)
    
    # 4. Forecasting Logic
    lookback = config["lookback"]
    def run_forecast(base_df, source_df):
        full_df = pd.concat([base_df, source_df], ignore_index=True)
        X_scaled = scaler_X.transform(full_df[all_features])
        start_idx = len(base_df)
        prices = []
        for i in range(len(source_df) + 1):
            win = X_scaled[start_idx + i - lookback : start_idx + i]
            if len(win) < lookback: continue
            win_t = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = [m(win_t).item() for m in models]
                avg_ret = scaler_y.inverse_transform([[np.mean(preds)]])[0,0]
            prev_p = full_df[target_col].iloc[start_idx + i - 1]
            prices.append(prev_p * (1 + avg_ret))
        return prices

    forecast_actual = run_forecast(train_df, test_df)
    forecast_gan = run_forecast(train_df, gan_df) if gan_df is not None else None
    
    # 5. Alignment
    pred_dates_act = (pd.to_datetime(test_df['Date'].tolist() + [test_df['Date'].iloc[-1] + pd.offsets.BDay(1)]) + pd.offsets.BDay(1)).values
    actual_pred_df = pd.DataFrame({'Date': pred_dates_act, 'Forecast': forecast_actual})
    
    if gan_df is not None:
        pred_dates_gan = (pd.to_datetime(future_dates.tolist() + [future_dates[-1] + pd.offsets.BDay(1)]) + pd.offsets.BDay(1)).values
        gan_pred_df = pd.DataFrame({'Date': pred_dates_gan, 'Forecast': forecast_gan})
    
    # 6. Visualization
    plt.figure(figsize=(15, 8))
    hist_tail = train_df[train_df['Date'] >= "2025-11-01"]
    plt.plot(hist_tail['Date'], hist_tail[target_col], label='Historical Data', color='black', alpha=0.3)
    
    test_dates_shifted = (pd.to_datetime(test_df['Date']) + pd.offsets.BDay(1)).values
    plt.plot(test_dates_shifted, test_df[target_col], label='Actual Data (Confirmed)', color='blue', linewidth=2.5)
    
    # Prediction: Dotted green
    plt.plot(actual_pred_df['Date'], actual_pred_df['Forecast'], label='Forecast (Actual-based)', color='green', linewidth=2, linestyle=':')
    
    if gan_df is not None:
        plt.plot(gan_df['Date'], gan_df[target_col], label='GAN Synthetic Path', color='orange', linewidth=1.8, alpha=0.75)
        # GAN Prediction: Dotted red
        plt.plot(gan_pred_df['Date'], gan_pred_df['Forecast'], label='Forecast (GAN-based)', color='red', linewidth=2, linestyle=':')
    
    plt.axvline(pd.to_datetime(config["test_start"]), color='red', linestyle='--', alpha=0.5, label='OOS Start')
    plt.axvline(pd.to_datetime(config["data_end"]), color='gray', linestyle=':', alpha=0.5, label='Actual Data End')
    
    plt.title(f"{asset.upper()} Comprehensive Forecast Alignment Report", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", config["report_name"])
    plt.savefig(out_path)
    logger.info(f"Report saved to {out_path}")
    
    # Metrics
    common_len = len(test_df)
    y_true = test_df[target_col].values
    y_pred = actual_pred_df['Forecast'].values[:common_len]
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    logger.info(f"Forecast RMSE (strictly aligned): {rmse:.4f}")

if __name__ == "__main__":
    main()
