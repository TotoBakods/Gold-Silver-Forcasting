import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_START = pd.Timestamp(os.getenv("FORECAST_START", "2026-04-01"))
FORECAST_END = pd.Timestamp(os.getenv("FORECAST_END", "2026-05-31"))
VALIDATION_START = pd.Timestamp(os.getenv("VALIDATION_START", "2026-04-01"))
VALIDATION_END = pd.Timestamp(os.getenv("VALIDATION_END", "2026-04-13"))

COUNCIL_SEEDS = [42, 123, 99]


class CNN_BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, filters=64, kernel_size=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(filters)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)
        self.dropout = torch.nn.Dropout(dropout)
        lstm_dropout = dropout if n_layers > 1 else 0
        self.lstm = torch.nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, 64)
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if x.shape[-1] > 1:
            x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.out(x)


def add_gold_indicators(df, target_col):
    df = df.copy()
    df["EMA_Fast"] = df[target_col].ewm(span=3, adjust=False).mean()
    df["EMA_Slow"] = df[target_col].ewm(span=8, adjust=False).mean()
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_7"] = 100 - (100 / (1 + rs))
    exp1 = df[target_col].ewm(span=6, adjust=False).mean()
    exp2 = df[target_col].ewm(span=13, adjust=False).mean()
    df["MACD_Flash"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_Flash"].ewm(span=5, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Flash"] - df["MACD_Signal"]
    df["BB_Mid"] = df[target_col].rolling(window=5).mean()
    df["BB_Std"] = df[target_col].rolling(window=5).std()
    df["BB_Width"] = (4 * df["BB_Std"]) / (df["BB_Mid"] + 1e-8)
    df["ROC_2"] = df[target_col].pct_change(periods=2).replace([np.inf, -np.inf], 0).fillna(0)
    if "Silver_Futures" in df.columns and "Gold_Futures" in df.columns:
        df["GS_Ratio"] = df["Gold_Futures"] / (df["Silver_Futures"] + 1e-8)
    else:
        df["GS_Ratio"] = 0.0
    return df.ffill().bfill().fillna(0)


def add_silver_indicators(df, target_col):
    df = df.copy()
    df["EMA_10"] = df[target_col].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df[target_col].ewm(span=20, adjust=False).mean()
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    exp1 = df[target_col].ewm(span=12, adjust=False).mean()
    exp2 = df[target_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["BB_Mid"] = df[target_col].rolling(window=20).mean()
    df["BB_Std"] = df[target_col].rolling(window=20).std()
    df["BB_Width"] = (4 * df["BB_Std"]) / (df["BB_Mid"] + 1e-8)
    df["ROC_5"] = df[target_col].pct_change(periods=5).replace([np.inf, -np.inf], 0).fillna(0)
    if "Silver_Futures" in df.columns and "Gold_Futures" in df.columns:
        df["GS_Ratio"] = df["Gold_Futures"] / (df["Silver_Futures"] + 1e-8)
    else:
        df["GS_Ratio"] = 0.0
    return df.ffill().bfill().fillna(0)


def create_sequences(data, lookback):
    X = []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
    return np.array(X)


def load_council_models(model_dir, input_dim, params, prefix):
    models = []
    for seed in COUNCIL_SEEDS:
        model_path = Path(model_dir) / f"{prefix}_model_seed_{seed}.pth"
        if not model_path.exists():
            continue
        model = CNN_BiLSTM(
            input_dim,
            hidden_dim=params["hidden_dim"],
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models.append(model)
    return models


def build_predictions(asset_cfg):
    train_df = pd.read_csv(asset_cfg["train_csv"], parse_dates=["Date"])
    test_df = pd.read_csv(asset_cfg["test_csv"], parse_dates=["Date"])
    synth_df = pd.read_csv(asset_cfg["synthetic_csv"], parse_dates=["Date"])

    train_df["Date"] = pd.to_datetime(train_df["Date"]).dt.normalize()
    test_df["Date"] = pd.to_datetime(test_df["Date"]).dt.normalize()
    synth_df["Date"] = pd.to_datetime(synth_df["Date"]).dt.normalize()

    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)
    synth_df = synth_df.sort_values("Date").reset_index(drop=True)

    synth_future = synth_df[(synth_df["Date"] >= FORECAST_START) & (synth_df["Date"] <= FORECAST_END)].copy()
    if synth_future.empty:
        raise ValueError(f"Synthetic data missing for {asset_cfg['asset']} in forecast window.")

    combined = pd.concat([train_df, synth_future], ignore_index=True)

    if asset_cfg["asset"] == "gold":
        combined = add_gold_indicators(combined, asset_cfg["target_col"])
    else:
        combined = add_silver_indicators(combined, asset_cfg["target_col"])

    feature_cols = asset_cfg["feature_cols"]
    combined_features = combined[feature_cols].copy()

    with open(Path(asset_cfg["model_dir"]) / "best_params.json", "r") as f:
        params = json.load(f)

    x_scaler = _load_pickle(Path(asset_cfg["model_dir"]) / "scaler_X.pkl")
    y_scaler = _load_pickle(Path(asset_cfg["model_dir"]) / "scaler_y.pkl")

    X_scaled = x_scaler.transform(combined_features)
    lookback = params["lookback"]
    X_seq = create_sequences(X_scaled, lookback)
    pred_dates = combined["Date"].iloc[lookback:].reset_index(drop=True)

    models = load_council_models(asset_cfg["model_dir"], len(feature_cols), params, asset_cfg["model_prefix"])
    if not models:
        raise ValueError(f"No trained models found in {asset_cfg['model_dir']}")

    preds = []
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        for model in models:
            preds.append(model(X_tensor).cpu().numpy().reshape(-1))

    ensemble_scaled = np.mean(np.vstack(preds), axis=0)
    pred_return = y_scaler.inverse_transform(ensemble_scaled.reshape(-1, 1)).reshape(-1)

    pred_df = pd.DataFrame({"Date": pred_dates, "pred_return": pred_return})
    forecast_df = pred_df[(pred_df["Date"] >= FORECAST_START) & (pred_df["Date"] <= FORECAST_END)].copy()

    real_all = pd.concat([train_df, test_df], ignore_index=True)
    real_all = real_all.sort_values("Date").reset_index(drop=True)
    actual_map = dict(zip(real_all["Date"], real_all[asset_cfg["target_col"]]))

    last_train_date = train_df["Date"].max()
    last_train_price = float(train_df.loc[train_df["Date"] == last_train_date, asset_cfg["target_col"]].iloc[-1])

    predicted_prices = []
    prev_pred = None
    prev_date = last_train_date
    for _, row in forecast_df.iterrows():
        date = row["Date"]
        anchor = actual_map.get(prev_date)
        if anchor is None:
            anchor = prev_pred if prev_pred is not None else last_train_price
        pred_price = anchor * (1.0 + row["pred_return"])
        predicted_prices.append(pred_price)
        prev_pred = pred_price
        prev_date = date

    forecast_df["predicted_price"] = predicted_prices
    forecast_df["synthetic_price"] = synth_future.set_index("Date").loc[forecast_df["Date"], asset_cfg["target_col"]].values
    forecast_df["actual_price"] = [actual_map.get(dt, np.nan) for dt in forecast_df["Date"]]

    validation_mask = (forecast_df["Date"] >= VALIDATION_START) & (forecast_df["Date"] <= VALIDATION_END)
    valid_rows = forecast_df[validation_mask & forecast_df["actual_price"].notna()]

    if len(valid_rows) >= 2:
        r2 = float(r2_score(valid_rows["actual_price"], valid_rows["predicted_price"]))
        rmse = float(np.sqrt(mean_squared_error(valid_rows["actual_price"], valid_rows["predicted_price"])))
        mae = float(mean_absolute_error(valid_rows["actual_price"], valid_rows["predicted_price"]))
    else:
        r2 = float("nan")
        rmse = float("nan")
        mae = float("nan")

    return forecast_df, {"r2": r2, "rmse": rmse, "mae": mae}


def _load_pickle(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    assets = {
        "gold": {
            "asset": "gold",
            "train_csv": os.getenv("GOLD_TRAIN_CSV", "df_gold_dataset_usa_epu_train_2015_2026.csv"),
            "test_csv": os.getenv("GOLD_TEST_CSV", "df_gold_dataset_usa_epu_test_2026_04_01_2026_04_13.csv"),
            "synthetic_csv": os.getenv("GOLD_SYNTH_CSV", "gold_apr2026_train_extended.csv"),
            "model_dir": os.getenv("GOLD_MODEL_DIR", "models/gold_apr2026"),
            "target_col": "Gold_Futures",
            "model_prefix": "gold",
            "feature_cols": [
                "Silver_Futures",
                "Crude_Oil_Futures",
                "UST10Y_Treasury_Yield",
                "gepu",
                "DFF",
                "gpr_daily",
                "Gold_Futures",
                "EMA_Fast",
                "EMA_Slow",
                "RSI_7",
                "MACD_Flash",
                "MACD_Signal",
                "MACD_Hist",
                "BB_Width",
                "ROC_2",
                "GS_Ratio",
            ],
        },
        "silver": {
            "asset": "silver",
            "train_csv": os.getenv("SILVER_TRAIN_CSV", "df_silver_dataset_train_2015_2026.csv"),
            "test_csv": os.getenv("SILVER_TEST_CSV", "df_silver_dataset_test_2026_04_01_2026_04_13.csv"),
            "synthetic_csv": os.getenv("SILVER_SYNTH_CSV", "silver_apr2026_train_extended.csv"),
            "model_dir": os.getenv("SILVER_MODEL_DIR", "models/silver_apr2026"),
            "target_col": "Silver_Futures",
            "model_prefix": "silver",
            "feature_cols": [
                "Silver_Futures",
                "Gold_Futures",
                "US30",
                "SnP500",
                "NASDAQ_100",
                "USD_index",
                "EMA_10",
                "EMA_20",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "MACD_Hist",
                "BB_Width",
                "ROC_5",
                "GS_Ratio",
            ],
        },
    }

    output = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "forecast_start": str(FORECAST_START.date()),
        "forecast_end": str(FORECAST_END.date()),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "assets": {},
    }

    for name, cfg in assets.items():
        forecast_df, metrics = build_predictions(cfg)
        output["assets"][name] = {
            "dates": forecast_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": [None if np.isnan(x) else float(x) for x in forecast_df["actual_price"].to_numpy()],
            "predicted": forecast_df["predicted_price"].astype(float).tolist(),
            "synthetic": forecast_df["synthetic_price"].astype(float).tolist(),
            "metrics": metrics,
        }

        forecast_df.to_csv(REPORTS_DIR / f"forecast_{name}_apr2026.csv", index=False)

    output_path = REPORTS_DIR / "forecast_dashboard_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved dashboard data to {output_path}")


if __name__ == "__main__":
    main()
