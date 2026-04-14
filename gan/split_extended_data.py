import pandas as pd
import os
import sys
from pathlib import Path

# Ensure we can import from the gan directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_catalog import (
    get_enabled_dataset_configs,
    get_extended_output_path,
    PROJECT_ROOT,
    ensure_prepared_source
)

def split_extended(config):
    asset = config['asset']
    print(f"--- Splitting extended data for {asset} ---")
    
    ext_path = get_extended_output_path(config)
    if not ext_path.exists():
        print(f"Error: {ext_path} not found. Run training first.")
        return

    df = pd.read_csv(ext_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load prepared source to find the historical boundary
    source_path = PROJECT_ROOT / (config['prepared_file'] or config['source_file'])
    source_df = pd.read_csv(source_path)
    source_df['Date'] = pd.to_datetime(source_df['Date'])
    last_hist_date = source_df['Date'].max()
    
    hist_df = df[df['Date'] <= last_hist_date].copy().sort_values('Date')
    future_df = df[df['Date'] > last_hist_date].copy().sort_values('Date')
    
    # Split historical 80/20
    split_idx = int(len(hist_df) * 0.8)
    train_df = hist_df.iloc[:split_idx].copy()
    test_hist_df = hist_df.iloc[split_idx:].copy()
    
    # Extended Test = Test Historical portion + Synthetic Future portion
    test_df = pd.concat([test_hist_df, future_df], ignore_index=True)
    
    # Target filenames (with "new_" prefix as requested by user)
    train_path = PROJECT_ROOT / f"new_extended_train_{asset}.csv"
    test_path = PROJECT_ROOT / f"new_extended_test_{asset}.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Successfully created:")
    print(f"  - {train_path} ({len(train_df)} rows)")
    print(f"  - {test_path} ({len(test_df)} rows: {len(test_hist_df)} hist + {len(future_df)} synth)")

if __name__ == "__main__":
    for config in get_enabled_dataset_configs():
        split_extended(config)
