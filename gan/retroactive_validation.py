import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "gan"))

from validation_utils import calculate_validation_metrics, log_validation_results
from dataset_catalog import get_dataset_config_by_asset, DATASET_CONFIGS

def validate_asset(asset_name, val_start="2026-04-01", val_end="2026-04-13"):
    print(f"--- Retroactive Validation for {asset_name.upper()} ---")
    
    # 1. Paths
    gan_dir = PROJECT_ROOT / "gan" / asset_name
    reports_dir = gan_dir / "reports"
    synthetic_csv = reports_dir / "synthetic_test_data.csv"
    
    if not synthetic_csv.exists():
        print(f"Error: {synthetic_csv} not found. Run training first.")
        return

    # 2. Get Real Data
    config = get_dataset_config_by_asset(asset_name)
    # Ensure we use the 2015-2026 val config if possible
    try:
        config = next(c for c in DATASET_CONFIGS if c['name'] == f"{asset_name}_2015_2026_val")
    except StopIteration:
        pass
        
    df_real = pd.read_csv(PROJECT_ROOT / config['source_file'])
    df_real['Date'] = pd.to_datetime(df_real['Date'])
    real_test_df = df_real[(df_real['Date'] >= val_start) & (df_real['Date'] <= val_end)].copy()
    
    # 3. Load Synthetic Data
    gen_test_df = pd.read_csv(synthetic_csv)
    gen_test_df['Date'] = pd.to_datetime(gen_test_df['Date'])
    
    # 4. Calculate Advanced Metrics
    print("Calculating advanced metrics (MMD, JSD, ARCH, MDD)...")
    metrics = calculate_validation_metrics(real_test_df, gen_test_df, config['target_col'])
    
    # 5. Log Results
    output_path = reports_dir / "validation_metrics.txt"
    log_validation_results(metrics, f"{asset_name} (Retroactive Extended)", output_path)
    print(f"Updated metrics saved to {output_path}")

if __name__ == "__main__":
    validate_asset("gold")
    validate_asset("silver")
