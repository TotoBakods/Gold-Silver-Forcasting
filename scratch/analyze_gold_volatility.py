import pandas as pd
import numpy as np

# Load the test dataset (LIVELY VERSION)
file_path = r'c:\Users\TotoBakod\Documents\Github\Gold-Silver-Forcasting\df_gold_dataset_gepu_extended_lively.csv'
df = pd.read_csv(file_path)

# Calculate Daily Returns
# Assuming the price column is 'Close' or the first numeric column if not sure
# In previous context, we saw 'Close' or the primary target.
# Let's check the columns first or use the first likely target.
target_col = 'Gold_Futures'

if target_col in df.columns:
    df['Returns'] = df[target_col].pct_change()
    volatility = df['Returns'].std()
    mean_ret = df['Returns'].mean()
    
    print(f"Target Column: {target_col}")
    print(f"Total Rows: {len(df)}")
    print(f"Daily Volatility (Std Dev of Returns): {volatility:.6f} ({volatility*100:.3f}%)")
    print(f"Average Daily Return: {mean_ret:.6f} ({mean_ret*100:.3f}%)")
    
    # Check for "Flatness"
    # Average Gold Volatility is typically 0.8% to 1.2% daily.
    if volatility < 0.005:
        print("Verdict: VERY FLAT (Volatility is less than 0.5% daily)")
    elif volatility < 0.008:
        print("Verdict: MODERATELY FLAT (Volatility is less than 0.8% daily)")
    else:
        print("Verdict: NORMAL VOLATILITY")
else:
    print(f"Target column not found. Columns: {df.columns.tolist()}")
