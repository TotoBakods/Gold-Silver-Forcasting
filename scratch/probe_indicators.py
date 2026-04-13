import pandas as pd
import numpy as np
from technical_indicators import calculate_indicators

# Create a dummy dataframe
df = pd.DataFrame({
    'Silver_Futures': np.random.randn(100)
})

# Attempt calculation
try:
    df_inds = calculate_indicators(df, 'Silver_Futures')
    print("Columns produced:", df_inds.columns.tolist())
    
    required = ['EMA_Fast', 'EMA_Slow', 'RSI_7', 'MACD_Flash', 'ROC_2']
    missing = [c for c in required if c not in df_inds.columns]
    
    if missing:
        print(f"CRITICAL: Missing columns: {missing}")
    else:
        print("SUCCESS: All 'Flash' sensors detected.")
except Exception as e:
    print(f"FATAL ERROR during calculation: {e}")
