import pandas as pd
import numpy as np

df = pd.read_csv('top100_stocks_cleaned.csv')

# Features that MUST be positive
must_be_positive = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
    'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_50',
    'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_20',
    'ATR_14', 'BB_Width', 'High_Low_Range'
]

# Features that CAN and SHOULD have negatives
can_be_negative = [
    'Daily_Return', 'Log_Return', 'Next_Day_Return',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'MACD', 'MACD_Signal', 'MACD_Diff',
    'Return_5D_Ahead', 'Return_10D_Ahead'
] + [col for col in df.columns if 'Return_Lag' in col]

# Features that should be bounded 0-100
bounded_0_100 = ['RSI_14', 'RSI_7', 'Stochastic_K', 'Stochastic_D']

print("="*80)
print("CHECKING FOR PROBLEMATIC NEGATIVE VALUES")
print("="*80)

# Check must_be_positive columns
print("\n1. Checking columns that MUST be positive:")
issues_found = False
for col in must_be_positive:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f" {col}: {neg_count} negative values (MIN: {df[col].min():.4f})")
            issues_found = True

if not issues_found:
    print(" All price/volume/volatility columns are positive!")

# Check can_be_negative (these SHOULD have negatives)
print("\n2. Checking columns that SHOULD have negatives:")
for col in can_be_negative[:5]:  # Show first 5
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        neg_pct = (neg_count / len(df) * 100)
        print(f"    {col}: {neg_count} negative ({neg_pct:.1f}%) - This is NORMAL")

# Check bounded columns
print("\n3. Checking bounded indicators (0-100):")
for col in bounded_0_100:
    if col in df.columns:
        below_0 = (df[col] < 0).sum()
        above_100 = (df[col] > 100).sum()
        if below_0 > 0 or above_100 > 0:
            print(f"    {col}: {below_0} below 0, {above_100} above 100")
        else:
            print(f"    {col}: Properly bounded")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If you see  for sections 1 and 3, your data is READY for ML!")
print("Section 2 having negatives is EXPECTED and NECESSARY!")
print("="*80)