import pandas as pd
import numpy as np

INPUT_FILE = "top100_stocks_cleaned.csv"
OUTPUT_FILE = "top100_stocks_final.csv"

def fix_negative_values():
    """
    Fix invalid negative values in moving averages and technical indicators
    """
    print("\n" + "="*80)
    print("FIXING INVALID NEGATIVE VALUES")
    print("="*80)
    
    # Load data
    print(f"\n Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Initial shape: {df.shape}")
    
    df_fixed = df.copy()
    
    # Columns that must be positive
    must_be_positive = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
        'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
        'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_50',
        'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_20',
        'ATR_14', 'BB_Width', 'BB_Upper', 'BB_Lower', 'BB_Middle',
        'High_Low_Range', 'High_Low_Range_MA_10'
    ]
    
    # Bounded 0-100
    bounded_0_100 = ['RSI_14', 'RSI_7', 'Stochastic_K', 'Stochastic_D']
    
    # Bounded 0-1
    bounded_0_1 = ['BB_Position']
    
    print("\n" + "="*80)
    print("STEP 1: Fixing columns that MUST be positive")
    print("="*80)
    
    total_fixed = 0
    
    for col in must_be_positive:
        if col not in df_fixed.columns:
            continue
            
        neg_count_before = (df_fixed[col] < 0).sum()
        
        if neg_count_before > 0:
            print(f"\n Fixing {col}...")
            print(f"   Found {neg_count_before} negative values")
            
            # Strategy: Replace negative values with NaN, then interpolate within each ticker
            for ticker in df_fixed['Ticker'].unique():
                mask = df_fixed['Ticker'] == ticker
                ticker_data = df_fixed.loc[mask, col].copy()
                
                # Replace negatives with NaN
                ticker_data[ticker_data < 0] = np.nan
                
                # Interpolate (linear)
                ticker_data = ticker_data.interpolate(method='linear', limit_direction='both')
                
                # If still NaN (e.g., all values were negative), use forward/backward fill
                ticker_data = ticker_data.fillna(method='ffill').fillna(method='bfill')
                
                # If STILL NaN (entire ticker column was bad), use a sensible default
                if ticker_data.isna().any():
                    # For moving averages, use the Close price as fallback
                    if 'MA' in col or 'SMA' in col or 'EMA' in col or 'BB' in col:
                        if 'Close' in df_fixed.columns:
                            ticker_data = ticker_data.fillna(df_fixed.loc[mask, 'Close'])
                    # For volume-based, use Volume
                    elif 'Volume' in col:
                        if 'Volume' in df_fixed.columns:
                            ticker_data = ticker_data.fillna(df_fixed.loc[mask, 'Volume'])
                    # For volatility/range, use a small positive value
                    else:
                        ticker_data = ticker_data.fillna(0.01)
                
                df_fixed.loc[mask, col] = ticker_data
            
            neg_count_after = (df_fixed[col] < 0).sum()
            fixed_count = neg_count_before - neg_count_after
            total_fixed += fixed_count
            
            print(f"  Fixed {fixed_count} values")
            if neg_count_after > 0:
                print(f"  Still {neg_count_after} negative values remaining")
    
    print(f"\n Total fixed in Step 1: {total_fixed}")
    
    # STEP 2: Fix bounded indicators
    print("\n" + "="*80)
    print("STEP 2: Fixing bounded indicators (0-100)")
    print("="*80)
    
    for col in bounded_0_100:
        if col not in df_fixed.columns:
            continue
            
        below_0 = (df_fixed[col] < 0).sum()
        above_100 = (df_fixed[col] > 100).sum()
        
        if below_0 > 0 or above_100 > 0:
            print(f"\n Fixing {col}...")
            print(f"   Below 0: {below_0}, Above 100: {above_100}")
            
            # Clip values to 0-100 range
            df_fixed[col] = df_fixed[col].clip(lower=0, upper=100)
            
            print(f"   Clipped to [0, 100] range")
    
    # STEP 3: Fix BB_Position (0-1 range)
    print("\n" + "="*80)
    print("STEP 3: Fixing BB_Position (0-1 range)")
    print("="*80)
    
    for col in bounded_0_1:
        if col not in df_fixed.columns:
            continue
            
        below_0 = (df_fixed[col] < 0).sum()
        above_1 = (df_fixed[col] > 1).sum()
        
        if below_0 > 0 or above_1 > 0:
            print(f"\n Fixing {col}...")
            print(f"   Below 0: {below_0}, Above 1: {above_1}")
            
            # Clip to 0-1 range
            df_fixed[col] = df_fixed[col].clip(lower=0, upper=1)
            
            print(f"    Clipped to [0, 1] range")
    
    # STEP 4: Recalculate Price_to_SMA ratios (in case SMAs were negative)
    print("\n" + "="*80)
    print("STEP 4: Recalculating Price-to-MA ratios")
    print("="*80)
    
    ratio_cols = ['Price_to_SMA_20', 'Price_to_SMA_50', 'Price_to_SMA_200']
    
    for col in ratio_cols:
        if col in df_fixed.columns:
            sma_col = col.replace('Price_to_', '')
            
            if sma_col in df_fixed.columns and 'Adj Close' in df_fixed.columns:
                print(f"\n Recalculating {col}...")
                
                # Recalculate where SMA is positive
                mask = df_fixed[sma_col] > 0
                df_fixed.loc[mask, col] = df_fixed.loc[mask, 'Adj Close'] / df_fixed.loc[mask, sma_col]
                
                # For any remaining issues, use forward fill
                df_fixed[col] = df_fixed.groupby('Ticker')[col].fillna(method='ffill').fillna(method='bfill')
                
                print(f" Recalculated {mask.sum()} values")
    
    # STEP 5: Final cleanup - drop any rows that still have issues
    print("\n" + "="*80)
    print("STEP 5: Final cleanup")
    print("="*80)
    
    # Check for any remaining NaN in critical columns
    critical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    before_drop = len(df_fixed)
    df_fixed = df_fixed.dropna(subset=critical_cols)
    dropped = before_drop - len(df_fixed)
    
    if dropped > 0:
        print(f"\n   Dropped {dropped} rows with NaN in critical columns")
    else:
        print(f"\n  No rows needed to be dropped")
    
    # VALIDATION
    print("\n" + "="*80)
    print("VALIDATION - Checking all issues are resolved")
    print("="*80)
    
    all_good = True
    
    # Check must_be_positive
    print("\n1. Checking columns that must be positive:")
    for col in must_be_positive:
        if col in df_fixed.columns:
            neg_count = (df_fixed[col] < 0).sum()
            if neg_count > 0:
                print(f"    {col}: Still has {neg_count} negative values")
                all_good = False
    
    if all_good:
        print(" All columns are now positive!")
    
    # Check bounded
    print("\n2. Checking bounded indicators:")
    for col in bounded_0_100:
        if col in df_fixed.columns:
            below = (df_fixed[col] < 0).sum()
            above = (df_fixed[col] > 100).sum()
            if below > 0 or above > 0:
                print(f" {col}: {below} below 0, {above} above 100")
                all_good = False
            else:
                print(f"   {col}: Properly bounded")
    
    #Returns Distribution 

    print("\n3. Checking returns distribution (should have ~50% negatives):")
    if 'Daily_Return' in df_fixed.columns:
        returns = df_fixed['Daily_Return'].dropna()
        neg_pct = (returns < 0).sum() / len(returns) * 100
        pos_pct = (returns > 0).sum() / len(returns) * 100
        
        print(f"   Negative: {neg_pct:.1f}%, Positive: {pos_pct:.1f}%")
        
        if 40 <= neg_pct <= 60:
            print(f"  Healthy distribution!")
        else:
            print(f"  Unusual distribution (expected ~50/50)")
    
    
    print("\n" + "="*80)
    print("SAVING FINAL DATASET")
    print("="*80)
    
    df_fixed.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n Saved to: {OUTPUT_FILE}")
    print(f"   Final shape: {df_fixed.shape}")
    print(f"   Rows removed: {len(df) - len(df_fixed):,}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n Dataset Statistics:")
    print(f"   Total rows: {len(df_fixed):,}")
    print(f"   Total columns: {len(df_fixed.columns)}")
    print(f"   Tickers: {df_fixed['Ticker'].nunique()}")
    print(f"   Date range: {df_fixed['Date'].min()} to {df_fixed['Date'].max()}")
    
    print(f"\n Data Quality:")
    print(f"   Missing values: {df_fixed.isnull().sum().sum()}")
    print(f"   Infinite values: {np.isinf(df_fixed.select_dtypes(include=[np.number])).sum().sum()}")
    
    if all_good:
        print("\n" + "="*80)
        print(" SUCCESS! YOUR DATA IS NOW READY FOR ML/DL MODELS!")
        print("="*80)
        print("\nNext steps:")
        print(f"   1. Load: df = pd.read_csv('{OUTPUT_FILE}')")
        print("   2. Split features and target")
        print("   3. Scale features (StandardScaler)")
        print("   4. Train your models!")
    else:
        print("\n" + "="*80)
        print(" Some issues may remain. Review the validation section above.")
        print("="*80)
    
    return df_fixed


if __name__ == "__main__":
    df_final = fix_negative_values()
    
    print("\n" + "="*80)
    print("SAMPLE OF FINAL DATA:")
    print("="*80)
    print(df_final.head(10).to_string())
    print("\n Done!")