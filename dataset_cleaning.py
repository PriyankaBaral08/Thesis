import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_FILE = "top100_stocks_features.csv"
OUTPUT_FILE = "top100_stocks_cleaned.csv"
ANALYSIS_REPORT = "data_quality_report.txt"

def analyze_data_quality(df):
    """
    Comprehensive data quality analysis
    """
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("DATA QUALITY ANALYSIS REPORT")
    report.append("="*80)
    
    # Basic info
    print(f"\n Dataset Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    report.append(f"\nDataset Shape: {df.shape}")
    report.append(f"Rows: {df.shape[0]:,}")
    report.append(f"Columns: {df.shape[1]}")
    
    # MISSING VALUES ANALYSIS 
    print("\n" + "="*80)
    print("1. MISSING VALUES ANALYSIS")
    print("="*80)
    report.append("\n" + "="*80)
    report.append("1. MISSING VALUES ANALYSIS")
    report.append("="*80)
    
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )
    
    if len(missing_stats) > 0:
        print(f"\n  Found {len(missing_stats)} columns with missing values:\n")
        print(missing_stats.to_string(index=False))
        report.append(f"\nFound {len(missing_stats)} columns with missing values:")
        report.append(missing_stats.to_string(index=False))
        
        # Categorize severity
        severe = missing_stats[missing_stats['Missing_Percent'] > 50]
        moderate = missing_stats[(missing_stats['Missing_Percent'] > 10) & 
                                 (missing_stats['Missing_Percent'] <= 50)]
        low = missing_stats[missing_stats['Missing_Percent'] <= 10]
        
        print(f"\n Severity Breakdown:")
        print(f" Severe (>50% missing): {len(severe)} columns")
        print(f" Moderate (10-50% missing): {len(moderate)} columns")
        print(f" Low (<10% missing): {len(low)} columns")
        
        report.append(f"\nSeverity Breakdown:")
        report.append(f"Severe (>50% missing): {len(severe)} columns")
        report.append(f"Moderate (10-50% missing): {len(moderate)} columns")
        report.append(f"Low (<10% missing): {len(low)} columns")
    else:
        print("\n No missing values found!")
        report.append("\nNo missing values found!")
    
    #INFINITE VALUES ANALYSIS 
    print("\n" + "="*80)
    print("2. INFINITE VALUES ANALYSIS")
    print("="*80)
    report.append("\n" + "="*80)
    report.append("2. INFINITE VALUES ANALYSIS")
    report.append("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_stats = []
    
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_stats.append({
                'Column': col,
                'Inf_Count': inf_count,
                'Inf_Percent': (inf_count / len(df) * 100).round(2)
            })
    
    if inf_stats:
        inf_df = pd.DataFrame(inf_stats).sort_values('Inf_Percent', ascending=False)
        print(f"\n Found {len(inf_df)} columns with infinite values:\n")
        print(inf_df.to_string(index=False))
        report.append(f"\nFound {len(inf_df)} columns with infinite values:")
        report.append(inf_df.to_string(index=False))
    else:
        print("\n No infinite values found!")
        report.append("\nNo infinite values found!")
    
    #  NEGATIVE VALUES ANALYSIS 
    print("\n" + "="*80)
    print("3. NEGATIVE VALUES ANALYSIS")
    print("="*80)
    report.append("\n" + "="*80)
    report.append("3. NEGATIVE VALUES ANALYSIS")
    report.append("="*80)
    
    # Features that SHOULD be positive
    should_be_positive = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
        'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
        'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_50',
        'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_20',
        'ATR_14', 'BB_Width', 'RSI_14', 'RSI_7'
    ]
    
    # Features that CAN be negative (returns, indicators, etc.)
    can_be_negative = [
        'Daily_Return', 'Log_Return', 'Momentum_5', 'Momentum_10', 'Momentum_20',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'Next_Day_Return',
        'Return_5D_Ahead', 'Return_10D_Ahead'
    ] + [col for col in df.columns if 'Return_Lag' in col]
    
    negative_issues = []
    for col in should_be_positive:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                negative_issues.append({
                    'Column': col,
                    'Negative_Count': neg_count,
                    'Negative_Percent': (neg_count / len(df) * 100).round(2),
                    'Min_Value': df[col].min()
                })
    
    if negative_issues:
        neg_df = pd.DataFrame(negative_issues).sort_values('Negative_Percent', ascending=False)
        print(f"\n Found {len(neg_df)} columns with unexpected negative values:\n")
        print(neg_df.to_string(index=False))
        report.append(f"\nFound {len(neg_df)} columns with unexpected negative values:")
        report.append(neg_df.to_string(index=False))
    else:
        print("\n No unexpected negative values found!")
        report.append("\nNo unexpected negative values found!")
    
    # Show which features can legitimately be negative
    print(f"\n Features that CAN be negative (by design): {len(can_be_negative)}")
    print("   (Returns, momentum, MACD, etc. - these are normal)")
    
    # ZERO VALUES ANALYSIS 
    print("\n" + "="*80)
    print("4. ZERO VALUES ANALYSIS (Volume)")
    print("="*80)
    report.append("\n" + "="*80)
    report.append("4. ZERO VALUES ANALYSIS")
    report.append("="*80)
    
    if 'Volume' in df.columns:
        zero_volume = (df['Volume'] == 0).sum()
        zero_pct = (zero_volume / len(df) * 100).round(2)
        print(f"\n   Volume = 0: {zero_volume:,} rows ({zero_pct}%)")
        report.append(f"\nVolume = 0: {zero_volume:,} rows ({zero_pct}%)")
        
        if zero_pct > 1:
            print("  High number of zero volume days - may indicate data quality issues")
            report.append("   WARNING: High number of zero volume days")
    
    #  OUTLIER ANALYSIS 
    print("\n" + "="*80)
    print("5. OUTLIER ANALYSIS (Returns)")
    print("="*80)
    report.append("\n" + "="*80)
    report.append("5. OUTLIER ANALYSIS")
    report.append("="*80)
    
    if 'Daily_Return' in df.columns:
        returns = df['Daily_Return'].dropna()
        extreme_up = (returns > 0.5).sum()  # >50% daily return
        extreme_down = (returns < -0.5).sum()  # <-50% daily return
        
        print(f"\n   Extreme daily returns (>±50%):")
        print(f"      Up moves >50%: {extreme_up:,}")
        print(f"      Down moves <-50%: {extreme_down:,}")
        print(f"   These might be stock splits, IPOs, or data errors")
        
        report.append(f"\nExtreme daily returns (>±50%):")
        report.append(f"Up moves >50%: {extreme_up:,}")
        report.append(f"Down moves <-50%: {extreme_down:,}")
    
    #  FEATURE STATISTICS 
    print("\n" + "="*80)
    print("6. FEATURE STATISTICS SUMMARY")
    print("="*80)
    
    stats = df[numeric_cols].describe().T
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)
    
    print("\nTop 10 features with highest standard deviation:")
    print(stats.nlargest(10, 'std')[['mean', 'std', 'min', 'max', 'missing_pct']].to_string())
    
    # Save report
    with open(ANALYSIS_REPORT, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n Full report saved to: {ANALYSIS_REPORT}")
    
    return missing_stats, inf_stats, negative_issues


def clean_data(df):
    """
    Clean the dataset based on identified issues
    """
    print("\n" + "="*80)
    print("DATA CLEANING PROCESS")
    print("="*80)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Replace infinite values with NaN
    print("\n1. Handling infinite values...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print("   ✓ Replaced inf/-inf with NaN")
    
    # 2. Handle missing values by column type
    print("\n2. Handling missing values...")
    
    # For lagged features, forward fill is appropriate
    lag_cols = [col for col in df_clean.columns if 'Lag' in col]
    if lag_cols:
        df_clean[lag_cols] = df_clean[lag_cols].fillna(method='ffill')
        print(f" Forward filled {len(lag_cols)} lagged features")
    
    # For moving averages and technical indicators, interpolate within each ticker
    for ticker in df_clean['Ticker'].unique():
        mask = df_clean['Ticker'] == ticker
        ticker_data = df_clean[mask].copy()
        
        # Interpolate numeric columns
        numeric_cols_to_fill = ticker_data.select_dtypes(include=[np.number]).columns
        ticker_data[numeric_cols_to_fill] = ticker_data[numeric_cols_to_fill].interpolate(
            method='linear', limit_direction='both'
        )
        
        df_clean.loc[mask] = ticker_data
    
    print(" Interpolated missing values within each ticker")
    
    # 3. Drop rows with remaining NaN in critical columns
    critical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols)
    dropped = before_drop - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped:,} rows with NaN in critical price/volume columns")
    
    # 4. Handle negative values in columns that should be positive
    print("\n3. Handling negative values...")
    should_be_positive = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in should_be_positive:
        if col in df_clean.columns:
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                # Replace negative values with NaN and interpolate
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear')
                print(f"   ✓ Fixed {neg_count} negative values in {col}")
    
    # 5. Handle zero volume
    print("\n4. Handling zero volume...")
    if 'Volume' in df_clean.columns:
        zero_vol_count = (df_clean['Volume'] == 0).sum()
        if zero_vol_count > 0:
            # Replace with previous day's volume
            df_clean.loc[df_clean['Volume'] == 0, 'Volume'] = np.nan
            df_clean['Volume'] = df_clean.groupby('Ticker')['Volume'].fillna(method='ffill')
            print(f"   ✓ Fixed {zero_vol_count:,} zero volume entries")
    
    # 6. Remove extreme outliers (optional - be careful!)
    print("\n5. Handling extreme outliers...")
    if 'Daily_Return' in df_clean.columns:
        # Cap returns at ±100% (stock splits, data errors)
        extreme = ((df_clean['Daily_Return'] > 1.0) | (df_clean['Daily_Return'] < -0.9)).sum()
        if extreme > 0:
            df_clean.loc[df_clean['Daily_Return'] > 1.0, 'Daily_Return'] = 1.0
            df_clean.loc[df_clean['Daily_Return'] < -0.9, 'Daily_Return'] = -0.9
            print(f"   ✓ Capped {extreme:,} extreme returns at ±100%")
    
    # 7. Final cleanup - drop any remaining rows with NaN
    print("\n6. Final cleanup...")
    before_final = len(df_clean)
    df_clean = df_clean.dropna()
    final_dropped = before_final - len(df_clean)
    if final_dropped > 0:
        print(f"   ✓ Dropped {final_dropped:,} rows with remaining NaN values")
    
    # Summary
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    print(f"   Initial rows: {initial_rows:,}")
    print(f"   Final rows: {len(df_clean):,}")
    print(f"   Rows removed: {initial_rows - len(df_clean):,} ({((initial_rows - len(df_clean))/initial_rows*100):.2f}%)")
    print(f"   Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def validate_cleaned_data(df):
    """
    Validate the cleaned dataset
    """
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: No missing values
    checks_total += 1
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("Check 1: No missing values")
        checks_passed += 1
    else:
        print(f" Check 1: Still has {missing:,} missing values")
    
    # Check 2: No infinite values
    checks_total += 1
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count == 0:
        print(" Check 2: No infinite values")
        checks_passed += 1
    else:
        print(f" Check 2: Still has {inf_count:,} infinite values")
    
    # Check 3: Price columns are positive
    checks_total += 1
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    neg_prices = sum((df[col] < 0).sum() for col in price_cols if col in df.columns)
    if neg_prices == 0:
        print(" Check 3: All price columns are positive")
        checks_passed += 1
    else:
        print(f" Check 3: Found {neg_prices} negative prices")
    
    # Check 4: Volume is positive
    checks_total += 1
    if 'Volume' in df.columns:
        neg_vol = (df['Volume'] < 0).sum()
        zero_vol = (df['Volume'] == 0).sum()
        if neg_vol == 0 and zero_vol < len(df) * 0.01:  # <1% zeros acceptable
            print(" Check 4: Volume values are valid")
            checks_passed += 1
        else:
            print(f" Check 4: Volume issues - negative: {neg_vol}, zero: {zero_vol}")
    
    # Check 5: Reasonable data distribution
    checks_total += 1
    if 'Daily_Return' in df.columns:
        returns_std = df['Daily_Return'].std()
        if 0.005 < returns_std < 0.1:  # Reasonable daily volatility
            print(" Check 5: Return distribution looks reasonable")
            checks_passed += 1
        else:
            print(f"  Check 5: Unusual return std: {returns_std:.4f}")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*80}")
    
    return checks_passed == checks_total


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECK & CLEANING PIPELINE")
    print("="*80)
    
    # Load data
    print(f"\n Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f" Loaded {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f" Error: File '{INPUT_FILE}' not found!")
        print("  Please run the feature engineering script first.")
        return
    
    # Analyze data quality
    missing_stats, inf_stats, neg_issues = analyze_data_quality(df)
    
    # Ask user if they want to clean
    print("\n" + "="*80)
    response = input("\n Do you want to clean the data? (yes/no): ").lower()
    
    if response in ['yes', 'y']:
        # Clean data
        df_cleaned = clean_data(df)
        
        # Validate
        all_checks_passed = validate_cleaned_data(df_cleaned)
        
        # Save cleaned data
        df_cleaned.to_csv(OUTPUT_FILE, index=False)
        print(f"\n Cleaned data saved to: {OUTPUT_FILE}")
        print(f" Final shape: {df_cleaned.shape}")
        
        if all_checks_passed:
            print("\n Data is now ready for ML/DL modeling!")
        else:
            print("\n Some validation checks failed. Review the data before modeling.")
        
        # Display sample
        print("\n" + "="*80)
        print("SAMPLE OF CLEANED DATA (first 5 rows):")
        print("="*80)
        print(df_cleaned.head().to_string())
        
    else:
        print("\n Skipping data cleaning. Review the analysis report.")
    
    print("\n Process complete!")


if __name__ == "__main__":
    main()