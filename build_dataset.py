import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

START_DATE = "2000-01-01"
END_DATE = None  # up to today (24/11/2025)

# -------- Top 100 tickers with region --------
TICKERS_REGION = {
    # US stock list
    "AAPL":"US","MSFT":"US","GOOGL":"US","GOOG":"US","AMZN":"US","META":"US",
    "TSLA":"US","NVDA":"US","BRK-B":"US","JPM":"US","V":"US","MA":"US","UNH":"US",
    "XOM":"US","JNJ":"US","WMT":"US","PG":"US","HD":"US","KO":"US","PEP":"US","BAC":"US",
    "PFE":"US","ABBV":"US","COST":"US","AVGO":"US","TSM":"US","LLY":"US","MRK":"US",
    "ORCL":"US","CRM":"US","CSCO":"US","MCD":"US","ADBE":"US","NFLX":"US",

    # Europe/UK
    "SAP.DE":"Europe","ASML.AS":"Europe","MC.PA":"Europe","OR.PA":"Europe",
    "AIR.PA":"Europe","NESN.SW":"Europe","ROG.SW":"Europe","NOVO-B.CO":"Europe",
    "TTE.PA":"Europe","SAN.PA":"Europe","SIE.DE":"Europe","ALV.DE":"Europe",
    "DTE.DE":"Europe","ULVR.L":"UK","HSBA.L":"UK","BP.L":"UK",
    "SHEL.L":"UK","RIO.L":"UK","BHP.L":"UK","GSK.L":"UK",

    # India
    "RELIANCE.NS":"India","TCS.NS":"India","HDFCBANK.NS":"India","ICICIBANK.NS":"India",
    "INFY.NS":"India","HINDUNILVR.NS":"India","SBIN.NS":"India","BHARTIARTL.NS":"India",
    "ITC.NS":"India","KOTAKBANK.NS":"India","AXISBANK.NS":"India","LT.NS":"India",
    "ASIANPAINT.NS":"India","MARUTI.NS":"India","BAJFINANCE.NS":"India",

    # China/HK
    "0700.HK":"China","9988.HK":"China","3690.HK":"China","0941.HK":"China",
    "1299.HK":"China","2318.HK":"China","1398.HK":"China","0939.HK":"China",
    "3968.HK":"China","9618.HK":"China","BABA":"China","TCEHY":"China",
    "PDD":"China","BIDU":"China","JD":"China",

    # Japan
    "7203.T":"Japan","6758.T":"Japan","9984.T":"Japan","9432.T":"Japan",
    "7974.T":"Japan","4502.T":"Japan","8306.T":"Japan","8411.T":"Japan",

    # Korea, Canada, Australia
    "005930.KS":"Korea","000660.KS":"Korea","035420.KS":"Korea",
    "SHOP.TO":"Canada","RY.TO":"Canada","BHP.AX":"Australia",
    "CBA.AX":"Australia","CSL.AX":"Australia"
}

OUTPUT_CSV_RAW = "top100_stocks_raw.csv"
OUTPUT_CSV_FEATURES = "top100_stocks_features.csv"


# ---------- Downloader ----------

def download_ticker(ticker, region):
    print(f"→ Downloading {ticker} ({region})")

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            actions=False
        )

        if df.empty:
            print(f"⚠ No data for {ticker}, skipping.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing_cols = [c for c in required if c not in df.columns]
        
        if missing_cols:
            print(f"Missing columns {missing_cols} for {ticker}, skipping.")
            return None

        df = df[required].copy()
        df = df.ffill().bfill().dropna()
        
        if df.empty:
            print(f"No valid data after cleaning for {ticker}, skipping.")
            return None

        df["Ticker"] = ticker
        df["Region"] = region
        df.reset_index(inplace=True)
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        
        print(f"Downloaded {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        print(f"ERROR with {ticker}: {e}")
        return None


# ---------- Feature Engineering ----------

def create_features(df):
    """
    Create comprehensive features for ML/DL models
    Features are calculated per ticker (grouped operation)
    """
    print("\n Creating features...")
    
    # Sort by ticker and date to ensure correct order
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    features_list = []
    
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # ========== Price-based Features ==========
        
        # Returns
        ticker_df['Daily_Return'] = ticker_df['Adj Close'].pct_change()
        ticker_df['Log_Return'] = np.log(ticker_df['Adj Close'] / ticker_df['Adj Close'].shift(1))
        
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            ticker_df[f'Return_Lag_{lag}'] = ticker_df['Daily_Return'].shift(lag)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            ticker_df[f'SMA_{window}'] = ticker_df['Adj Close'].rolling(window=window).mean()
            ticker_df[f'EMA_{window}'] = ticker_df['Adj Close'].ewm(span=window, adjust=False).mean()
        
        # Price momentum
        ticker_df['Momentum_5'] = ticker_df['Adj Close'] - ticker_df['Adj Close'].shift(5)
        ticker_df['Momentum_10'] = ticker_df['Adj Close'] - ticker_df['Adj Close'].shift(10)
        ticker_df['Momentum_20'] = ticker_df['Adj Close'] - ticker_df['Adj Close'].shift(20)
        
        # Price position relative to moving averages
        ticker_df['Price_to_SMA_20'] = ticker_df['Adj Close'] / ticker_df['SMA_20']
        ticker_df['Price_to_SMA_50'] = ticker_df['Adj Close'] / ticker_df['SMA_50']
        ticker_df['Price_to_SMA_200'] = ticker_df['Adj Close'] / ticker_df['SMA_200']
        
        # ========== Volatility Features ==========
        
        # Rolling standard deviation of returns
        ticker_df['Volatility_5'] = ticker_df['Daily_Return'].rolling(window=5).std()
        ticker_df['Volatility_10'] = ticker_df['Daily_Return'].rolling(window=10).std()
        ticker_df['Volatility_20'] = ticker_df['Daily_Return'].rolling(window=20).std()
        ticker_df['Volatility_50'] = ticker_df['Daily_Return'].rolling(window=50).std()
        
        # Intraday range
        ticker_df['High_Low_Range'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Low']
        ticker_df['High_Low_Range_MA_10'] = ticker_df['High_Low_Range'].rolling(window=10).mean()
        
        # ========== Volume Features ==========
        
        # Volume moving averages
        ticker_df['Volume_MA_5'] = ticker_df['Volume'].rolling(window=5).mean()
        ticker_df['Volume_MA_10'] = ticker_df['Volume'].rolling(window=10).mean()
        ticker_df['Volume_MA_20'] = ticker_df['Volume'].rolling(window=20).mean()
        
        # Volume ratio
        ticker_df['Volume_Ratio_5'] = ticker_df['Volume'] / ticker_df['Volume_MA_5']
        ticker_df['Volume_Ratio_10'] = ticker_df['Volume'] / ticker_df['Volume_MA_10']
        
        # ========== Technical Indicators ==========
        
        # RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        ticker_df['RSI_14'] = calculate_rsi(ticker_df['Adj Close'], 14)
        ticker_df['RSI_7'] = calculate_rsi(ticker_df['Adj Close'], 7)
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = ticker_df['Adj Close'].ewm(span=12, adjust=False).mean()
        ema_26 = ticker_df['Adj Close'].ewm(span=26, adjust=False).mean()
        ticker_df['MACD'] = ema_12 - ema_26
        ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
        ticker_df['MACD_Diff'] = ticker_df['MACD'] - ticker_df['MACD_Signal']
        
        # Bollinger Bands
        ticker_df['BB_Middle'] = ticker_df['Adj Close'].rolling(window=20).mean()
        bb_std = ticker_df['Adj Close'].rolling(window=20).std()
        ticker_df['BB_Upper'] = ticker_df['BB_Middle'] + (2 * bb_std)
        ticker_df['BB_Lower'] = ticker_df['BB_Middle'] - (2 * bb_std)
        ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / ticker_df['BB_Middle']
        ticker_df['BB_Position'] = (ticker_df['Adj Close'] - ticker_df['BB_Lower']) / (ticker_df['BB_Upper'] - ticker_df['BB_Lower'])
        
        # Average True Range (ATR)
        high_low = ticker_df['High'] - ticker_df['Low']
        high_close = np.abs(ticker_df['High'] - ticker_df['Close'].shift())
        low_close = np.abs(ticker_df['Low'] - ticker_df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['ATR_14'] = true_range.rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = ticker_df['Low'].rolling(window=14).min()
        high_14 = ticker_df['High'].rolling(window=14).max()
        ticker_df['Stochastic_K'] = 100 * ((ticker_df['Close'] - low_14) / (high_14 - low_14))
        ticker_df['Stochastic_D'] = ticker_df['Stochastic_K'].rolling(window=3).mean()
        
        # ========== Time-based Features ==========
        
        ticker_df['Day_of_Week'] = ticker_df['Date'].dt.dayofweek
        ticker_df['Month'] = ticker_df['Date'].dt.month
        ticker_df['Quarter'] = ticker_df['Date'].dt.quarter
        ticker_df['Year'] = ticker_df['Date'].dt.year
        ticker_df['Day_of_Month'] = ticker_df['Date'].dt.day
        ticker_df['Week_of_Year'] = ticker_df['Date'].dt.isocalendar().week
        
        # ========== Target Variables (for supervised learning) ==========
        
        # Next day return (classification target: up=1, down=0)
        ticker_df['Next_Day_Return'] = ticker_df['Daily_Return'].shift(-1)
        ticker_df['Target_Binary'] = (ticker_df['Next_Day_Return'] > 0).astype(int)
        
        # Multi-day ahead returns
        ticker_df['Return_5D_Ahead'] = ticker_df['Adj Close'].shift(-5) / ticker_df['Adj Close'] - 1
        ticker_df['Return_10D_Ahead'] = ticker_df['Adj Close'].shift(-10) / ticker_df['Adj Close'] - 1
        
        # Multi-class target (strong down, down, neutral, up, strong up)
        def classify_return(ret):
            if pd.isna(ret):
                return np.nan
            elif ret < -0.02:
                return 0  # Strong down
            elif ret < 0:
                return 1  # Down
            elif ret < 0.02:
                return 2  # Neutral
            elif ret < 0.05:
                return 3  # Up
            else:
                return 4  # Strong up
        
        ticker_df['Target_MultiClass'] = ticker_df['Next_Day_Return'].apply(classify_return)
        
        # Regression target (next day price)
        ticker_df['Next_Day_Price'] = ticker_df['Adj Close'].shift(-1)
        
        features_list.append(ticker_df)
    
    # Combine all tickers
    result = pd.concat(features_list, ignore_index=True)
    
    print(f"Created {len(result.columns) - len(df.columns)} new features")
    return result


# ---------- Build Dataset ----------

def build_dataset():
    print("\n Building global stock dataset with features...\n")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE or 'Today'}")
    print(f"Total tickers to download: {len(TICKERS_REGION)}\n")
    
    frames = []
    success_count = 0
    fail_count = 0

    for ticker, region in TICKERS_REGION.items():
        df = download_ticker(ticker, region)
        if df is not None:
            frames.append(df)
            success_count += 1
        else:
            fail_count += 1

    print(f"\n Download Summary: {success_count} successful, {fail_count} failed")

    if not frames:
        print("\n NO DATA COLLECTED")
        return None, None

    # Concatenate raw data
    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df[["Date", "Ticker", "Region", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    raw_df = raw_df.sort_values(["Region", "Ticker", "Date"]).reset_index(drop=True)
    
    # Save raw data
    raw_df.to_csv(OUTPUT_CSV_RAW, index=False)
    print(f"\n Saved raw data: {OUTPUT_CSV_RAW}")
    
    # Create features
    feature_df = create_features(raw_df)
    
    # Remove rows with NaN in target variable (last few rows per ticker)
    feature_df_clean = feature_df.dropna(subset=['Target_Binary', 'Next_Day_Return'])
    
    # Save featured data
    feature_df_clean.to_csv(OUTPUT_CSV_FEATURES, index=False)
    
    # Summary
    print("\n" + "="*80)
    print("DATASET SUMMARY:")
    print("="*80)
    print(f" Raw data saved to: {OUTPUT_CSV_RAW}")
    print(f" Featured data saved to: {OUTPUT_CSV_FEATURES}")
    print(f"\n Raw data shape: {raw_df.shape}")
    print(f" Featured data shape: {feature_df_clean.shape}")
    print(f" Total features: {feature_df_clean.shape[1]}")
    print(f" Date range: {feature_df_clean['Date'].min()} to {feature_df_clean['Date'].max()}")
    print(f"\n Regions: {feature_df_clean['Region'].nunique()}")
    print(feature_df_clean['Region'].value_counts().to_string())
    print(f"\n Tickers: {feature_df_clean['Ticker'].nunique()}")
    print(f"\n Target Distribution (Binary):")
    print(feature_df_clean['Target_Binary'].value_counts(normalize=True).to_string())
    print(f"\n Target Distribution (MultiClass):")
    print(feature_df_clean['Target_MultiClass'].value_counts(normalize=True).sort_index().to_string())
    
    # Feature list
    print("\n ALL FEATURES CREATED:")
    print("="*80)
    feature_cols = [col for col in feature_df_clean.columns if col not in raw_df.columns]
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:3d}. {col}")
    
    print("\n" + "="*80)
    print("SAMPLE DATA:")
    print("="*80)
    print(feature_df_clean.head(10).to_string())
    print("="*80)
    
    return raw_df, feature_df_clean


if __name__ == "__main__":
    raw_data, featured_data = build_dataset()
    if featured_data is not None:
        print("\n Done! Our dataset is ready.")
    else:
        print("\n Failed to create dataset.")