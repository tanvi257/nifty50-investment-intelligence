import pandas as pd
import numpy as np

def compute_technical_indicators(df):
    """Computes technical indicators for a given stock DataFrame.
    
    Expected columns in df: Date, Close, Open, High, Low, Volume
    Returns: DataFrame with technical indicator columns added.
    """
    df = df.copy()
    
    # 1. Simple Moving Averages (SMA)
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    
    # 2. Exponential Moving Averages (EMA)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False, min_periods=1).mean()
    
    # 3. MACD (Moving Average Convergence Divergence)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # 4. Relative Strength Index (RSI) - 14 Days
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, 1e-9) # Avoid division by zero
    df["RSI_14"] = 100 - (100 / (1 + rs))
    # Replace initial NaNs with neutral 50
    df["RSI_14"] = df["RSI_14"].fillna(50)
    
    # 5. Bollinger Bands (20 Days)
    sma_20 = df["Close"].rolling(window=20, min_periods=1).mean()
    std_20 = df["Close"].rolling(window=20, min_periods=1).std()
    # Handle early rows with std=NaN
    std_20 = std_20.fillna(0)
    
    df["BB_Middle"] = sma_20
    df["BB_Upper"] = sma_20 + (2 * std_20)
    df["BB_Lower"] = sma_20 - (2 * std_20)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"].replace(0, 1e-9)
    
    # 6. Volatility & Returns
    df["Return_1d"] = df["Close"].pct_change(fill_method=None)
    df["Return_1d"] = df["Return_1d"].fillna(0)
    
    # Historical Volatility (20-day rolling standard deviation of daily returns)
    df["Volatility_20d"] = df["Return_1d"].rolling(window=20, min_periods=1).std() * np.sqrt(252) # Annualized Volatility
    df["Volatility_20d"] = df["Volatility_20d"].fillna(0)
    
    # 7. Momentum Indicators
    df["ROC_10"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10).replace(0, 1e-9)) * 100
    df["ROC_10"] = df["ROC_10"].fillna(0)
    
    df["Momentum_Ratio"] = df["Close"] / df["SMA_50"].replace(0, 1e-9)
    
    # Historical cumulative returns over 5 and 10 business days
    df["Return_5d"] = df["Close"].pct_change(periods=5, fill_method=None)
    df["Return_5d"] = df["Return_5d"].fillna(0)
    
    df["Return_10d"] = df["Close"].pct_change(periods=10, fill_method=None)
    df["Return_10d"] = df["Return_10d"].fillna(0)
    
    # 8. Volume Features
    df["Vol_SMA_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Vol_SMA_20"].replace(0, 1e-9)
    
    return df

def generate_predictor_features(df, forecast_horizon=5):
    """Generates features and targets for training predictive models.
    
    Target 1: Returns over the next `forecast_horizon` days (Regression target).
    Target 2: Direction of stock movement (Classification target: 1 if return > 0, else 0).
    """
    df = compute_technical_indicators(df)
    
    # Forecast target: Forward returns
    df["Target_Return"] = df["Close"].pct_change(periods=forecast_horizon, fill_method=None).shift(-forecast_horizon)
    df["Target_Direction"] = (df["Target_Return"] > 0).astype(int)
    
    # Drop rows at the end where target is NaN (since we can't forecast past the end of the dataset)
    df_clean = df.dropna(subset=["Target_Return"]).copy()
    
    # Feature columns list
    feature_cols = [
        "SMA_10", "SMA_50", "SMA_200", 
        "EMA_12", "EMA_26", "EMA_50",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_14", 
        "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width",
        "Volatility_20d", "ROC_10", "Momentum_Ratio",
        "Return_1d", "Return_5d", "Return_10d",
        "Volume_Ratio"
    ]
    
    # Normalize price-dependent features by dividing by the close price to make them scale-invariant
    df_features = df_clean.copy()
    price_features = ["SMA_10", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "EMA_50", "BB_Middle", "BB_Upper", "BB_Lower"]
    
    for feat in price_features:
        df_features[feat] = df_features[feat] / df_features["Close"]
        
    return df_features, feature_cols
