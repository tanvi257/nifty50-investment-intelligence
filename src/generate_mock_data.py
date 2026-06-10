import os
import pandas as pd
import numpy as np
import datetime as dt

def generate_stock_price_path(start_price, num_days, drift=0.0002, volatility=0.015):
    """Generates a realistic stock price path using Geometric Brownian Motion."""
    returns = np.random.normal(drift, volatility, num_days)
    price_path = start_price * np.exp(np.cumsum(returns))
    return price_path

def generate_mock_dataset(target_dir="data"):
    # Enforce deterministic behavior for numpy random selections to ensure reproducibility
    np.random.seed(42)
    os.makedirs(target_dir, exist_ok=True)
    
    # Define NIFTY-50 stocks metadata
    metadata = [
        {"Symbol": "RELIANCE", "Company Name": "Reliance Industries Ltd.", "Sector": "Energy", "Industry": "Oil & Gas"},
        {"Symbol": "TCS", "Company Name": "Tata Consultancy Services Ltd.", "Sector": "Information Technology", "Industry": "IT Services"},
        {"Symbol": "INFY", "Company Name": "Infosys Ltd.", "Sector": "Information Technology", "Industry": "IT Services"},
        {"Symbol": "HDFCBANK", "Company Name": "HDFC Bank Ltd.", "Sector": "Banking", "Industry": "Private Bank"},
        {"Symbol": "SBIN", "Company Name": "State Bank of India", "Sector": "Banking", "Industry": "Public Bank"},
        {"Symbol": "ITC", "Company Name": "ITC Ltd.", "Sector": "Consumer Goods", "Industry": "Cigarettes & FMCG"},
        {"Symbol": "HINDUNILVR", "Company Name": "Hindustan Unilever Ltd.", "Sector": "Consumer Goods", "Industry": "FMCG"},
        {"Symbol": "SUNPHARMA", "Company Name": "Sun Pharmaceutical Industries Ltd.", "Sector": "Pharmaceuticals", "Industry": "Generics"},
        {"Symbol": "TATAMOTORS", "Company Name": "Tata Motors Ltd.", "Sector": "Manufacturing", "Industry": "Automobiles"},
        {"Symbol": "LTI", "Company Name": "Larsen & Toubro Infotech Ltd.", "Sector": "Information Technology", "Industry": "IT Services"},
    ]
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(target_dir, "stock_metadata.csv")
    if not os.path.exists(metadata_path):
        metadata_df.to_csv(metadata_path, index=False)
        print("Generated stock_metadata.csv")
        
    # Generate daily records from 2018-01-01 to 2021-04-30
    start_date = dt.date(2018, 1, 1)
    end_date = dt.date(2021, 4, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    num_days = len(date_range)
    
    all_stocks_data = []
    
    # Starting prices for stocks to keep them realistic
    start_prices = {
        "RELIANCE": 1000.0,
        "TCS": 2500.0,
        "INFY": 800.0,
        "HDFCBANK": 900.0,
        "SBIN": 250.0,
        "ITC": 260.0,
        "HINDUNILVR": 1800.0,
        "SUNPHARMA": 450.0,
        "TATAMOTORS": 300.0,
        "LTI": 1500.0
    }
    
    for stock in metadata:
        symbol = stock["Symbol"]
        stock_file_path = os.path.join(target_dir, f"{symbol}.csv")
        
        # Determine drift and volatility depending on sector to simulate realistic behaviors
        drift = 0.0001
        vol = 0.015
        if stock["Sector"] == "Information Technology":
            drift = 0.0003
            vol = 0.018
        elif stock["Sector"] == "Banking":
            drift = 0.00015
            vol = 0.02
        elif stock["Sector"] == "Consumer Goods":
            drift = 0.00008
            vol = 0.01
            
        prices = generate_stock_price_path(start_prices[symbol], num_days, drift=drift, volatility=vol)
        
        # Create OHLC from generated Close prices
        df = pd.DataFrame(index=date_range)
        df["Date"] = date_range.strftime("%Y-%m-%d")
        df["Symbol"] = symbol
        df["Series"] = "EQ"
        df["Close"] = prices
        
        # Introduce daily high/low/open variations
        df["Open"] = df["Close"].shift(1)
        df["Open"] = df["Open"].fillna(start_prices[symbol])
        # Add random noise to Open
        df["Open"] = df["Open"] * (1 + np.random.normal(0, 0.003, num_days))
        
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.abs(np.random.normal(0.005, 0.004, num_days)))
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.abs(np.random.normal(0.005, 0.004, num_days)))
        df["Last"] = df["Close"] * (1 + np.random.normal(0, 0.001, num_days))
        df["Prev Close"] = df["Close"].shift(1)
        df["Prev Close"] = df["Prev Close"].fillna(start_prices[symbol] * 0.99)
        
        # VWAP estimation
        df["VWAP"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
        
        # Volume generation (log-normal distribution)
        base_volume = {
            "RELIANCE": 5000000,
            "TCS": 2000000,
            "INFY": 4000000,
            "HDFCBANK": 3000000,
            "SBIN": 10000000,
            "ITC": 8000000,
            "HINDUNILVR": 1500000,
            "SUNPHARMA": 2500000,
            "TATAMOTORS": 12000000,
            "LTI": 500000
        }
        
        vols_noise = np.random.lognormal(0.5, 0.4, num_days)
        df["Volume"] = (base_volume[symbol] * vols_noise).astype(int)
        
        # Introduce a few volume and volatility spikes (anomalies)
        anomaly_indices = np.random.choice(num_days, size=int(num_days * 0.02), replace=False)
        for idx in anomaly_indices:
            # Volatility spike
            df.iloc[idx, df.columns.get_loc("High")] *= 1.08
            df.iloc[idx, df.columns.get_loc("Low")] *= 0.92
            # Volume spike
            df.iloc[idx, df.columns.get_loc("Volume")] *= 4
            # Update Close/Last/VWAP accordingly
            df.iloc[idx, df.columns.get_loc("Close")] = df.iloc[idx, df.columns.get_loc("Open")] * (1 + np.random.normal(0, 0.04))
            df.iloc[idx, df.columns.get_loc("Last")] = df.iloc[idx, df.columns.get_loc("Close")]
            df.iloc[idx, df.columns.get_loc("VWAP")] = (df.iloc[idx, df.columns.get_loc("Open")] + df.iloc[idx, df.columns.get_loc("High")] + df.iloc[idx, df.columns.get_loc("Low")] + df.iloc[idx, df.columns.get_loc("Close")]) / 4

        df["Turnover"] = df["Volume"] * df["VWAP"] * 100000  # scaled turnover representation
        df["Trades"] = (df["Volume"] * np.random.uniform(0.02, 0.08, num_days)).astype(int)
        df["Deliverable Volume"] = (df["Volume"] * np.random.uniform(0.3, 0.6, num_days)).astype(int)
        df["%Deliverble"] = df["Deliverable Volume"] / df["Volume"]
        
        # Save individual stock file if it doesn't exist
        if not os.path.exists(stock_file_path):
            df.to_csv(stock_file_path, index=False)
            print(f"Generated data/{symbol}.csv")
            
        all_stocks_data.append(df)
        
    # Generate combined NIFTY50_all.csv
    combined_path = os.path.join(target_dir, "NIFTY50_all.csv")
    if not os.path.exists(combined_path):
        combined_df = pd.concat(all_stocks_data, ignore_index=True)
        # Sort by Date and Symbol
        combined_df = combined_df.sort_values(by=["Date", "Symbol"])
        combined_df.to_csv(combined_path, index=False)
        print("Generated NIFTY50_all.csv")
        
    print("All mock files successfully generated/verified.")

if __name__ == "__main__":
    generate_mock_dataset()
