import os
import pandas as pd
import numpy as np
from .generate_mock_data import generate_mock_dataset

class NiftyDataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Detect path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data")
        else:
            self.data_dir = data_dir
            
        self._ensure_data_exists()
        
    def _ensure_data_exists(self):
        """Checks if data directory and files exist. If not, generates mock data."""
        metadata_path = os.path.join(self.data_dir, "stock_metadata.csv")
        combined_path = os.path.join(self.data_dir, "NIFTY50_all.csv")
        
        if not os.path.exists(self.data_dir) or not os.path.exists(metadata_path) or not os.path.exists(combined_path):
            print("Data files not found. Generating mock data for NIFTY-50...")
            generate_mock_dataset(self.data_dir)
            
    def load_metadata(self):
        """Loads stock metadata details (Symbol, Company Name, Sector, Industry)."""
        path = os.path.join(self.data_dir, "stock_metadata.csv")
        df = pd.read_csv(path)
        if "Sector" not in df.columns and "Industry" in df.columns:
            df["Sector"] = df["Industry"]
        elif "Industry" not in df.columns and "Sector" in df.columns:
            df["Industry"] = df["Sector"]
        return df
        
    def load_stock_data(self, symbol):
        """Loads daily trading records for a single stock by symbol."""
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            # Try finding it in NIFTY50_all.csv if single file doesn't exist
            combined_path = os.path.join(self.data_dir, "NIFTY50_all.csv")
            if os.path.exists(combined_path):
                print(f"File {symbol}.csv not found. Loading from NIFTY50_all.csv...")
                full_df = pd.read_csv(combined_path)
                df = full_df[full_df["Symbol"] == symbol].copy()
                if df.empty:
                    raise FileNotFoundError(f"Stock symbol {symbol} not found in dataset.")
            else:
                raise FileNotFoundError(f"Data file for {symbol} not found.")
        else:
            df = pd.read_csv(file_path)
            
        # Parse Date and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Clean missing values: Forward fill first, then backward fill if there are leading NaNs
        numeric_cols = ["Open", "High", "Low", "Close", "Last", "VWAP", "Volume", "Turnover"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Consistent bounds check (High must be max, Low must be min)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
            
        return df

    def load_all_stocks(self):
        """Loads data for all available symbols as a dictionary of DataFrames."""
        metadata = self.load_metadata()
        symbols = metadata["Symbol"].unique()
        
        all_data = {}
        for sym in symbols:
            try:
                df = self.load_stock_data(sym)
                if len(df) < 100:
                    print(f"Warning: Stock {sym} has insufficient data ({len(df)} rows) and will be excluded.")
                else:
                    all_data[sym] = df
            except Exception as e:
                print(f"Warning: Could not load data for {sym}: {e}")
                
        return all_data

    def load_combined_index_data(self):
        """Loads the raw combined NIFTY50_all.csv dataframe."""
        path = os.path.join(self.data_dir, "NIFTY50_all.csv")
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
