import pandas as pd
import numpy as np

class MarketAnomalyDetector:
    def __init__(self, vol_window=20, vol_threshold=3.0, volume_threshold=3.0, return_threshold=-0.035):
        """
        vol_threshold: Number of standard deviations for volatility spike
        volume_threshold: Multiplier for volume surge (e.g. 3x 20-day SMA)
        return_threshold: Minimum daily return for extreme drawdown (-3.5% default)
        """
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.volume_threshold = volume_threshold
        self.return_threshold = return_threshold

    def detect_anomalies(self, df_stock):
        """Scans stock data and returns a list of detected anomaly incidents."""
        df = df_stock.copy()
        
        # Ensure returns are computed
        if "Return_1d" not in df.columns:
            df["Return_1d"] = df["Close"].pct_change(fill_method=None).fillna(0)
            
        # 1. Compute rolling statistics
        df["Rolling_Mean_Return"] = df["Return_1d"].rolling(window=self.vol_window).mean()
        df["Rolling_Std_Return"] = df["Return_1d"].rolling(window=self.vol_window).std().fillna(0)
        
        df["Rolling_Mean_Volume"] = df["Volume"].rolling(window=self.vol_window).mean().fillna(df["Volume"])
        
        anomalies = []
        
        # Scan each row starting after rolling window size
        for i in range(self.vol_window, len(df)):
            row = df.iloc[i]
            date_str = row["Date"].strftime("%Y-%m-%d") if isinstance(row["Date"], pd.Timestamp) else str(row["Date"])
            
            # Volatility Spike Check
            mean_ret = row["Rolling_Mean_Return"]
            std_ret = row["Rolling_Std_Return"]
            daily_ret = row["Return_1d"]
            
            if std_ret > 0 and np.abs(daily_ret - mean_ret) > (self.vol_threshold * std_ret):
                anomaly_type = "Volatility Spike"
                severity = "Medium" if np.abs(daily_ret - mean_ret) < 4 * std_ret else "High"
                anomalies.append({
                    "Date": date_str,
                    "Symbol": row["Symbol"],
                    "Type": anomaly_type,
                    "Metric": f"Daily Return: {daily_ret*100:.2f}%",
                    "Details": f"Return deviated by {(daily_ret - mean_ret)/std_ret:.1f} standard deviations from {self.vol_window}-day mean.",
                    "Severity": severity
                })
                
            # Volume Surge Check
            daily_vol = row["Volume"]
            mean_vol = row["Rolling_Mean_Volume"]
            
            if mean_vol > 0 and daily_vol > (self.volume_threshold * mean_vol):
                anomaly_type = "Volume Surge"
                severity = "Medium" if daily_vol < 5 * mean_vol else "High"
                anomalies.append({
                    "Date": date_str,
                    "Symbol": row["Symbol"],
                    "Type": anomaly_type,
                    "Metric": f"Volume: {daily_vol:,} shares",
                    "Details": f"Volume was {daily_vol/mean_vol:.1f}x higher than the {self.vol_window}-day average of {int(mean_vol):,}.",
                    "Severity": severity
                })
                
            # Extreme Drawdown Check
            if daily_ret <= self.return_threshold:
                anomaly_type = "Extreme Drawdown"
                severity = "High" if daily_ret <= -0.05 else "Medium"
                anomalies.append({
                    "Date": date_str,
                    "Symbol": row["Symbol"],
                    "Type": anomaly_type,
                    "Metric": f"Daily Drop: {daily_ret*100:.2f}%",
                    "Details": f"Daily price decline fell below the critical risk threshold of {self.return_threshold*100:.1f}%.",
                    "Severity": severity
                })
                
        # Return as DataFrame sorted by date descending
        if len(anomalies) > 0:
            return pd.DataFrame(anomalies).sort_values(by="Date", ascending=False).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=["Date", "Symbol", "Type", "Metric", "Details", "Severity"])
