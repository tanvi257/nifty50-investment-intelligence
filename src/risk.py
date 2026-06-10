import numpy as np
import pandas as pd

class RiskAssessor:
    def __init__(self, risk_free_rate=0.06):
        self.rf = risk_free_rate

    def calculate_stock_risk_metrics(self, df_stock, benchmark_returns=None):
        """Calculates historical risk metrics for a single stock.
        
        df_stock: DataFrame containing 'Close' prices and 'Date'
        benchmark_returns: Series or array of daily returns for the benchmark index (NIFTY 50)
        """
        prices = df_stock["Close"].values
        if len(prices) < 2:
            return {}
            
        daily_returns = df_stock["Close"].pct_change(fill_method=None).fillna(0).values
        
        # 1. Annualized Return (Geometric Mean)
        cum_ret = (prices[-1] / prices[0]) - 1
        num_years = len(prices) / 252.0
        annualized_return = (1 + cum_ret) ** (1 / (num_years if num_years > 0 else 1.0)) - 1
        
        # 2. Annualized Volatility
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # 3. Sharpe Ratio
        sharpe = (annualized_return - self.rf) / (volatility if volatility > 0 else 1e-9)
        
        # 4. Downside Volatility & Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_vol = np.std(negative_returns) * np.sqrt(252)
            sortino = (annualized_return - self.rf) / (downside_vol if downside_vol > 0 else 1e-9)
        else:
            downside_vol = 0.0
            sortino = np.nan
            
        # 5. Maximum Drawdown
        cum_returns = (1 + pd.Series(daily_returns)).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 6. Beta relative to benchmark
        beta = 1.0
        if benchmark_returns is not None:
            # If passed as a Pandas Series (with Date index), align by Date to handle mismatches
            if isinstance(benchmark_returns, pd.Series):
                stock_series = df_stock.set_index("Date")["Close"].pct_change(fill_method=None).fillna(0)
                combined = pd.concat([stock_series, benchmark_returns], axis=1).dropna()
                if not combined.empty:
                    cov = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
                    idx_var = np.var(combined.iloc[:, 1])
                    beta = cov[0, 1] / idx_var if idx_var > 0 else 1.0
            # Fallback for raw numpy arrays
            elif len(benchmark_returns) == len(daily_returns):
                cov = np.cov(daily_returns, benchmark_returns)
                idx_var = np.var(benchmark_returns)
                beta = cov[0, 1] / idx_var if idx_var > 0 else 1.0
            
        return {
            "Annualized_Return": annualized_return,
            "Volatility": volatility,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Max_Drawdown": max_drawdown,
            "Beta": beta
        }

    def calculate_portfolio_risk_metrics(self, weights, returns_df, benchmark_returns=None):
        """Calculates risk metrics for a portfolio given asset weights and daily returns."""
        # Calculate daily portfolio returns
        portfolio_daily_returns = np.dot(returns_df.values, weights)
        
        # Build a temporary dataframe of portfolio prices starting at 100
        p_prices = 100.0 * np.cumprod(1 + portfolio_daily_returns)
        p_df = pd.DataFrame({"Close": p_prices})
        
        # Add temporary dates index to align with benchmark if available
        if isinstance(benchmark_returns, pd.Series):
            p_df["Date"] = returns_df.index
        else:
            p_df["Date"] = pd.date_range(start="2018-01-01", periods=len(p_df), freq="B")
            
        metrics = self.calculate_stock_risk_metrics(p_df, benchmark_returns)
        return metrics
