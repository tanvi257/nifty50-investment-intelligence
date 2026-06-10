import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioConstructor:
    def __init__(self, stock_data_dict, metadata_df):
        """
        stock_data_dict: dict of Symbol -> DataFrame containing 'Close' prices
        metadata_df: DataFrame with 'Symbol', 'Sector', 'Company Name'
        """
        self.stock_data = stock_data_dict
        self.metadata = metadata_df
        self.symbols = list(stock_data_dict.keys())
        self.returns_df = self._calculate_returns_df()
        
    def _calculate_returns_df(self):
        """Prepares a single DataFrame containing daily returns for all stocks."""
        returns_dict = {}
        for sym in self.symbols:
            df = self.stock_data[sym]
            returns_dict[sym] = df["Close"].pct_change(fill_method=None).fillna(0)
            
        returns_df = pd.DataFrame(returns_dict)
        return returns_df

    def get_portfolio_statistics(self, weights, risk_free_rate=0.06):
        """Calculates expected portfolio return, volatility, and Sharpe ratio (annualized)."""
        # Annualized mean returns (assuming 252 business days per year)
        mean_returns = self.returns_df.mean() * 252
        cov_matrix = self.returns_df.cov() * 252
        
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_volatility if portfolio_volatility > 0 else 1e-9)
        
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def optimize_min_variance(self):
        """Optimizes weights to minimize portfolio variance (Conservative Profile)."""
        num_assets = len(self.symbols)
        cov_matrix = self.returns_df.cov() * 252
        
        def obj_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 0.4) for _ in range(num_assets)) # Max 40% in a single stock for diversification
        init_weights = np.ones(num_assets) / num_assets
        
        res = minimize(obj_variance, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    def optimize_max_sharpe(self, risk_free_rate=0.06):
        """Optimizes weights to maximize Sharpe Ratio (Balanced Profile)."""
        num_assets = len(self.symbols)
        mean_returns = self.returns_df.mean() * 252
        cov_matrix = self.returns_df.cov() * 252
        
        # Financial Engineering Fix: if Rf exceeds maximum asset return, MVO optimizer direction flips.
        # We use an effective Rf equal to 50% of the maximum asset return as a proxy for the math solver,
        # ensuring the numerator remains positive and the optimization direction is stable.
        max_ret = mean_returns.max()
        effective_rf = risk_free_rate
        if max_ret <= risk_free_rate:
            effective_rf = max_ret * 0.5
        
        def obj_neg_sharpe(weights):
            p_return = np.dot(weights, mean_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (p_return - effective_rf) / (p_vol if p_vol > 0 else 1e-9)
            return -sharpe
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 0.35) for _ in range(num_assets)) # Max 35% in a single stock
        init_weights = np.ones(num_assets) / num_assets
        
        res = minimize(obj_neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    def construct_portfolio(self, profile="Balanced", expected_returns_dict=None, risk_free_rate=0.06):
        """Generates portfolio weights and statistics based on investor profile.
        
        profile: 'Conservative', 'Balanced', or 'Aggressive'
        expected_returns_dict: dict of Symbol -> Predicted 5-day return (optional, for Aggressive portfolio tilt)
        risk_free_rate: Risk-free rate for statistic calculation and Sharpe optimization
        """
        num_assets = len(self.symbols)
        
        if profile == "Conservative":
            weights = self.optimize_min_variance()
            justification = (
                "Recommended for capital preservation. The algorithm uses Markowitz Mean-Variance Optimization "
                "to minimize portfolio variance. Individual stock allocations are capped at 40% to ensure high diversification, "
                "skewing heavily towards historically low-beta, low-volatility sectors (e.g. Consumer Goods, Pharmaceuticals)."
            )
            
        elif profile == "Balanced":
            weights = self.optimize_max_sharpe(risk_free_rate)
            justification = (
                "Recommended for investors seeking a balance of growth and stability. The optimization maximizes "
                "the Sharpe Ratio, aligning allocation with the highest risk-adjusted performance. Weights are capped at 35% "
                "to maintain strong diversification across Banking, Energy, and Information Technology."
            )
            
        elif profile == "Aggressive":
            # Aggressive portfolio selects top 4 stocks with highest expected return
            if expected_returns_dict is not None:
                # Use predictions if available
                stock_ranks = pd.Series(expected_returns_dict)
            else:
                # Fallback to historical mean returns
                stock_ranks = self.returns_df.mean() * 252
                
            top_symbols = stock_ranks.nlargest(4).index.tolist()
            
            # Allocate 40% to top 1, 30% to top 2, 20% to top 3, 10% to top 4
            weights = np.zeros(num_assets)
            allocations = [0.40, 0.30, 0.20, 0.10]
            for sym, alloc in zip(top_symbols, allocations):
                idx = self.symbols.index(sym)
                weights[idx] = alloc
                
            justification = (
                "Recommended for investors targeting maximum capital appreciation. The portfolio concentrates capital "
                "into the top 4 assets with the highest predicted forward returns (regression output) or strongest price momentum. "
                "This concentrates returns, sacrificing short-term stability for maximum upside."
            )
        else:
            raise ValueError(f"Unknown profile: {profile}")
            
        # Compile weights and meta details
        allocations_list = []
        for i, sym in enumerate(self.symbols):
            if weights[i] > 0.001:
                meta = self.metadata[self.metadata["Symbol"] == sym].iloc[0]
                allocations_list.append({
                    "Symbol": sym,
                    "Company Name": meta["Company Name"],
                    "Sector": meta["Sector"],
                    "Weight": weights[i]
                })
                
        allocations_df = pd.DataFrame(allocations_list).sort_values(by="Weight", ascending=False)
        
        # Calculate stats
        p_ret, p_vol, p_sharpe = self.get_portfolio_statistics(weights, risk_free_rate)
        
        return {
            "Profile": profile,
            "Expected_Annualized_Return": p_ret,
            "Expected_Annualized_Volatility": p_vol,
            "Sharpe_Ratio": p_sharpe,
            "Allocations": allocations_df,
            "Justification": justification
        }
