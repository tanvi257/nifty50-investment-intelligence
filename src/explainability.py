import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .features import generate_predictor_features

class ExplainabilityEngine:
    def __init__(self):
        pass

    def compute_model_feature_importance(self, model, df_stock, feature_cols, target_col="Target_Return"):
        """Computes Permutation Feature Importance for any scikit-learn estimator.
        
        Permutation importance measures how much the scoring metric decreases when a feature is randomized.
        """
        df_feat, _ = generate_predictor_features(df_stock)
        
        # Test on the last 20% of data (similar to our validation split)
        split_idx = int(len(df_feat) * 0.8)
        test_df = df_feat.iloc[split_idx:]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Calculate permutation importance
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance_Mean": r.importances_mean,
            "Importance_Std": r.importances_std
        }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)
        
        return importance_df

    def generate_financial_reasoning(self, latest_row):
        """Generates text explanations of the technical indicator states."""
        reasons = []
        signals = []
        
        # 1. RSI Check
        rsi = latest_row["RSI_14"].values[0]
        if rsi < 30:
            signals.append("BULLISH")
            reasons.append(f"RSI is oversold at {rsi:.1f} (< 30), suggesting a strong potential for a price reversal upward.")
        elif rsi > 70:
            signals.append("BEARISH")
            reasons.append(f"RSI is overbought at {rsi:.1f} (> 70), indicating the stock might be overvalued in the short term.")
        else:
            reasons.append(f"RSI is at a neutral level of {rsi:.1f}, reflecting steady trading momentum.")
            
        # 2. MACD Crossover Check
        macd = latest_row["MACD"].values[0]
        signal = latest_row["MACD_Signal"].values[0]
        hist = latest_row["MACD_Hist"].values[0]
        
        if macd > signal and hist > 0:
            signals.append("BULLISH")
            reasons.append("MACD is above its signal line with expanding positive bars, indicating strong upward momentum.")
        elif macd < signal and hist < 0:
            signals.append("BEARISH")
            reasons.append("MACD has crossed below its signal line, demonstrating building downward pressure.")
        else:
            reasons.append("MACD is converging with the signal line, indicating a potential consolidation phase.")
            
        # 3. Bollinger Bands Check
        close = latest_row["Close"].values[0]
        bb_upper = latest_row["BB_Upper"].values[0]
        bb_lower = latest_row["BB_Lower"].values[0]
        bb_mid = latest_row["BB_Middle"].values[0]
        
        if close <= bb_lower * 1.01:
            signals.append("BULLISH")
            reasons.append(f"Price ({close:.2f}) is trading near or below the lower Bollinger Band ({bb_lower:.2f}), suggesting a mean-reversion buying opportunity.")
        elif close >= bb_upper * 0.99:
            signals.append("BEARISH")
            reasons.append(f"Price ({close:.2f}) is trading near or above the upper Bollinger Band ({bb_upper:.2f}), suggesting a resistance zone.")
        else:
            pct_band = (close - bb_lower) / (bb_upper - bb_lower + 1e-9) * 100
            reasons.append(f"Price is stable within the Bollinger Bands, trading at the {pct_band:.1f}% percentile of the range.")
            
        # 4. Volatility Check
        vol = latest_row["Volatility_20d"].values[0]
        if vol > 0.35:
            reasons.append(f"Historical annualized volatility is elevated at {vol*100:.1f}%. Expect larger price swings and manage position sizes accordingly.")
        else:
            reasons.append(f"Historical annualized volatility is low to moderate at {vol*100:.1f}%, indicating stable price behavior.")
            
        # Determine overall technical sentiment
        bullish_count = signals.count("BULLISH")
        bearish_count = signals.count("BEARISH")
        
        if bullish_count > bearish_count:
            sentiment = "BULLISH"
        elif bearish_count > bullish_count:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
            
        return {
            "Sentiment": sentiment,
            "Bullet_Points": reasons
        }
