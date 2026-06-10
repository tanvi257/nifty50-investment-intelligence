import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from .features import generate_predictor_features

class StockPredictorEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_dir = os.path.join(base_dir, "models")
        else:
            self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
    def train_models(self, df_stock, symbol, train_split=0.8):
        """Trains both the Return Regressor and Direction Classifier for a specific stock."""
        df_feat, feature_cols = generate_predictor_features(df_stock)
        
        if len(df_feat) < 100:
            raise ValueError(f"Insufficient data for training {symbol} ({len(df_feat)} rows).")
            
        # Time-series split (no random shuffling to prevent data leakage)
        split_idx = int(len(df_feat) * train_split)
        train_df = df_feat.iloc[:split_idx]
        test_df = df_feat.iloc[split_idx:]
        
        X_train, y_train_ret, y_train_dir = train_df[feature_cols], train_df["Target_Return"], train_df["Target_Direction"]
        X_test, y_test_ret, y_test_dir = test_df[feature_cols], test_df["Target_Return"], test_df["Target_Direction"]
        
        # 1. Train Regressor (predict forward returns)
        reg_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        reg_model.fit(X_train, y_train_ret)
        
        # 2. Train Classifier (predict direction)
        clf_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        clf_model.fit(X_train, y_train_dir)
        
        # Evaluate Regressor
        reg_preds = reg_model.predict(X_test)
        reg_metrics = {
            "MAE": float(mean_absolute_error(y_test_ret, reg_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_ret, reg_preds))),
            "R2": float(r2_score(y_test_ret, reg_preds))
        }
        
        # Evaluate Classifier
        clf_preds = clf_model.predict(X_test)
        clf_metrics = {
            "Accuracy": float(accuracy_score(y_test_dir, clf_preds)),
            "F1_Score": float(f1_score(y_test_dir, clf_preds, average='binary'))
        }
        
        # Save models and metrics to disk
        reg_path = os.path.join(self.models_dir, f"{symbol}_regressor.pkl")
        clf_path = os.path.join(self.models_dir, f"{symbol}_classifier.pkl")
        
        with open(reg_path, 'wb') as f:
            pickle.dump((reg_model, feature_cols, reg_metrics), f)
            
        with open(clf_path, 'wb') as f:
            pickle.dump((clf_model, feature_cols, clf_metrics), f)
            
        return reg_metrics, clf_metrics

    def load_model(self, symbol, model_type="regressor"):
        """Loads a pre-trained model and its evaluation metrics from disk."""
        model_path = os.path.join(self.models_dir, f"{symbol}_{model_type}.pkl")
        if not os.path.exists(model_path):
            return None, None, None
            
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 3:
                model, feature_cols, metrics = data
            elif isinstance(data, tuple) and len(data) == 2:
                model, feature_cols = data
                metrics = {}
            else:
                model = data
                feature_cols = []
                metrics = {}
            
        return model, feature_cols, metrics

    def predict(self, df_latest, symbol):
        """Generates future return and direction prediction for a stock using the latest available data."""
        reg_model, feat_cols, _ = self.load_model(symbol, "regressor")
        clf_model, _, _ = self.load_model(symbol, "classifier")
        
        # If models are not trained yet, train them automatically on the incoming dataframe
        if reg_model is None or clf_model is None:
            print(f"Models for {symbol} not found. Training on historical data...")
            self.train_models(df_latest, symbol)
            reg_model, feat_cols, _ = self.load_model(symbol, "regressor")
            clf_model, _, _ = self.load_model(symbol, "classifier")
            
        # Get indicators for latest records
        from .features import compute_technical_indicators
        df_indicators = compute_technical_indicators(df_latest)
        
        # Prepare the features for the final row
        latest_row = df_indicators.tail(1).copy()
        
        # Normalize the latest row features using the latest close price
        close_val = latest_row["Close"].values[0]
        price_features = ["SMA_10", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "EMA_50", "BB_Middle", "BB_Upper", "BB_Lower"]
        
        for feat in price_features:
            latest_row[feat] = latest_row[feat] / close_val
            
        X_latest = latest_row[feat_cols]
        
        predicted_return = reg_model.predict(X_latest)[0]
        predicted_direction = clf_model.predict(X_latest)[0]
        proba_direction = clf_model.predict_proba(X_latest)[0]  # [proba_0, proba_1]
        
        return {
            "Symbol": symbol,
            "Date": latest_row["Date"].values[0],
            "Current_Close": close_val,
            "Predicted_5d_Return": predicted_return,
            "Predicted_Direction": "UP" if predicted_direction == 1 else "DOWN",
            "Direction_Probability": proba_direction[1] if predicted_direction == 1 else proba_direction[0]
        }
