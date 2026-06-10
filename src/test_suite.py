import unittest
import os
import pandas as pd
import numpy as np
import shutil

# Make absolute imports work relative to parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import NiftyDataLoader
from src.features import compute_technical_indicators, generate_predictor_features
from src.predictor import StockPredictorEngine
from src.portfolio import PortfolioConstructor
from src.risk import RiskAssessor
from src.anomaly import MarketAnomalyDetector
from src.explainability import ExplainabilityEngine

class TestNiftyInvestmentPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for tests
        cls.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data_dir")
        cls.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_models_dir")
        
        # Initialize data loader which will generate mock data in the test folder
        cls.loader = NiftyDataLoader(data_dir=cls.test_dir)
        cls.metadata = cls.loader.load_metadata()
        cls.all_data = cls.loader.load_all_stocks()
        cls.symbol = "RELIANCE"
        cls.df_reliance = cls.all_data[cls.symbol]
        
    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test directories
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        if os.path.exists(cls.models_dir):
            shutil.rmtree(cls.models_dir)

    def test_data_loader(self):
        """Verifies metadata and stock daily price details are loaded correctly."""
        self.assertIsNotNone(self.metadata)
        self.assertTrue("Symbol" in self.metadata.columns)
        self.assertTrue("Sector" in self.metadata.columns)
        
        self.assertFalse(self.df_reliance.empty)
        self.assertTrue("Close" in self.df_reliance.columns)
        self.assertTrue("Volume" in self.df_reliance.columns)
        self.assertEqual(self.df_reliance["Symbol"].iloc[0], self.symbol)

    def test_features(self):
        """Checks if rolling indicators are successfully computed without NaNs."""
        df_indicators = compute_technical_indicators(self.df_reliance)
        
        # Verify columns exist
        self.assertTrue("RSI_14" in df_indicators.columns)
        self.assertTrue("MACD" in df_indicators.columns)
        self.assertTrue("BB_Upper" in df_indicators.columns)
        self.assertTrue("Volatility_20d" in df_indicators.columns)
        
        # Verify no NaN values in indicators after cleaning
        self.assertFalse(df_indicators["RSI_14"].isna().any())
        self.assertFalse(df_indicators["BB_Middle"].isna().any())

    def test_predictor_engine(self):
        """Tests that models can be trained, saved, loaded, and used for scoring."""
        engine = StockPredictorEngine(models_dir=self.models_dir)
        reg_metrics, clf_metrics = engine.train_models(self.df_reliance, self.symbol)
        
        # Check that metrics dictionary has entries
        self.assertIn("MAE", reg_metrics)
        self.assertIn("R2", reg_metrics)
        self.assertIn("Accuracy", clf_metrics)
        
        # Run prediction
        pred = engine.predict(self.df_reliance, self.symbol)
        self.assertEqual(pred["Symbol"], self.symbol)
        self.assertIn(pred["Predicted_Direction"], ["UP", "DOWN"])
        self.assertGreaterEqual(pred["Direction_Probability"], 0.0)

    def test_portfolio_optimizer(self):
        """Checks weights calculation and justifications for all profiles."""
        constructor = PortfolioConstructor(self.all_data, self.metadata)
        
        # Conservative
        p_cons = constructor.construct_portfolio("Conservative")
        self.assertEqual(p_cons["Profile"], "Conservative")
        self.assertGreater(p_cons["Expected_Annualized_Return"], 0)
        self.assertGreater(p_cons["Expected_Annualized_Volatility"], 0)
        self.assertIsNotNone(p_cons["Justification"])
        
        # Balanced
        p_bal = constructor.construct_portfolio("Balanced")
        self.assertEqual(p_bal["Profile"], "Balanced")
        
        # Aggressive
        p_agg = constructor.construct_portfolio("Aggressive")
        self.assertEqual(p_agg["Profile"], "Aggressive")

    def test_risk_assessor(self):
        """Verifies Sharpe Ratio and Max Drawdown calculation results."""
        assessor = RiskAssessor()
        metrics = assessor.calculate_stock_risk_metrics(self.df_reliance)
        
        self.assertIn("Sharpe_Ratio", metrics)
        self.assertIn("Max_Drawdown", metrics)
        self.assertIn("Sortino_Ratio", metrics)
        self.assertLessEqual(metrics["Max_Drawdown"], 0) # Drawdowns are negative or 0

    def test_anomaly_detector(self):
        """Verifies anomalies table returns expected columns."""
        detector = MarketAnomalyDetector()
        anomalies_df = detector.detect_anomalies(self.df_reliance)
        
        self.assertTrue(isinstance(anomalies_df, pd.DataFrame))
        self.assertTrue("Type" in anomalies_df.columns)
        self.assertTrue("Severity" in anomalies_df.columns)

    def test_explainability_engine(self):
        """Tests Feature Importance calculation and text explanations."""
        engine = StockPredictorEngine(models_dir=self.models_dir)
        engine.train_models(self.df_reliance, self.symbol)
        reg_model, feat_cols, _ = engine.load_model(self.symbol, "regressor")
        
        explainer = ExplainabilityEngine()
        
        # Feature Importance
        imp_df = explainer.compute_model_feature_importance(reg_model, self.df_reliance, feat_cols)
        self.assertFalse(imp_df.empty)
        self.assertEqual(imp_df.iloc[0]["Feature"], imp_df.sort_values(by="Importance_Mean", ascending=False).iloc[0]["Feature"])
        
        # Rule-based Explanations
        latest_row = compute_technical_indicators(self.df_reliance).tail(1)
        reasoning = explainer.generate_financial_reasoning(latest_row)
        self.assertIn(reasoning["Sentiment"], ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertGreater(len(reasoning["Bullet_Points"]), 0)

if __name__ == "__main__":
    unittest.main()
