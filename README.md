# NIFTY-50 AI-Powered Investment Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nifty50-investment-intelligence-4fcdepeafslwnzcguxsogh.streamlit.app/)

An intelligent decision-support system that transforms raw historical NIFTY-50 stock market data and metadata into actionable trading, risk assessment, and portfolio optimization insights.

---

## 1. Problem Overview
Modern stock markets generate overwhelming streams of daily price and volume data. For retail and institutional investors, turning this raw data into structured insights is a major challenge. 

This platform leverages 20+ years of historical NIFTY-50 market data (spanning January 2000 to April 2021) across critical Indian sectors (IT, Banking, Consumer Goods, Energy, Manufacturing, and Pharma) to provide a quantitative decision-support framework. The platform focuses on:
* **Predictive Forecasting**: Predicting 5-day forward price returns and movement direction.
* **Risk Assessment**: Calculating annualized volatility, Sharpe/Sortino ratios, Maximum Drawdowns, and systematic Risk Beta.
* **Diversified Portfolio Construction**: Formulating optimized portfolios tailored to Conservative, Balanced, and Aggressive profiles.
* **Explainability**: Applying machine learning feature importances and rulebased explanations for transparency.
* **Anomaly Detection**: Identifying extreme volatility spikes, volume surges, and drawdowns.

---

## 2. Quantitative Approach & Methodology

The platform relies on a data-driven pipeline:

1. **Feature Engineering**: Calculates a wide matrix of technical indicators:
   * *Trend*: 10-day, 50-day, and 200-day SMAs, EMA crossings.
   * *Momentum*: 14-day RSI, MACD histograms, Rate of Change (ROC).
   * *Volatility*: Bollinger Bands and 20-day annualized rolling volatility.
   * *Volume*: Rolling volume averages and ratios.

2. **Machine Learning Predictor Engine**:
   * *Target 1 (Return)*: Forecasts 5-day forward returns using a **Gradient Boosting Regressor** (`HistGradientBoostingRegressor`).
   * *Target 2 (Direction)*: Predicts whether the stock moves up or down using a **Gradient Boosting Classifier** (`HistGradientBoostingClassifier`).
   * *Evaluation*: Evaluates models using MAE, RMSE, and $R^2$ for regression, and Accuracy, F1-Score, and ROC AUC for classification. Uses a time-series split to prevent data leakage.

3. **Portfolio Optimization**:
   * *Conservative Profile*: Solves for the Minimum Variance portfolio using SciPy's sequential quadratic programming (`SLSQP`) solver.
   * *Balanced Profile*: Optimizes asset weights to maximize the portfolio Sharpe Ratio.
   * *Aggressive Profile*: Allocates capital based on highest-ranked expected stock returns and positive momentum indicators.

4. **Explainable AI (XAI)**:
   * Calculates **Permutation Feature Importance** on the test dataset to rank feature drivers.
   * Employs a rule-based expert system translating technical states into plain-English reasoning.

5. **Anomaly Detection**:
   * Flags daily returns exceeding 3 standard deviations from rolling averages (Volatility Spikes).
   * Flags daily volume exceeding 3x of the 20-day SMA (Volume Surges).
---

## 3. Repository Structure

```
nifty50_market_intelligence/
├── .devcontainer/               # VS Code Development Container configuration
│   └── devcontainer.json
├── data/                        # Market datasets (Kaggle, NSE, or synthetic)
│   ├── stock_metadata.csv       # Sector and company classifications
│   ├── NIFTY50_all.csv          # Combined index historical dataset
│   └── [SYMBOL].csv             # Individual stock history (e.g., RELIANCE.csv, TCS.csv)
├── models/                      # Saved artifacts (Trained models, scalers, pickle files)
├── src/                         # Core application source code
│   ├── __init__.py
│   ├── generate_mock_data.py    # Synthetic market data generator
│   ├── data_loader.py           # Parsing, cleaning, and interpolating data
│   ├── features.py              # Technical indicator calculation engine
│   ├── predictor.py             # Machine learning model training & inference
│   ├── portfolio.py             # Modern Portfolio Theory (MPT) weight optimization
│   ├── risk.py                  # Downside risk, Sortino, Drawdown, and Beta calculators
│   ├── anomaly.py               # Volatility spikes & volume surge detectors
│   ├── explainability.py        # Permutation importance & rule-based reasoning reports
│   └── test_suite.py            # Unit test suite verifying modules
├── .gitignore                   # Files and folders to ignore in Git (e.g., venv, __pycache__)
├── app.py                       # Main Streamlit web application dashboard
├── nifty50_exploration_and_training.ipynb  # Interactive EDA and model prototyping (Google Colab)
├── README.md                    # Project documentation and setup guide
├── requirements.txt             # Python project dependencies
└── technical report.pdf         # Comprehensive project documentation & analysis report
```

---

## 4. Setup & Installation

### Prerequisites
* Python 3.8 or higher installed on your system.

### Steps
1. **Clone or navigate** to the project workspace directory:
   ```bash
   git clone [https://github.com/tanvi257/nifty50-investment-intelligence.git](https://github.com/tanvi257/nifty50-investment-intelligence.git)
   cd nifty50-investment-intelligencel
   ```

2. **Install Python dependencies** listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup**:
   * *Option A (Mock Data)*: No setup is required. The platform automatically generates a realistic mock NIFTY-50 dataset in `data/` on first launch so that all tabs run immediately.
   * *Option B (Real Kaggle Data)*: Download the NIFTY-50 Stock Market Dataset by Rohan Rao from Kaggle, and extract the CSV files directly into the `data/` directory. The data loader will automatically parse the real files.

---

## 5. How to Run

### Run Unit Tests
To verify the codebase math and modules:
```bash
python src/test_suite.py
```

### Launch the Streamlit Interactive Dashboard
Run the following command to start the web app locally:
```bash
streamlit run app.py
```
After launching, open your browser and navigate to the local URL (typically `http://localhost:8501`).

---

## 6. Results
On the synthetic testing suite (verifiable via `test_suite.py` on the mock data), we achieve the following average metrics on unseen test splits:
* **Return Regressor**: Mean Absolute Error (MAE) of $\sim 2.1\%$, R² Score demonstrates positive predictive power.
* **Direction Classifier**: Directional accuracy ranges between $52\% - 61\%$, showing strong ability to capture sector-specific trends.

---

## 7. References & Tools
* **Dataset**: Rohan Rao's Kaggle NIFTY-50 Dataset ([Link](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data))
* **Optimization Theory**: Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.
* **Libraries**: 
  * `scikit-learn` for Gradient Boosting and evaluation metrics.
  * `SciPy` for numerical optimizations.
  * `Plotly` for interactive financial data plotting.
  * `Streamlit` for dashboard application deployment.
