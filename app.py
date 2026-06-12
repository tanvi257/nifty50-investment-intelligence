import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import NiftyDataLoader
from src.features import compute_technical_indicators
from src.predictor import StockPredictorEngine
from src.portfolio import PortfolioConstructor
from src.risk import RiskAssessor
from src.anomaly import MarketAnomalyDetector
from src.explainability import ExplainabilityEngine

# Page setup
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Custom CSS overides for dark theme
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #F8FAFC;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data and model helpers
@st.cache_resource
def init_platform_components():
    loader = NiftyDataLoader()
    metadata = loader.load_metadata()
    stock_data = loader.load_all_stocks()
    predictor = StockPredictorEngine()
    
    for sym in stock_data.keys():
        reg_model, _, _ = predictor.load_model(sym, "regressor")
        clf_model, _, _ = predictor.load_model(sym, "classifier")
        if reg_model is None or clf_model is None:
            try:
                predictor.train_models(stock_data[sym], sym)
            except Exception as e:
                print(f"Warning: Failed to train models for {sym}: {e}")
            
    risk_assessor = RiskAssessor()
    anomaly_detector = MarketAnomalyDetector()
    explainer = ExplainabilityEngine()
    
    return loader, metadata, stock_data, predictor, risk_assessor, anomaly_detector, explainer

loader, metadata, stock_data, predictor, risk_assessor, anomaly_detector, explainer = init_platform_components()

# Cache model predictions
@st.cache_data
def get_predictions_and_metrics(symbol):
    df_stock = stock_data[symbol]
    
    # Load model and pre-saved metrics directly from disk 
    reg_model, feat_cols, reg_metrics = predictor.load_model(symbol, "regressor")
    clf_model, _, clf_metrics = predictor.load_model(symbol, "classifier")
    
    if reg_model is None or clf_model is None:
        predictor.train_models(df_stock, symbol)
        reg_model, feat_cols, reg_metrics = predictor.load_model(symbol, "regressor")
        clf_model, _, clf_metrics = predictor.load_model(symbol, "classifier")
        
    pred_res = predictor.predict(df_stock, symbol)
    importance_df = explainer.compute_model_feature_importance(reg_model, df_stock, feat_cols)
    
    return pred_res, reg_metrics, clf_metrics, importance_df

# Cache anomaly detection across all stocks
@st.cache_data
def run_global_anomaly_detection():
    all_anomalies = []
    for sym, df in stock_data.items():
        anoms = anomaly_detector.detect_anomalies(df)
        if not anoms.empty:
            all_anomalies.append(anoms)
            
    if all_anomalies:
        return pd.concat(all_anomalies, ignore_index=True).sort_values(by="Date", ascending=False).reset_index(drop=True)
    return pd.DataFrame()

# Cache heavy portfolio optimization and covariance computations
@st.cache_data
def get_portfolio_allocations(profile, risk_free_rate):
    p_constructor = PortfolioConstructor(stock_data, metadata)
    
    expected_returns_dict = {}
    for sym in stock_data.keys():
        try:
            # Predict 5day returns 
            p_res = predictor.predict(stock_data[sym], sym)
            expected_returns_dict[sym] = p_res["Predicted_5d_Return"]
        except:
            expected_returns_dict[sym] = 0.0
            
    portfolio = p_constructor.construct_portfolio(profile, expected_returns_dict, risk_free_rate)
    cov_matrix = p_constructor.returns_df.cov() * 252 
    
    return portfolio, cov_matrix

global_anomalies = run_global_anomaly_detection()

# Sidebar Navigation and Controls
st.sidebar.markdown("<h2 style='margin-top:0;'>⚙️ Settings</h2>", unsafe_allow_html=True)

selected_symbol = st.sidebar.selectbox(
    "Select Target Stock",
    options=sorted(list(stock_data.keys())),
    index=0
)

investor_profile = st.sidebar.selectbox(
    "Investor Risk Profile",
    options=["Conservative", "Balanced", "Aggressive"],
    index=1
)

risk_free_input = st.sidebar.slider(
    "Risk-Free Interest Rate (%)",
    min_value=0.0,
    max_value=12.0,
    value=6.75,
    step=0.25
) / 100.0

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Retrain Models", use_container_width=True):
    st.sidebar.warning("Retraining models... please wait.")
    for sym in stock_data.keys():
        predictor.train_models(stock_data[sym], sym)
    st.cache_data.clear()
    st.sidebar.success("All models successfully retrained!")
    st.rerun()

st.sidebar.markdown(
    "<div style='margin-top:30px; font-size:0.8rem; color:#64748B; text-align:center;'>"
    "<b>NIFTY-50 AI Platform v1.0</b><br/>"
    "Decision Support Engine"
    "</div>",
    unsafe_allow_html=True
)

# Main Dashboard Layout
st.markdown("<h1 style='margin-bottom:0px;'>📈 NIFTY-50 Market Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:1.1rem; margin-top:5px; margin-bottom:25px;'>"
            "Deep analytics, predictive forecasting, risk monitoring, and portfolio optimization for NSE India blue-chips."
            "</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Market Overview & Anomalies", 
    "🤖 Stock Predictor & Explainable AI", 
    "💼 Portfolio Construction",
    "🎯 Personalized Advisor"
])

df_selected = stock_data[selected_symbol]
meta_selected = metadata[metadata["Symbol"] == selected_symbol].iloc[0]

# TAB 1: MARKET OVERVIEW & ANOMALIES 
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{meta_selected['Company Name']} ({selected_symbol}) — Price & Technical Overlays")
        df_indicators = compute_technical_indicators(df_selected)
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df_indicators['Date'],
            open=df_indicators['Open'],
            high=df_indicators['High'],
            low=df_indicators['Low'],
            close=df_indicators['Close'],
            name="Price"
        ))
     
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['SMA_10'], line=dict(color='#38BDF8', width=1.2), name="SMA 10"))
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['SMA_50'], line=dict(color='#F59E0B', width=1.5), name="SMA 50"))
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['SMA_200'], line=dict(color='#EF4444', width=1.8), name="SMA 200"))
        
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['BB_Upper'], line=dict(color='#10B981', width=0.8, dash='dash'), name="Bollinger Upper"))
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['BB_Lower'], line=dict(color='#10B981', width=0.8, dash='dash'), name="Bollinger Lower"))
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Key Performance Stats")
        
        # Risk Calculations
        stock_risk_assessor = RiskAssessor(risk_free_rate=risk_free_input)
        returns_dict = {}
        for sym, df in stock_data.items():
            returns_dict[sym] = df.set_index("Date")["Close"].pct_change(fill_method=None).fillna(0)
        returns_df = pd.DataFrame(returns_dict)
        benchmark_series = returns_df.mean(axis=1)
        
        risk_metrics = stock_risk_assessor.calculate_stock_risk_metrics(df_selected, benchmark_series)
 
        last_close = df_selected["Close"].iloc[-1]
        prev_close = df_selected["Prev Close"].iloc[-1]
        price_change = ((last_close - prev_close) / prev_close) * 100
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:15px;">
            <div class="metric-label">Current Close Price</div>
            <div class="metric-value">₹{last_close:,.2f} <span style="font-size:1.1rem; color:{'#10B981' if price_change >= 0 else '#EF4444'};">({price_change:+.2f}%)</span></div>
            <div style="font-size:0.85rem; color:#64748B;">Sector: {meta_selected['Sector']} | Industry: {meta_selected['Industry']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:15px;">
            <div class="metric-label">Annualized Return</div>
            <div class="metric-value">{risk_metrics.get('Annualized_Return', 0)*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:15px;">
            <div class="metric-label">Annualized Volatility</div>
            <div class="metric-value">{risk_metrics.get('Volatility', 0)*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
   
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.metric("Sharpe Ratio", f"{risk_metrics.get('Sharpe_Ratio', 0):.2f}")
            st.metric("Beta vs Nifty", f"{risk_metrics.get('Beta', 1.0):.2f}")
        with sub_col2:
            st.metric("Sortino Ratio", f"{risk_metrics.get('Sortino_Ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{risk_metrics.get('Max_Drawdown', 0)*100:.1f}%")
            
    # Global Anomaly Feed
    st.markdown("---")
    st.subheader("🚨 Global Market Anomaly Monitoring Feed")
    
    if global_anomalies.empty:
        st.info("No anomalies detected in the historical timeframe.")
    else:
        # Filtering dashboard
        anomaly_filter = st.selectbox("Filter Anomaly Severity", ["All Severities", "High Severity Only"])
        
        display_anoms = global_anomalies
        if anomaly_filter == "High Severity Only":
            display_anoms = display_anoms[display_anoms["Severity"] == "High"]
            
        if display_anoms.empty:
            st.success("No anomalies matching the selected filter.")
        else:
            # Custom styled dataframe view
            st.dataframe(
                display_anoms.head(30),
                column_config={
                    "Date": st.column_config.TextColumn("Date/Time", width="medium"),
                    "Symbol": st.column_config.TextColumn("Stock Symbol", width="small"),
                    "Type": st.column_config.TextColumn("Anomaly Category", width="medium"),
                    "Metric": st.column_config.TextColumn("Incident Metrics", width="medium"),
                    "Details": st.column_config.TextColumn("Description & Severity Breakdown", width="large"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                },
                use_container_width=True,
                hide_index=True
            )

# TAB 2: STOCK PREDICTOR & EXPLAINABLE AI 
with tab2:
    try:
        pred_res, reg_metrics, clf_metrics, importance_df = get_predictions_and_metrics(selected_symbol)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Machine Learning Predictive Output")
            
            # Prediction Results Card
            dir_color = "#10B981" if pred_res["Predicted_Direction"] == "UP" else "#EF4444"
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:20px;">
                <div class="metric-label">5-Day Expected Return Trend</div>
                <div class="metric-value" style="color:{dir_color};">{pred_res['Predicted_Direction']}</div>
                <div style="font-size:1rem; color:#F8FAFC; margin-top:5px;">
                    Predicted 5-Day Return: <b>{pred_res['Predicted_5d_Return']*100:+.2f}%</b>
                </div>
                <div style="font-size:0.9rem; color:#94A3B8;">
                    Prediction Confidence: <b>{pred_res['Direction_Probability']*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Model performance details
            st.subheader("Model Evaluation Metrics (Test Split)")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Regression R² Score", f"{reg_metrics.get('R2', 0):.4f}")
                st.metric("Regression MAE", f"{reg_metrics.get('MAE', 0)*100:.3f}%")
            with p_col2:
                st.metric("Classification Accuracy", f"{clf_metrics.get('Accuracy', 0)*100:.2f}%")
                st.metric("Classification F1-Score", f"{clf_metrics.get('F1_Score', 0):.3f}")
                
        with col2:
            st.subheader("🤖 Explainable AI: Feature Attribution (SHAP-Proxy)")
            
            # Plot permutation feature importance
            fig_imp = px.bar(
                importance_df.head(10),
                x="Importance_Mean",
                y="Feature",
                orientation="h",
                labels={"Importance_Mean": "Permutation Importance (Loss Decrease)", "Feature": "Technical Factor"},
                title=f"Top 10 Feature Drivers of Predictions for {selected_symbol}",
                color="Importance_Mean",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig_imp.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        # Explanations block
        st.markdown("---")
        st.subheader("💡 Technical Indicator Signals & Explanations")
        
        latest_ind_row = compute_technical_indicators(df_selected).tail(1)
        reasoning = explainer.generate_financial_reasoning(latest_ind_row)
        
        sent_color = "#10B981" if reasoning["Sentiment"] == "BULLISH" else "#EF4444" if reasoning["Sentiment"] == "BEARISH" else "#94A3B8"
        st.markdown(f"**Overall Technical Sentiment Score:** <span style='color:{sent_color}; font-weight:bold; font-size:1.2rem;'>{reasoning['Sentiment']}</span>", unsafe_allow_html=True)
        
        for bp in reasoning["Bullet_Points"]:
            st.markdown(f"- {bp}")
            
    except Exception as e:
        st.error(f"Error executing predictive models for {selected_symbol}: {e}")

# TAB 3: PORTFOLIO CONSTRUCTION 
with tab3:
    st.subheader(f"Optimal Asset Allocation: {investor_profile} Profile")
    portfolio, cov_matrix = get_portfolio_allocations(investor_profile, risk_free_input)
    allocs_df = portfolio["Allocations"]
    st.markdown("### 🏆 Top Portfolio Asset Allocations")
    top_allocs = allocs_df.head(4)
    cols = st.columns(len(top_allocs))
    for col, (_, row) in zip(cols, top_allocs.iterrows()):
        col.metric(
            label=row["Symbol"], 
            value=f"{row['Weight']*100:.1f}%", 
            help=f"{row['Company Name']} - Sector: {row['Sector']}"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_pie = px.pie(
            allocs_df,
            values="Weight",
            names="Symbol",
            hover_data=["Company Name", "Sector"],
            title=f"Allocation Weightings Breakdown",
            color_discrete_sequence=px.colors.sequential.Tealgrn,
            hole=0.4,
            template="plotly_dark"
        )
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # Portfolio statistics block 
        sharpe_ratio = portfolio['Sharpe_Ratio']
        
        st.markdown(f"""
        <div class="metric-card" style="margin-top:40px; margin-bottom:10px;">
            <div class="metric-label">Annualized Expected Return</div>
            <div class="metric-value">{portfolio['Expected_Annualized_Return']*100:.2f}%</div>
            <div style="font-size:0.9rem; color:#94A3B8; margin-top:10px;">
                Annualized Volatility: <b>{portfolio['Expected_Annualized_Volatility']*100:.2f}%</b>
            </div>
            <div style="font-size:0.9rem; color:#94A3B8;">
                Portfolio Sharpe Ratio: <b>{sharpe_ratio:.2f}</b> (Risk-Free: {risk_free_input*100:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Legibly handle and explain negative Sharpe Ratios
        if sharpe_ratio < 0:
            st.caption(
                "⚠️ **Note:** The negative Sharpe ratio indicates the portfolio's expected return "
                "is lower than the risk-free rate. This is typical for a heavily constrained, "
                "defensive Conservative profile in high-interest environments, where the focus "
                "is strictly on minimizing portfolio variance rather than outperforming cash yields."
            )
            
        st.markdown(f"**Optimization Justification:**")
        st.write(portfolio["Justification"])
        
    st.markdown("---")
    st.subheader("Asset Weight Breakdowns & Covariance Analysis")
    
    alloc_col, cov_col = st.columns([1, 1.2])
    
    with alloc_col:
        st.markdown("**Allocation Weights Table**")
        st.dataframe(
            allocs_df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True
        )
        
    with cov_col:
        # Covariance Heatmap 
        st.markdown("**Historical Stock Returns Covariance Matrix Heatmap**")
        
        fig_heat = px.imshow(
            cov_matrix,
            labels=dict(x="Stock Symbol", y="Stock Symbol", color="Covariance"),
            x=cov_matrix.columns,
            y=cov_matrix.index,
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_heat.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

#TAB 4: PERSONALIZED ADVISOR 
with tab4:
    st.subheader("🎯 Personalized Investment Strategy Recommendation Engine")
    st.write("Complete the profiling questions below to generate a tailored allocation and investment strategy.")
    
    # Input questionnaire
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        capital = st.number_input("Total Investment Capital (INR)", min_value=10000, value=100000, step=10000)
        age = st.slider("Investor Age (Years)", min_value=18, max_value=90, value=35)
        horizon = st.selectbox("Investment Time Horizon", ["Short-term (< 1 Year)", "Medium-term (1-3 Years)", "Long-term (> 3 Years)"])
        
    with col_input2:
        objective = st.selectbox(
            "Primary Investment Objective",
            ["Capital Preservation & Dividend Income", "Balanced Growth & Dividend Income", "Maximum Capital Appreciation"]
        )
        loss_tolerance = st.selectbox(
            "Reaction to a 15% Market Drop",
            [
                "Sell immediately to protect capital (Low Risk Tolerance)",
                "Hold and wait for recovery (Medium Risk Tolerance)",
                "Buy more assets at discount prices (High Risk Tolerance)"
            ]
        )
        
    # Strategic evaluation logic
    risk_score = 0
    if age < 30: risk_score += 3
    elif age < 50: risk_score += 2
    else: risk_score += 1
    
    if horizon == "Short-term (< 1 Year)": risk_score += 0
    elif horizon == "Medium-term (1-3 Years)": risk_score += 2
    else: risk_score += 4
    
    if objective == "Capital Preservation & Dividend Income": risk_score += 0
    elif objective == "Balanced Growth & Dividend Income": risk_score += 2
    else: risk_score += 4
    
    if "Low Risk Tolerance" in loss_tolerance: risk_score += 0
    elif "Medium Risk Tolerance" in loss_tolerance: risk_score += 2
    else: risk_score += 4
    
    # Map score to profile recommendation
    if risk_score <= 4:
        recommended_profile = "Conservative"
        target_allocation = "70% Fixed-Income / Defensive stocks, 30% Blue-Chip growth stocks"
    elif risk_score <= 9:
        recommended_profile = "Balanced"
        target_allocation = "50% Blue-Chip Growth stocks, 30% Defensive stocks, 20% Sectoral Leaders"
    else:
        recommended_profile = "Aggressive"
        target_allocation = "80% High-Beta Sectoral Leaders, 20% Growth Stocks"
        
    if st.button("Generate Strategy Recommendation", type="primary"):
        st.markdown("---")
        st.markdown(f"### Recommended Strategic Profile: **{recommended_profile} Portfolio**")
        st.markdown(f"**Target Strategic Mix:** {target_allocation}")
        
        # Pull allocation details from Tab 3 module 
        rec_portfolio, _ = get_portfolio_allocations(recommended_profile, risk_free_input)
        
        advisor_col1, advisor_col2 = st.columns([1, 1.2])
        
        with advisor_col1:
            st.markdown(f"**Expected Annualized Return:** {rec_portfolio['Expected_Annualized_Return']*100:.2f}%")
            st.markdown(f"**Portfolio Risk (Volatility):** {rec_portfolio['Expected_Annualized_Volatility']*100:.2f}%")
            st.markdown(f"**Portfolio Sharpe Ratio:** {rec_portfolio['Sharpe_Ratio']:.2f}")
            st.write(rec_portfolio["Justification"])
            
        with advisor_col2:
            st.markdown(f"**Allocation of Capital: ₹{capital:,}**")
            allocs = rec_portfolio["Allocations"].copy()
            allocs["Capital Allocation (INR)"] = allocs["Weight"] * capital
            
            # Format display
            st.dataframe(
                allocs.style.format({
                    "Weight": "{:.2%}",
                    "Capital Allocation (INR)": "₹{:,.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Action Plan
            st.info("💡 **Advisor Action Plan:** Build your NIFTY-50 portfolio by deploying capital according to the weights shown. Monitor risk monthly. If individual stock technicals show consistent BEARISH sentiments in the 'Stock Predictor' tab, consider re-optimizing allocations.")
