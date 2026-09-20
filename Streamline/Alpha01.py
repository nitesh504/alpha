import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hurst import compute_Hc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from pypfopt import expected_returns, risk_models, EfficientFrontier
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="UPRETY CAPITAL QUANTSTACK ANALYZER",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define password
CORRECT_PASSWORD = "utycapital"

# Initialize session state
if 'password_correct' not in st.session_state:
    st.session_state.password_correct = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Enhanced CSS matching the modern UI design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0a0a0a;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        background-color: #0a0a0a;
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background-color: #0a0a0a;
        border: 2px solid #ffffff;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    
    /* Section headers */
    .section-header {
        background-color: #0a0a0a;
        border: 1px solid #ffffff;
        color: #ffffff;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1rem 0 0.8rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Compact metrics */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    
    .metric-box {
        background-color: #0a0a0a;
        border: 1px solid #ffffff;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 300;
        color: #ffffff;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #cccccc;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Compact tables */
    .dataframe {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
    }
    
    .dataframe th {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
    }
    
    .dataframe td {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        font-size: 0.7rem !important;
        padding: 0.4rem !important;
    }
    
    /* Compact summary boxes */
    .summary-box {
        background-color: #0a0a0a;
        border: 1px solid #ffffff;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    
    .summary-title {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input styling */
    .input-container {
        background-color: #0a0a0a;
        border: 1px solid #ffffff;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 0px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
    }
    
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stNumberInput > label,
    .stSlider > label,
    .stCheckbox > label {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 2px solid #ffffff !important;
        border-radius: 0px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #ffffff !important;
        color: #0a0a0a !important;
    }
    
    /* Color coding for EMA, TREND, and REGIME */
    .ema-above { 
        color: #90EE90 !important; 
        background-color: rgba(144, 238, 144, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .ema-below { 
        color: #FFB6C1 !important; 
        background-color: rgba(255, 182, 193, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .trend-trending { 
        color: #90EE90 !important; 
        background-color: rgba(144, 238, 144, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .trend-other { 
        color: #FFB6C1 !important; 
        background-color: rgba(255, 182, 193, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .regime-bullish { 
        color: #90EE90 !important; 
        background-color: rgba(144, 238, 144, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .regime-bearish { 
        color: #FFB6C1 !important; 
        background-color: rgba(255, 182, 193, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    /* Status grid */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }
    
    .status-item {
        background-color: #0a0a0a;
        border: 1px solid #ffffff;
        padding: 1rem;
        text-align: center;
    }
    
    .status-fraction {
        font-size: 1.4rem;
        font-weight: 300;
        color: #ffffff;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }
    
    .status-text {
        font-size: 0.65rem;
        color: #cccccc;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Compact footer */
    .footer {
        border-top: 1px solid #ffffff;
        padding: 1rem;
        text-align: center;
        color: #cccccc;
        font-size: 0.8rem;
        font-family: 'Inter', sans-serif;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# StockRiskAnalyzer class (unchanged)
class StockRiskAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()
        self.calculate_log_returns()
    
    def fetch_data(self, period="1y"):
        data = yf.download(self.ticker, period=period)
        return data
    
    def calculate_log_returns(self):
        self.data['LogReturn'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data = self.data.dropna()

    def calculate_var_stop_loss(self, holding_days: int, position_type: str, confidence_level: float = 0.95):
        if self.data is None or len(self.data) < 2:
            raise ValueError("Not enough historical data to calculate returns.")

        daily_volatility = float(self.data['LogReturn'].std())
        z_score = norm.ppf(confidence_level)
        var_percent = z_score * daily_volatility * np.sqrt(holding_days)
        entry_price = float(self.data['Close'].iloc[-1])
        
        if position_type.lower() == 'long':
            stop_loss_price = entry_price * (1 - var_percent)
        elif position_type.lower() == 'short':
            stop_loss_price = entry_price * (1 + var_percent)
        else:
            raise ValueError("Position type must be 'long' or 'short'.")
        
        return {
            "Stock": self.ticker,
            "Position": position_type,
            "Holding Days": holding_days,
            "Confidence Level": confidence_level,
            "VaR %": round(var_percent * 100, 2),
            "Entry Price": round(entry_price, 2),
            "Stop Loss Price": round(stop_loss_price, 2)
        }

    def calculate_target_price(self, stop_loss_price: float, position_type: str, reward_ratio: float = 2.0):
        entry_price = float(self.data['Close'].iloc[-1])

        if position_type.lower() == 'long':
            risk = entry_price - stop_loss_price
            target_price = entry_price + reward_ratio * risk
        elif position_type.lower() == 'short':
            risk = stop_loss_price - entry_price
            target_price = entry_price - reward_ratio * risk
        else:
            raise ValueError("Position type must be 'long' or 'short'.")
        
        return {
            "Entry Price": round(entry_price, 2),
            "Stop Loss Price": round(stop_loss_price, 2),
            "Risk per Unit": round(risk, 2),
            "Reward Ratio": reward_ratio,
            "Target Price": round(target_price, 2)
        }

# All analysis functions (unchanged but with compact display)
@st.cache_data(ttl=3600*24)
def get_all_tickers():
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        tickers_df = pd.read_csv(url, header=None)
        tickers = sorted(tickers_df[0].tolist())
        return tickers
    except:
        return sorted([
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "INTC", "VZ", "DIS", "ADBE",
            "CSCO", "CRM", "NFLX", "PFE", "KO", "PEP", "T", "MRK", "CMCSA"
        ])

def verify_ticker(ticker):
    try:
        data = yf.download(ticker, period="1d", progress=False)
        return not data.empty
    except:
        return False

def hurst_exponent(prices):
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    try:
        H, c, data_reg = compute_Hc(prices, kind='price', simplified=True)
        return H
    except Exception as e:
        return 0.5

def interpret_hurst(hurst_value):
    if hurst_value < 0.5:
        return "Mean Reversion"
    elif hurst_value > 0.5:
        return "Trending"
    else:
        return "Random Walk"

def ema_status(data):
    current_price = float(data['Close'].iloc[-1])
    ema_50 = float(data['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
    return "Above" if current_price > ema_50 else "Below", ema_50

def volatility_regime(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.rolling(window=20).std().dropna()
    vol_threshold = volatility.median()
    
    current_vol = float(volatility.iloc[-1])
    vol_threshold_value = float(vol_threshold)
    
    regime = "Low Vol" if current_vol < vol_threshold_value else "High Vol"
    return regime, current_vol, vol_threshold_value

def hmm_regime(data):
    try:
        df = pd.DataFrame()
        df['Close'] = data['Close']
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(10).std()
        df['Volume'] = data['Volume']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if len(df) < 30:
            return "Insufficient Data", None, None
        
        scaler = StandardScaler()
        
        try:
            X = scaler.fit_transform(df[['Return', 'Volatility', 'Volume_Change']])
        except Exception as e:
            for col in ['Return', 'Volatility', 'Volume_Change']:
                mean = df[col].mean()
                std = df[col].std()
                if not np.isnan(std) and std != 0:
                    lower_bound = mean - 5 * std
                    upper_bound = mean + 5 * std
                    df[col] = df[col].clip(lower_bound, upper_bound)
            
            X = scaler.fit_transform(df[['Return', 'Volatility', 'Volume_Change']])
        
        best_score = -np.inf
        best_model = None
        best_states = None
        
        for seed in range(5):
            model = GaussianHMM(n_components=2, covariance_type="tied", n_iter=1000, random_state=seed)
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_states = model.predict(X)
            except:
                continue
        
        if best_model is None:
            return "Model Failed", None, None
        
        df['Regime'] = best_states
        regime_returns = df.groupby('Regime')['Return'].mean()
        
        if len(regime_returns) < 2:
            return "Insufficient Regimes", None, None
        
        bull_state = regime_returns.idxmax()
        bear_state = regime_returns.idxmin()
        
        if len(df) == 0 or 'Regime' not in df.columns or df['Regime'].empty:
            return "Insufficient Data", None, None
            
        last_state = df['Regime'].iloc[-1]
        
        regime = "Bullish" if last_state == bull_state else "Bearish"
        avg_return = regime_returns[last_state]
        
        return regime, avg_return, df['Regime']
    
    except Exception as e:
        return f"Error", None, None

def analyze_ticker(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        
        if len(data) < 50:
            return {"ticker": ticker, "error": "Insufficient data"}
        
        ema_position, ema_value = ema_status(data)
        hurst_value = hurst_exponent(data['Close'])
        hurst_interpretation = interpret_hurst(hurst_value)
        vol_regime, current_vol, vol_threshold = volatility_regime(data)
        bull_bear_regime, avg_return, regime_series = hmm_regime(data)
        current_price = float(data['Close'].iloc[-1])
        
        try:
            ticker_info = yf.Ticker(ticker).info
            company_name = ticker_info.get('shortName', ticker)
        except:
            company_name = ticker
        
        result = {
            "ticker": ticker,
            "name": company_name,
            "current_price": current_price,
            "50_ema": round(ema_value, 2),
            "50_ema_status": ema_position,
            "hurst_exponent": round(hurst_value, 3),
            "hurst_interpretation": hurst_interpretation,
            "volatility_regime": vol_regime,
            "current_volatility": round(current_vol, 4),
            "vol_threshold": round(vol_threshold, 4),
            "bull_bear_regime": bull_bear_regime,
            "avg_regime_return": None if avg_return is None else round(avg_return, 4),
            "regime_series": regime_series,
            "data": data
        }
        
        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def analyze_correlations(tickers, period="1y"):
    try:
        end_date = datetime.now()
        
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)
            
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        returns = data.pct_change().dropna()
        correlation_matrix = returns.corr()
        
        threshold = 0.3
        G = nx.Graph()
        G.add_nodes_from(tickers)
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                stock1 = tickers[i]
                stock2 = tickers[j]
                if stock1 in correlation_matrix.index and stock2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[stock1, stock2]
                    if abs(corr) <= threshold:
                        G.add_edge(stock1, stock2)
        
        cliques = list(nx.find_cliques(G))
        largest_clique = max(cliques, key=len) if cliques else []
        
        return {
            "correlation_matrix": correlation_matrix,
            "diversified_subset": largest_clique,
            "price_data": data
        }
    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")
        return None

def optimize_portfolio(price_data, allow_shorting=False):
    try:
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
        
        weight_bounds = (-1, 1) if allow_shorting else (0, 1)
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        
        return {
            "weights": cleaned_weights,
            "expected_annual_return": expected_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        st.error(f"Portfolio optimization error: {str(e)}")
        return None

def perform_sltp_analysis(tickers, holding_days=10, position_type='long', reward_ratio=2.0):
    sltp_results = []
    
    for ticker in tickers:
        try:
            analyzer = StockRiskAnalyzer(ticker)
            var_result = analyzer.calculate_var_stop_loss(
                holding_days=holding_days, 
                position_type=position_type
            )
            
            target_result = analyzer.calculate_target_price(
                stop_loss_price=var_result['Stop Loss Price'], 
                position_type=position_type,
                reward_ratio=reward_ratio
            )
            
            combined_result = {**var_result}
            
            for key, value in target_result.items():
                if key not in combined_result:
                    combined_result[key] = value
            
            sltp_results.append(combined_result)
        
        except Exception as e:
            sltp_results.append({
                "Stock": ticker,
                "Error": str(e)
            })
    
    return sltp_results

# Function to create HTML table with proper color styling
def create_styled_table(df):
    def get_cell_style(val, col_name):
        if col_name == 'EMA':
            if val == 'Above':
                return 'color: #90EE90; background-color: rgba(144, 238, 144, 0.1); padding: 4px 8px; border-radius: 3px;'
            elif val == 'Below':
                return 'color: #FFB6C1; background-color: rgba(255, 182, 193, 0.1); padding: 4px 8px; border-radius: 3px;'
        elif col_name == 'TREND':
            if val == 'Trending':
                return 'color: #90EE90; background-color: rgba(144, 238, 144, 0.1); padding: 4px 8px; border-radius: 3px;'
            else:  # Mean Reversion, Random Walk, or any other value
                return 'color: #FFB6C1; background-color: rgba(255, 182, 193, 0.1); padding: 4px 8px; border-radius: 3px;'
        elif col_name == 'REGIME':
            if val == 'Bullish':
                return 'color: #90EE90; background-color: rgba(144, 238, 144, 0.1); padding: 4px 8px; border-radius: 3px;'
            elif val == 'Bearish':
                return 'color: #FFB6C1; background-color: rgba(255, 182, 193, 0.1); padding: 4px 8px; border-radius: 3px;'
        return 'color: #ffffff; padding: 4px 8px;'
    
    # Create HTML table
    html = '<table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.75rem;">'
    
    # Header
    html += '<thead><tr style="background-color: #0a0a0a; border: 1px solid #ffffff;">'
    for col in df.columns:
        html += f'<th style="color: #ffffff; border: 1px solid #ffffff; padding: 8px; text-align: left; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.7rem;">{col}</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr style="background-color: #0a0a0a; border: 1px solid #333333;">'
        for col in df.columns:
            val = row[col]
            cell_style = get_cell_style(val, col)
            html += f'<td style="border: 1px solid #333333; {cell_style}">{val}</td>'
        html += '</tr>'
    html += '</tbody>'
    
    html += '</table>'
    return html

# Main application
st.markdown('<div class="main-header"><h1>UPRETY CAPITAL QUANTSTACK ANALYZER</h1></div>', unsafe_allow_html=True)

# Password protection
if not st.session_state.password_correct:
    st.markdown('<div class="section-header">ACCESS CONTROL</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password_input = st.text_input("ENTER PASSWORD", type="password", key="password")
        if st.button("ACCESS TOOL"):
            if password_input == CORRECT_PASSWORD:
                st.session_state.password_correct = True
                st.success("ACCESS GRANTED")
                st.rerun()
            else:
                st.error("INCORRECT PASSWORD")
    st.stop()

# Create two-column layout after password is entered
left_column, right_column = st.columns([1, 2])

# Left Column - Analysis Settings
with left_column:
    st.markdown('<div class="section-header">ANALYSIS SETTINGS</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Ticker input
    ticker_input = st.text_area(
        "STOCK TICKERS (COMMA SEPARATED)",
        value="AAPL, MSFT, AMZN",
        height=80
    )

    # Time period
    period = st.selectbox(
        "TIME PERIOD",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    # Analyze button
    analyze_button = st.button("ANALYZE STOCKS")

    st.markdown('</div>', unsafe_allow_html=True)

    # Advanced settings
    st.markdown("**ADVANCED SETTINGS**")
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    correlation_threshold = st.slider(
        "CORRELATION THRESHOLD",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05
    )

    allow_shorting = st.checkbox("ENABLE SHORT SELLING", value=False)

    holding_days = st.number_input(
        "HOLDING DAYS", 
        min_value=1, 
        max_value=365, 
        value=10
    )

    position_type = st.selectbox("POSITION TYPE", ["Long", "Short"])

    reward_ratio = st.number_input(
        "REWARD RATIO", 
        min_value=1.0, 
        max_value=10.0, 
        value=2.0,
        step=0.5
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Results
with right_column:
    st.markdown('<div class="section-header">ANALYSIS RESULTS</div>', unsafe_allow_html=True)

    # Initialize input_tickers as empty list
    input_tickers = []
    
    # Process analysis
    if ticker_input and analyze_button:
        input_tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    if input_tickers:
        # Validate tickers
        with st.spinner("VALIDATING TICKERS..."):
            valid_tickers = []
            invalid_tickers = []
            
            for ticker in input_tickers:
                if verify_ticker(ticker):
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
        
        if invalid_tickers:
            st.warning(f"Invalid tickers: {', '.join(invalid_tickers)}")
        
        if valid_tickers:
            # Compact status display
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-value">{len(valid_tickers)}</div>
                        <div class="metric-label">STOCKS</div>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-value">{period}</div>
                        <div class="metric-label">PERIOD</div>
                    </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Perform analysis
            with st.spinner("ANALYZING..."):
                
                # 1. INDIVIDUAL STOCK ANALYSIS (MAIN FOCUS)
                st.markdown('<div class="section-header">STOCK ANALYSIS</div>', unsafe_allow_html=True)
                
                results = []
                for ticker in valid_tickers:
                    result = analyze_ticker(ticker, period)
                    results.append(result)
                
                valid_results = [r for r in results if "error" not in r]
                
                if valid_results:
                    # Create compact display DataFrame with updated EMA, TREND, and REGIME columns
                    display_data = []
                    for r in valid_results:
                        display_data.append({
                            "TICKER": r["ticker"],
                            "PRICE": f"${r['current_price']:.2f}",
                            "EMA": r["50_ema_status"],  # Shows "Above" or "Below"
                            "HURST": f"{r['hurst_exponent']:.3f}",
                            "TREND": r["hurst_interpretation"],  # Shows full trend name
                            "VOL": r["volatility_regime"][:8],
                            "REGIME": r["bull_bear_regime"][:8]  # Shows "Bullish" or "Bearish"
                        })
                    
                    df = pd.DataFrame(display_data)
                    
                    # Create and display styled HTML table
                    styled_table_html = create_styled_table(df)
                    st.markdown(styled_table_html, unsafe_allow_html=True)
                    
                    # Compact summary metrics
                    st.markdown('<div class="status-grid">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        above_ema = sum(1 for r in valid_results if r["50_ema_status"] == "Above")
                        st.markdown(f'''
                            <div class="status-item">
                                <div class="status-fraction">{above_ema}/{len(valid_results)}</div>
                                <div class="status-text">EMA</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        bullish_count = sum(1 for r in valid_results if r["bull_bear_regime"] == "Bullish")
                        st.markdown(f'''
                            <div class="status-item">
                                <div class="status-fraction">{bullish_count}/{len(valid_results)}</div>
                                <div class="status-text">BULL</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        trending_count = sum(1 for r in valid_results if r["hurst_interpretation"] == "Trending")
                        st.markdown(f'''
                            <div class="status-item">
                                <div class="status-fraction">{trending_count}/{len(valid_results)}</div>
                                <div class="status-text">TREND</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col4:
                        high_vol_count = sum(1 for r in valid_results if r["volatility_regime"] == "High Vol")
                        st.markdown(f'''
                            <div class="status-item">
                                <div class="status-fraction">{high_vol_count}/{len(valid_results)}</div>
                                <div class="status-text">HIGH VOL</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 2. STOP LOSS - TARGET PRICE ANALYSIS (COMPACT)
                st.markdown('<div class="section-header">RISK MANAGEMENT</div>', unsafe_allow_html=True)
                
                sltp_results = perform_sltp_analysis(
                    valid_tickers, 
                    holding_days=holding_days, 
                    position_type=position_type.lower(), 
                    reward_ratio=reward_ratio
                )
                
                if sltp_results:
                    successful_results = [r for r in sltp_results if 'Error' not in r]
                    
                    if successful_results:
                        # Compact SLTP display
                        compact_sltp = []
                        for r in successful_results:
                            compact_sltp.append({
                                "STOCK": r["Stock"],
                                "ENTRY": f"${r['Entry Price']:.2f}",
                                "STOP": f"${r['Stop Loss Price']:.2f}",
                                "TARGET": f"${r['Target Price']:.2f}",
                                "VAR%": f"{r['VaR %']:.1f}%"
                            })
                        
                        df_sltp = pd.DataFrame(compact_sltp)
                        st.dataframe(df_sltp, use_container_width=True, hide_index=True)
                        
                        # Key metrics
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_var = sum(r['VaR %'] for r in successful_results) / len(successful_results)
                            st.markdown(f'''
                                <div class="metric-box">
                                    <div class="metric-value">{avg_var:.1f}%</div>
                                    <div class="metric-label">AVG VAR</div>
                                </div>
                            ''', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'''
                                <div class="metric-box">
                                    <div class="metric-value">{reward_ratio}:1</div>
                                    <div class="metric-label">R:R</div>
                                </div>
                            ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # 3. PORTFOLIO OPTIMIZATION (COMPACT - ONLY IF MULTIPLE STOCKS)
                if len(valid_tickers) > 1:
                    st.markdown('<div class="section-header">PORTFOLIO</div>', unsafe_allow_html=True)
                    
                    correlation_results = analyze_correlations(valid_tickers, period)
                    
                    if correlation_results:
                        price_data = correlation_results["price_data"].dropna(axis=1)
                        
                        if price_data.shape[1] > 1:
                            optimization_results = optimize_portfolio(price_data, allow_shorting)
                            
                            if optimization_results:
                                # Compact portfolio metrics
                                st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f'''
                                        <div class="metric-box">
                                            <div class="metric-value">{optimization_results["expected_annual_return"]:.1%}</div>
                                            <div class="metric-label">RETURN</div>
                                        </div>
                                    ''', unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f'''
                                        <div class="metric-box">
                                            <div class="metric-value">{optimization_results["annual_volatility"]:.1%}</div>
                                            <div class="metric-label">VOLATILITY</div>
                                        </div>
                                    ''', unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f'''
                                        <div class="metric-box">
                                            <div class="metric-value">{optimization_results["sharpe_ratio"]:.2f}</div>
                                            <div class="metric-label">SHARPE</div>
                                        </div>
                                    ''', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show only top 3 weights
                                weights = optimization_results['weights']
                                top_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3])
                                
                                compact_weights = []
                                for stock, weight in top_weights.items():
                                    compact_weights.append({
                                        "STOCK": stock,
                                        "WEIGHT": f"{weight:.1%}"
                                    })
                                
                                if compact_weights:
                                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                                    st.markdown('<div class="summary-title">TOP 3 WEIGHTS</div>', unsafe_allow_html=True)
                                    st.dataframe(pd.DataFrame(compact_weights), use_container_width=True, hide_index=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enter stock tickers and click 'ANALYZE STOCKS' to begin analysis.")

# Compact footer
st.markdown('<div class="footer">**UTY CAPITAL** - EQUITY ANALYSIS TOOL</div>', unsafe_allow_html=True)
