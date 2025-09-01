import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("📈 Stock Market Analyzer")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Create a function to generate sample data since we can't access the actual files
def generate_sample_data(symbol, days=365*2):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = np.random.randint(50, 200)
    prices = base_price + np.cumsum(np.random.normal(0, 1, days))
    volume = np.random.randint(100000, 1000000, days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.01, days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'Close': prices,
        'Volume': volume
    })
    df.set_index('Date', inplace=True)
    return df

# Function to load stock data (using sample data for demonstration)
@st.cache_data
def load_stock_data(symbol):
    try:
        # For demonstration, we'll generate sample data
        # In a real scenario, you would load from your CSV files
        df = generate_sample_data(symbol)
        return df
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

# Get list of available stocks (using a sample list for demonstration)
@st.cache_data
def get_stock_list():
    # Sample stock symbols - in a real app, you would get these from your directory
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# Load stock symbols
stock_symbols = get_stock_list()

# User inputs
selected_stocks = st.sidebar.multiselect(
    'Select stocks to analyze',
    stock_symbols,
    default=['AAPL', 'GOOGL', 'MSFT']
)

# Date range selector
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # Default to 2 years

col1, col2 = st.sidebar.columns(2)
with col1:
    selected_start = st.date_input("Start date", start_date)
with col2:
    selected_end = st.date_input("End date", end_date)

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Price Trends", "Returns Analysis", "Volatility Analysis", "Correlation Matrix"]
)

# Load and process data
if selected_stocks:
    data = {}
    for symbol in selected_stocks:
        df = load_stock_data(symbol)
        if df is not None:
            # Filter by date range
            mask = (df.index >= pd.to_datetime(selected_start)) & (df.index <= pd.to_datetime(selected_end))
            data[symbol] = df.loc[mask]
    
    if not data:
        st.error("No data available for the selected stocks and date range.")
    else:
        # Display stock data
        st.subheader("Stock Price Data")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Data Table", "Analysis"])
        
        with tab1:
            # Interactive price chart
            chart_type = st.radio("Chart Type", ["Line", "Candle"], horizontal=True)
            
            if chart_type == "Line":
                fig = go.Figure()
                for symbol, df in data.items():
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Close'], 
                        name=symbol,
                        mode='lines'
                    ))
                fig.update_layout(
                    title="Stock Price Trends",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Show candle chart for first selected stock
                if selected_stocks:
                    symbol = selected_stocks[0]
                    df = data[symbol]
                    fig = go.Figure(data=[go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=symbol
                    )])
                    fig.update_layout(
                        title=f"{symbol} Candlestick Chart",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Display data table
            selected_symbol = st.selectbox("Select stock to view data", list(data.keys()))
            st.dataframe(data[selected_symbol].style.format({
                'Open': '{:.2f}',
                'High': '{:.2f}',
                'Low': '{:.2f}',
                'Close': '{:.2f}',
                'Volume': '{:,}'
            }), use_container_width=True)
        
        with tab3:
            # Perform selected analysis
            if analysis_type == "Price Trends":
                st.subheader("Normalized Price Comparison")
                fig = go.Figure()
                for symbol, df in data.items():
                    normalized = df['Close'] / df['Close'].iloc[0] * 100
                    fig.add_trace(go.Scatter(x=df.index, y=normalized, name=symbol))
                
                fig.update_layout(
                    title="Normalized Price Comparison (Base=100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Returns Analysis":
                st.subheader("Daily Returns Analysis")
                
                returns_fig = go.Figure()
                for symbol, df in data.items():
                    returns = df['Close'].pct_change().dropna()
                    returns_fig.add_trace(go.Scatter(x=returns.index, y=returns, name=symbol, mode='lines'))
                
                returns_fig.update_layout(
                    title="Daily Returns",
                    xaxis_title="Date",
                    yaxis_title="Daily Return",
                    hovermode="x unified"
                )
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Histogram of returns
                st.subheader("Returns Distribution")
                dist_fig = go.Figure()
                for symbol, df in data.items():
                    returns = df['Close'].pct_change().dropna()
                    dist_fig.add_trace(go.Histogram(x=returns, name=symbol, opacity=0.7))
                
                dist_fig.update_layout(
                    title="Distribution of Daily Returns",
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                dist_fig.update_traces(opacity=0.75)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            elif analysis_type == "Volatility Analysis":
                st.subheader("Volatility Analysis (Rolling Standard Deviation)")
                
                vol_window = st.slider("Rolling Window (days)", 5, 100, 30)
                
                vol_fig = go.Figure()
                for symbol, df in data.items():
                    returns = df['Close'].pct_change()
                    volatility = returns.rolling(window=vol_window).std() * np.sqrt(252)  # Annualized
                    vol_fig.add_trace(go.Scatter(x=volatility.index, y=volatility, name=symbol))
                
                vol_fig.update_layout(
                    title=f"Rolling Volatility ({vol_window}-day window)",
                    xaxis_title="Date",
                    yaxis_title="Annualized Volatility",
                    hovermode="x unified"
                )
                st.plotly_chart(vol_fig, use_container_width=True)
            
            elif analysis_type == "Correlation Matrix":
                st.subheader("Correlation Matrix of Returns")
                
                # Create a DataFrame of closing prices
                closes_df = pd.DataFrame()
                for symbol, df in data.items():
                    closes_df[symbol] = df['Close']
                
                # Calculate correlation of returns - FIXED
                returns_df = closes_df.pct_change().dropna()
                
                # Check if we have enough data for correlation
                if len(returns_df) < 2:
                    st.warning("Not enough data points to calculate correlation. Please select a longer time period.")
                else:
                    correlation_matrix = returns_df.corr()
                    
                    # Display the correlation matrix as a heatmap
                    fig = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix of Daily Returns",
                        zmin=-1,  # Set the scale from -1 to 1
                        zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also display the correlation values as a table
                    st.write("Correlation Values:")
                    st.dataframe(correlation_matrix.style.format("{:.3f}").background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))
else:
    st.info("Please select at least one stock from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown("### 📊 Stock Market Analyzer | Built with Streamlit")