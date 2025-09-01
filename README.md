# 📈 Time Series Analysis and Forecasting for Stock Market  

This repository contains code and report for **stock market time series forecasting** using multiple approaches:  
- **ARIMA (AutoRegressive Integrated Moving Average)**  
- **SARIMA (Seasonal ARIMA)**  
- **LSTM (Long Short-Term Memory Neural Networks)**  
- **Prophet (by Facebook/Meta)**  

The goal of this project is to compare traditional statistical models and modern deep learning methods for predicting **stock/ETF closing prices** and analyze their strengths and limitations.  

---

## 📂 Dataset  

We used the [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset), which contains:  
- **8050 CSV files** (size: ~2.75 GB)  
- Each file corresponds to a stock/ETF ticker symbol  
- Columns include:  
  - `Date`: Trading date  
  - `Open`: Opening price  
  - `High`: Maximum daily price  
  - `Low`: Minimum daily price  
  - `Close`: Closing price  
  - `Adj Close`: Adjusted closing price  
  - `Volume`: Number of shares traded  

For forecasting, we used the **`Close` price** because it is the most financially relevant variable for investors.  

---

## ⚙️ Methodology  

### 1. Data Preprocessing  
- Converted `Date` to datetime format and set as index  
- Sorted dataset chronologically  
- Removed missing values  
- Selected **Close price** for univariate forecasting  

### 2. Train-Test Split  
- Historical data split into training and testing sets  
- Train set: Older portion of data  
- Test set: Most recent portion for evaluation  

### 3. Models Implemented  
- **ARIMA:** Captures linear dependencies in stationary time series  
- **SARIMA:** Extends ARIMA with seasonal components  
- **LSTM:** Deep learning model that captures non-linear dependencies and long-term patterns  
- **Prophet:** Decomposes time series into trend, seasonality, and holiday effects  

### 4. Evaluation Metrics  
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAPE (Mean Absolute Percentage Error)**  

---

## 🚀 Installation  

Clone this repository and install required dependencies:  

```bash
git clone https://github.com/your-username/stock-market-forecasting.git
cd stock-market-forecasting
pip install -r requirements.txt
