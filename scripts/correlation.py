from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
def load_data(price_path, gdp_path, inflation_path, exchange_path):
    data = pd.read_csv(price_path, parse_dates=['Date'], index_col='Date')
    gdp_data = pd.read_csv(gdp_path, parse_dates=['Date'], index_col='Date')
    inflation_data = pd.read_csv(inflation_path, parse_dates=['Date'], index_col='Date')
    exchange_data = pd.read_csv(exchange_path, parse_dates=['Date'], index_col='Date')
    
    data = data.join([gdp_data, inflation_data, exchange_data], how='inner')
    data.dropna(inplace=True)
    return data


def exploratory_data_analysis(data):
    plt.figure(figsize=(14, 8))
    for i, column in enumerate(data.columns[1:], 1): 
        plt.subplot(2, 2, i)
        plt.plot(data['Price'], label='Brent Oil Price', color='blue')
        plt.plot(data[column], label=column, color='orange')
        plt.title(f"Brent Oil Price vs {column}")
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("Correlation matrix:")
    print(data.corr())
    
    
def fit_arimax_model(data):
    model = ARIMA(data['Price'], exog=data[['GDP', 'Inflation', 'ExchangeRate']], order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit


def fit_var_model(data):
    model = VAR(data)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

