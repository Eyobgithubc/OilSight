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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense



def apply_lstm_model(data, model_save_path='brent_oil_lstm_model.h5'):
    """
    Apply LSTM for time series prediction on Brent oil prices.
    """
    # Preprocess data for LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

    # Prepare data for LSTM
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Split into train/test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate model performance on test set
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # Save the trained model to a H5 file (recommended for Keras models)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, color='blue', label='Actual Brent Oil Price')
    plt.plot(y_pred_rescaled, color='red', label='Predicted Brent Oil Price')
    plt.title('Brent Oil Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return model
