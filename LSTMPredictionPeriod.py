import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import matplotlib.pyplot as plt

# Input Data
data = pd.read_csv('usd_cny_exchange_rate.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Predict the Close price
close_data = data[['Close']]

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Create dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

# Split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Adjust the input data shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# A function to create the LSTM model
def create_model(optimizer='adam', units=100):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Use the best parameters to re-train the best model
best_params = {'batch_size': 64, 'epochs': 100, 'optimizer': 'adam', 'units': 100}
best_model = create_model(optimizer=best_params['optimizer'], units=best_params['units'])
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Create a directory to save model
model_dir = 'best_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
best_model.save(os.path.join(model_dir, 'model.h5'))

# Predict
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

# Anti-normalized predicted value
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))

# Reverse normalize the actual value
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculation error (RMSE)
train_score = np.sqrt(np.mean((train_predict - train_actual) ** 2))
test_score = np.sqrt(np.mean((test_predict - test_actual) ** 2))
print(f'Train Score: {train_score} RMSE')
print(f'Test Score: {test_score} RMSE')

# Show in picture
plt.figure(figsize=(12, 6))
plt.plot(close_data.index[-len(test_predict):], test_actual, label='Actual')
plt.plot(close_data.index[-len(test_predict):], test_predict, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.title('USD to CNY Exchange Rate Prediction')
plt.legend()
plt.show()

# Predict future values
def predict_future(model, data, look_back, n_days):
    future_predictions = []
    current_input = data[-look_back:]  # Starting with the last look_back days from the training data
    current_input = current_input.reshape((1, look_back, 1))

    for _ in range(n_days):
        future_pred = model.predict(current_input)
        future_predictions.append(future_pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], np.reshape(future_pred, (1, 1, 1)), axis=1)  # Slide the window

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

n_days = 10  # Number of days to predict into the future
future_predictions = predict_future(best_model, scaled_data, look_back, n_days)

# Show future predictions
future_dates = pd.date_range(start=close_data.index[-1], periods=n_days + 1, inclusive='right')

# Plot only the last 20 days of historical data and future predictions
plt.figure(figsize=(12, 6))
plt.plot(close_data.index[-20:], close_data['Close'][-20:], label='Historical Data (Last 20 days)')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.title('USD to CNY Exchange Rate Future Prediction')
plt.legend()
plt.show()
