import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import matplotlib.pyplot as plt

# Input Data
data = pd.read_csv('usd_cny_exchange_rate.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Predict the Close price
close_data = data[['Close']]

# data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)


# normalize the dataset
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
def create_model(optimizer='adam', units=50):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


# use KerasRegressor to pack the model
model = create_model(optimizer='adam', units=50)
model.compile()
LSTMmodel = KerasRegressor(build_fn=create_model)

# Define the parameters of the grid search
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 50, 100],
    'units': [50, 100],
    'optimizer': ['adam', 'rmsprop']
}

# create GridCV
grid = GridSearchCV(estimator=LSTMmodel, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

grid_result = grid.fit(X_train, y_train)
print("Best parameters found: ", grid_result.best_params_)

# Use the best parameters to re-train the best model
best_model = create_model(optimizer=grid_result.best_params_['optimizer'], units=grid_result.best_params_['units'])
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'],
               batch_size=grid_result.best_params_['batch_size'], verbose=1)

# Create a directory to save model
model_dir = 'best_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
best_model.save(os.path.join(model_dir, 'model.h5'))

# predict
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

# Anti-normalized predicted value
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))

# Reverse normalize the actual value
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# calculation error(RMSE)
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
