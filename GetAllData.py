# This script is used to retrieve all historical information about the exchange rate of USD/RMB.
# Once you have the csv file, change the first column header space to "Date"
# The API limit is 25 times a day
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set Alpha Vantage API key
def get_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.readline().strip()
    return api_key

api_key = get_api_key('Alpha_Vantage_API')


# functions to get exchange rate data
def get_exchange_rate_data(from_currency, to_currency, api_key):
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    if 'Time Series FX (Daily)' in data:
        df = pd.DataFrame(data['Time Series FX (Daily)']).T
        df.columns = ['Open', 'High', 'Low', 'Close']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print("Error fetching data:", data)
        return None


# Get USD/RMB exchange rate data
df = get_exchange_rate_data('USD', 'CNY', api_key)

if df is not None:
    print(df.head())

    df.to_csv('usd_cny_exchange_rate.csv')
    print("Data saved to usd_cny_exchange_rate.csv")

    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    df_recent = df[(df.index >= start_date) & (df.index <= end_date)]

    # Plot the closing price history over the last five years
    plt.figure(figsize=(12, 6))
    plt.plot(df_recent['Close'].astype(float))
    plt.title('USD/CNY Exchange Rate Over Last 5 Years')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.grid(True)
    plt.show()
