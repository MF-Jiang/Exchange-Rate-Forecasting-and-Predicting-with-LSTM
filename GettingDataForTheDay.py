# This SCRIPT is used TO check and get the latest exchange rate information on a daily basis
import requests
import pandas as pd
from datetime import datetime

# Set Alpha Vantage API key
def get_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.readline().strip()
        print(api_key)
    return api_key

api_key = get_api_key('Alpha_Vantage_API')
csv_file = 'usd_cny_exchange_rate.csv'


# Define functions to get exchange rate data
def get_today_exchange_rate(from_currency, to_currency, api_key):
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    if 'Time Series FX (Daily)' in data:
        rate = data['Time Series FX (Daily)']
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        if today in rate:
            today_entry = data[today]
            print(today_entry)
            exchange_rate = {
                'Date': datetime.now().strftime('%Y-%m-%d  0:00:00'),
                'Open': today_entry['1. open'],
                'High': today_entry['2. high'],
                'Low': today_entry['3. low'],
                'Close': today_entry['4. close']
            }
            return exchange_rate
        else:
            # print("The data on the web page is not updated.")
            return None
    else:
        print("Error fetching data:", data)
        return None


# 检查并更新CSV文件
def check_and_update_csv(csv_file, api_key):
    try:
        # Read the CSV file and specify the column names
        df = pd.read_csv(csv_file, header=0, names=['Date', 'Open', 'High', 'Low', 'Close'], parse_dates=['Date'],
                         index_col='Date')

        # Gets the latest date in the CSV file
        last_date = df.index[-1]
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d  0:00:00'))

        # Check if the latest date is today
        if last_date == today:
            print("The CSV file is the latest and does not need to be updated.")
            return

        # Get today's exchange rate data
        new_data = get_today_exchange_rate('USD', 'CNY', api_key)

        if new_data:
            # Adds new data to the data box
            new_row = pd.DataFrame([new_data]).set_index('Date')
            df = pd.concat([df, new_row])

            # Remove possible empty lines
            df = df.dropna(how='all')

            df.to_csv(csv_file, header=True)
            print("The CSV file has been updated.")
        else:
            print("No new data today.")
    except Exception as e:
        print(f"Temporal error occurred：{e}")


check_and_update_csv(csv_file, api_key)
