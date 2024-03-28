from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Function to fetch stock price data from Alpha Vantage
def fetch_stock_data(symbol):
    api_key = 'XC2LIVVPKXOO6K2U'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    return data['Time Series (Daily)']

# Function to train the linear regression model
def train_model(stock_data):
    df = pd.DataFrame(stock_data).transpose()
    df.index = pd.to_datetime(df.index)

    column_names = df.columns.tolist()
    open_index = column_names.index('1. open')
    high_index = column_names.index('2. high')
    low_index = column_names.index('3. low')
    volume_index = column_names.index('5. volume')
    close_index = column_names.index('4. close')
    X = df.iloc[:, [open_index, high_index, low_index, volume_index]].astype(float)
    y = df.iloc[:, close_index].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['stockSymbol']
        stock_data = fetch_stock_data(symbol)
        model = train_model(stock_data)
        next_day_features = pd.DataFrame(stock_data).iloc[-1:].astype(float)
        next_day_price = model.predict(next_day_features[['1. open', '2. high', '3. low', '5. volume']])
        return render_template('index.html', next_day_price=next_day_price[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
