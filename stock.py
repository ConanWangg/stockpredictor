import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to fetch stock price data from Alpha Vantage
def fetch_stock_data(symbol):
    api_key = 'XC2LIVVPKXOO6K2U'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    return data['Time Series (Daily)']

# Fetch historical stock price data for a specific symbol (e.g., AAPL for Apple)
symbol = 'AAPL'
stock_data = fetch_stock_data(symbol)

# Convert the data to a DataFrame
df = pd.DataFrame(stock_data).transpose()
df.index = pd.to_datetime(df.index)
print(df.columns)

# Extract relevant features (Open, High, Low, Volume) and target (Close)
column_names = df.columns.tolist()
open_index = column_names.index('1. open')
high_index = column_names.index('2. high')
low_index = column_names.index('3. low')
volume_index = column_names.index('5. volume')
close_index = column_names.index('4. close')
X = df.iloc[:, [open_index, high_index, low_index, volume_index]].astype(float)
y = df.iloc[:, close_index].astype(float)

#X = df[['1. open', '2. high', '3. low', '6. volume']].astype(float)
#y = df['4. close'].astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Example of using the model to predict the next day's closing price
next_day_features = X.tail(1)  # Features for the most recent day
next_day_price = model.predict(next_day_features)
print("Predicted Close Price for Next Day:", next_day_price[0])
