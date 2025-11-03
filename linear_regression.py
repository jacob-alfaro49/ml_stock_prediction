import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Download data
ticker = 'TSLA'  # Stock name
df = yf.download(ticker, start='2020-01-01', end='2025-01-01')

# calculate moving averages
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()

# Drop any rows with missing values
df.dropna(inplace=True)

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']]
y = df['Close']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.5, shuffle=False
)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation for {ticker}:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Price', linewidth=2)
plt.plot(y_pred, label='Predicted Price', linewidth=2)
plt.title(f'{ticker} Stock Price Prediction (Linear Regression)')
plt.xlabel('Time (test period)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()
