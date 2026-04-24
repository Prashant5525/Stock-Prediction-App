import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime

# Step 1: Download stock data
stock = "GOOG"
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

df = yf.download(stock, start, end)
close_prices = df[["Close"]].dropna()

# Step 2: Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Step 3: Create sequences of 100 days
X, y = [], []
sequence_length = 100

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Optional: Split into train/test if you want
# Here we train on the entire data (like your app assumes)

# Step 4: Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X, y, epochs=30, batch_size=64, callbacks=[early_stop])

# Step 6: Save the model
model.save("Latest_stock_price_model.keras")

print("✅ Model training complete and saved.")
