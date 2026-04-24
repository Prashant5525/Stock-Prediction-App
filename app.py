import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import yfinance as yf

# Title and Instructions
st.title("📈 Stock Price Predictor App")
st.markdown("""
This app uses a pre-trained machine learning model to predict stock prices based on historical data.
You can input the stock ticker (e.g., AAPL for Apple), choose the date range, select the moving average window,
and see predictions compared to actual prices.
""")

# Stock Input
stock = st.text_input("Enter the Stock ID (e.g., GOOG, AAPL, MSFT)", "GOOG")

# Date Range Selection
start = st.date_input("Select Start Date", datetime.now() - timedelta(days=365*15))
end = st.date_input("Select End Date", datetime.now())

# Future Forecasting Days
forecast_days = st.slider("Predict Next N Days", min_value=1, max_value=30, value=7)

# Fetch Data
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start, end)
        return data
    except:
        return pd.DataFrame()

google_data = fetch_data(stock, start, end)

if google_data.empty:
    st.error("No data found. Please enter a valid stock symbol and try again.")
    st.stop()

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = google_data[['Close']].iloc[splitting_len:].copy()

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig, use_container_width=True)

# Future Forecasting
st.subheader(f"Forecasting Next {forecast_days} Days")
latest_data = scaled_data[-100:]
future_predictions = []
input_seq = latest_data.reshape(1, 100, 1)

for _ in range(forecast_days):
    next_pred = model.predict(input_seq)[0]
    future_predictions.append(next_pred[0])
    next_input = np.append(input_seq[0][1:], [[next_pred[0]]], axis=0)
    input_seq = next_input.reshape(1, 100, 1)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=google_data.index[-1] + timedelta(days=1), periods=forecast_days)
future_df = pd.DataFrame(future_prices, index=future_dates, columns=["Forecasted Close"])

st.dataframe(future_df, use_container_width=True)

fig3 = plt.figure(figsize=(15,6))
plt.plot(google_data['Close'], label="Historical Close")
plt.plot(future_df, label="Forecast")
plt.legend()
st.pyplot(fig3, use_container_width=True)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(inv_y_test, inv_pre))
st.metric("RMSE", f"{rmse:.2f}")

csv = ploting_data.to_csv().encode('utf-8')
st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")