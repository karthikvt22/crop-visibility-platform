import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Manual Tomato Price Data (Jan + Feb 2021) ---
def get_tomato_data():
    data = [
        ("01/01/2021", 1200, 1600, 1400), ("02/01/2021", 1200, 1600, 1400), ("03/01/2021", 1200, 1600, 1400),
        ("04/01/2021", 1200, 1600, 1400), ("06/01/2021", 1000, 1600, 1300), ("07/01/2021", 1000, 1500, 1300),
        ("08/01/2021", 1000, 1500, 1200), ("09/01/2021", 1000, 1400, 1200), ("11/01/2021", 1000, 1400, 1200),
        ("12/01/2021", 1000, 1400, 1200), ("13/01/2021", 1000, 1400, 1200), ("16/01/2021", 1000, 1400, 1200),
        ("17/01/2021", 1000, 1400, 1200), ("18/01/2021", 1000, 1400, 1200), ("20/01/2021", 1000, 1400, 1200),
        ("21/01/2021", 1000, 1400, 1200), ("22/01/2021", 1000, 1400, 1200), ("23/01/2021", 1000, 1400, 1200),
        ("24/01/2021", 1000, 1400, 1200), ("25/01/2021", 1000, 1400, 1200), ("27/01/2021", 1000, 1400, 1200),
        ("28/01/2021", 1000, 1400, 1200), ("30/01/2021", 1000, 1400, 1200),
        ("01/02/2021", 1200, 1600, 1400), ("02/02/2021", 1200, 1600, 1300), ("03/02/2021", 1200, 1600, 1300),
        ("04/02/2021", 1200, 1600, 1300), ("05/02/2021", 1200, 1600, 1300), ("06/02/2021", 1200, 1600, 1300),
        ("08/02/2021", 1200, 1600, 1300), ("09/02/2021", 1200, 1600, 1300), ("10/02/2021", 1200, 1600, 1300),
        ("11/02/2021", 1200, 1600, 1300), ("12/02/2021", 1200, 1600, 1400), ("15/02/2021", 1200, 1600, 1400),
        ("18/02/2021", 1000, 1400, 1200), ("20/02/2021", 1000, 1400, 1200), ("22/02/2021", 1000, 1400, 1200),
        ("23/02/2021", 1000, 1400, 1200), ("24/02/2021", 1000, 1400, 1200), ("25/02/2021", 1000, 1400, 1200),
    ]
    df = pd.DataFrame(data, columns=["Date", "Min", "Max", "Modal"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

# --- Train Prophet Model ---
def train_model(df, column_name):
    model_df = df.rename(columns={"Date": "ds", column_name: "y"})[["ds", "y"]]
    model = Prophet()
    model.fit(model_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast

# --- Plot Forecast ---
def plot_forecast(forecast, title, user_date):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3)
    plt.axvline(pd.to_datetime(user_date), color="red", linestyle="--", label="Prediction Date")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (INR/quintal)")
    plt.legend()
    return fig

# --- Main Tomato Price Predictor App ---
def tomato_price_predictor():
    st.header("🍅 Tomato Price Predictor (Min / Max / Modal)")
    user_date = st.date_input("📅 Pick a future date", pd.to_datetime("2024-12-01"))

    df = get_tomato_data()

    # Forecasts
    forecast_modal = train_model(df, "Modal")
    forecast_min = train_model(df, "Min")
    forecast_max = train_model(df, "Max")

    # Get predicted values for selected date
    def get_price(forecast):
        row = forecast[forecast["ds"] == pd.to_datetime(user_date)]
        return row["yhat"].values[0] if not row.empty else None

    modal_price = get_price(forecast_modal)
    min_price = get_price(forecast_min)
    max_price = get_price(forecast_max)

    if None not in (modal_price, min_price, max_price):
        st.success(f"📌 Predicted Prices for {user_date.strftime('%d-%b-%Y')}:")
        st.markdown(f"""
        - ✅ **Modal Price**: ₹{modal_price:.2f}  
        - 🔻 **Min Price**: ₹{min_price:.2f}  
        - 🔺 **Max Price**: ₹{max_price:.2f}
        """)
    else:
        st.warning("⚠️ Unable to predict for the selected date. Try a closer one.")

    # Forecast Graphs
    st.subheader("📈 Modal Price Forecast")
    st.pyplot(plot_forecast(forecast_modal, "Forecast: Modal Price", user_date))

    st.subheader("📉 Min Price Forecast")
    st.pyplot(plot_forecast(forecast_min, "Forecast: Minimum Price", user_date))

    st.subheader("📊 Max Price Forecast")
    st.pyplot(plot_forecast(forecast_max, "Forecast: Maximum Price", user_date))

    # Historical data
    st.subheader("📋 Historical Data Used")
    st.dataframe(df)

# --- Streamlit App Layout ---
def main():
    st.title("🌾 Smart Agriculture Assistant")
    st.sidebar.title("🔍 Choose Feature")
    app_mode = st.sidebar.radio("Select Option", ["Tomato Price Predictor"])

    if app_mode == "Tomato Price Predictor":
        tomato_price_predictor()

if __name__ == "__main__":
    main()
