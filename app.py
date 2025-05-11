import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Helper to suppress Prophet output (optional, makes Streamlit cleaner)
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os, sys # Import sys here

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# --- Dataset (included directly in the code) ---
data = {
    'Date': ['01/01/2021', '02/01/2021', '03/01/2021', '04/01/2021', '05/01/2021', '06/01/2021', '07/01/2021', '08/01/2021',
             '01/02/2021', '02/02/2021', '03/02/2021', '04/02/2021', '05/02/2021', '06/02/2021', '07/02/2021', '08/02/2021',
             '01/03/2021', '02/03/2021', '03/03/2021', '04/03/2021', '05/03/2021', '06/03/2021', '07/03/2021', '08/03/2021',
             '01/04/2021', '02/04/2021', '03/04/2021', '04/04/2021', '05/04/2021', '06/04/2021', '07/04/2021', '08/04/2021',
             '01/05/2021', '02/05/2021', '03/05/2021', '04/05/2021', '05/05/2021', '06/05/2021', '07/05/2021', '08/05/2021',
             '01/06/2021', '02/06/2021', '03/06/2021', '04/06/2021', '05/06/2021', '06/06/2021', '07/06/2021', '08/06/2021',
             '01/07/2021', '02/07/2021', '03/07/2021', '04/07/2021', '05/07/2021', '06/07/2021', '07/07/2021', '08/07/2021',
             '01/08/2021', '02/08/2021', '03/08/2021', '04/08/2021', '05/08/2021', '06/08/2021', '07/08/2021', '08/08/2021',
             '01/09/2021', '02/09/2021', '03/09/2021', '04/09/2021', '05/09/2021', '06/09/2021', '07/09/2021', '08/09/2021',
             '01/10/2021', '02/10/2021', '03/10/2021', '04/10/2021', '05/10/2021', '06/10/2021', '07/10/2021', '08/10/2021',
             '01/11/2021', '02/11/2021', '03/11/2021', '04/11/2021', '05/11/2021', '06/11/2021', '07/11/2021', '08/11/2021',
             '01/12/2021', '02/12/2021', '03/12/2021', '04/12/2021', '05/12/2021', '06/12/2021', '07/12/2021', '08/12/2021'], # 96 entries
    'Min_Price': [400, 600, 600, 600, 500, 600, 500, 600, 800, 1000, 800, 1000, 1200, 1000, 1200, 1200, 1000, 1000, 900, 1000, 1200,
                  1000, 1200, 1000, 1200, 800, 1000, 1200, 1000, 1200, 800, 1200, 1000, 1200, 1500, 1500, 1500, 1200, 1200, 1500,
                  2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 2000, 2000, 2000, 2000, 2000, 2000, 2500, 2000, 2000, 2000,
                  4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], # 68 entries
    'Max_Price': [1000, 1200, 1000, 1200, 1200, 1000, 1200, 1000, 1200, 1400, 1200, 1400, 1600, 1400, 1600, 1600, 1400, 1400,
                  1300, 1400, 1600, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1700, 1800,
                  1800, 1600, 1600, 1800, 2000, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 2000, 2000, 2000, 2000, 2000, 2000,
                  2500, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], # 68 entries
    'Avg_Price': [800, 900, 800, 900, 900, 800, 900, 800, 900, 1100, 900, 1100, 1400, 1100, 1400, 1400, 1100, 1100, 1000, 1100,
                  1400, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1350, 1400, 1400, 1300,
                  1300, 1400, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1600, 1600, 1600, 1600, 1600, 1600, 2000, 1600,
                  1600, 1600, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000] # 68 entries
}

st.title("🌾 Smart Agriculture Assistant")
st.header("🍅 Tomato Price Predictor")
st.write("Select a date and click 'Predict Prices' to see the forecast.")

# --- Prepare Data for Training/Evaluation (using available data) ---

# Determine the minimum length of the lists (assuming price lists dictate available data length)
min_len = min(len(data['Min_Price']), len(data['Max_Price']), len(data['Avg_Price']))
date_list_len = len(data['Date'])

st.sidebar.warning(f"Using built-in data: Dates for {date_list_len} days, price data for {min_len} days.")

# Create a DataFrame using only the first `min_len` dates and all price data
# This aligns the 68 price points with the first 68 dates in the Date list.
data_aligned_for_training = {
    'Date': data['Date'][:min_len],
    'Min_Price': data['Min_Price'][:min_len],
    'Max_Price': data['Max_Price'][:min_len],
    'Avg_Price': data['Avg_Price'][:min_len]
}
df_available = pd.DataFrame(data_aligned_for_training)

# Convert the Date column to datetime format
df_available['Date'] = pd.to_datetime(df_available['Date'], format='%d/%m/%Y')

# Prepare data for Prophet: rename columns and ensure numeric types
# Use the 'Avg_Price' as the 'Modal_Price' for forecasting consistency with the user's previous code structure
df_available = df_available.rename(columns={'Date': 'ds', 'Avg_Price': 'y_modal', 'Min_Price': 'y_min', 'Max_Price': 'y_max'})
df_available['y_modal'] = pd.to_numeric(df_available['y_modal'])
df_available['y_min'] = pd.to_numeric(df_available['y_min'])
df_available['y_max'] = pd.to_numeric(df_available['y_max'])

# Check if there's enough data for training
if len(df_available) < 2:
     st.error("Error: Not enough historical data points to train the model (need at least 2).")
     # Stop here if data is insufficient
     st.stop() # Use st.stop() to halt execution


# --- User Input: Select Date and Button ---
user_date = st.date_input("Select a date for prediction", pd.to_datetime("2025-12-01"))
predict_button = st.button("Predict Prices")

# --- Prediction and Display Logic (runs only after button click) ---
if predict_button:
    st.subheader("📈 Generating Forecast...")

    # --- Train Prophet Models (on available data - happens on button click) ---
    models = {}
    forecasts_full_range = {}
    price_types = ['modal', 'min', 'max']

    try:
        # Define a future range for the overall forecast plot (e.g., end of 2025)
        last_historical_date = df_available['ds'].max()
        future_end_date = pd.to_datetime('2026-12-31') # Forecast further into the future for plot
        # Ensure future dataframe covers the historical range + forecast range
        future_df = pd.DataFrame({'ds': pd.date_range(start=df_available['ds'].min(), end=future_end_date, freq='D')})

        progress_bar = st.progress(0)
        status_text = st.empty()


        for i, price_type in enumerate(price_types):
            col_name = f'y_{price_type}'
            df_prophet = df_available[['ds', col_name]].rename(columns={col_name: 'y'})

            status_text.text(f"Training model for {price_type.capitalize()} Price...")
            progress_bar.progress((i + 1) / len(price_types))

            # Initialize and train Prophet model
            model = Prophet()
            # Suppress model fitting output in Streamlit
            with suppress_stdout():
                 model.fit(df_prophet)

            models[price_type] = model # Store the trained model

            # Predict for the full range of dates for the plot
            forecast = model.predict(future_df)
            forecasts_full_range[price_type] = forecast

        status_text.text("✅ Models Trained Successfully!")
        progress_bar.empty() # Hide progress bar

        # --- Display Predicted Price for Selected Date ---
        st.subheader(f"Predicted Prices on {user_date}:")

        user_date_dt = pd.to_datetime(user_date)
        user_future_df = pd.DataFrame({'ds': [user_date_dt]})

        predicted_prices_user_date = {}

        for price_type in price_types:
             model = models[price_type]
             user_date_forecast = model.predict(user_future_df)
             predicted_prices_user_date[price_type] = user_date_forecast['yhat'].values[0]

        # Display the predictions
        st.success(f"📈 Predicted Prices on {user_date}:")
        st.write(f" - **Modal Price**: ₹{predicted_prices_user_date['modal']:.2f} per quintal")
        st.write(f" - **Min Price**: ₹{predicted_prices_user_date['min']:.2f} per quintal")
        st.write(f" - **Max Price**: ₹{predicted_prices_user_date['max']:.2f} per quintal")


        # --- Display Forecast Plot ---
        st.subheader("📊 Price Forecast: Historical and Predicted Trend")
        fig = plt.figure(figsize=(12, 7))

        # Plot Historical Data Points
        plt.scatter(df_available['ds'], df_available['y_modal'], color='blue', label='Historical Modal Price', s=10)
        plt.scatter(df_available['ds'], df_available['y_min'], color='red', label='Historical Min Price', s=10)
        plt.scatter(df_available['ds'], df_available['y_max'], color='green', label='Historical Max Price', s=10)

        # Plot Predicted Future Trend and Confidence Intervals
        colors = {'modal': 'darkblue', 'min': 'darkred', 'max': 'darkgreen'}

        for price_type in price_types:
             forecast = forecasts_full_range[price_type]
             color = colors[price_type]
             # Plot only the forecast part of the line, starting from the last historical date
             plt.plot(forecast['ds'], forecast['yhat'], label=f'Predicted {price_type.capitalize()} Price Trend', color=color)
             plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color=color, alpha=0.2)


        # Optional: Mark the user-selected date on the plot
        if user_date_dt in future_df['ds'].values:
             # Find the yhat values for the user date from the pre-computed full range forecast
             y_modal_user_pred_plot = forecasts_full_range['modal'][forecasts_full_range['modal']['ds'] == user_date_dt]['yhat'].values
             y_min_user_pred_plot = forecasts_full_range['min'][forecasts_full_range['min']['ds'] == user_date_dt]['yhat'].values
             y_max_user_pred_plot = forecasts_full_range['max'][forecasts_full_range['max']['ds'] == user_date_dt]['yhat'].values

             if y_modal_user_pred_plot.size > 0: # Check if prediction exists for the date
                 plt.axvline(user_date_dt, color='purple', linestyle='--', alpha=0.7, label=f'Selected Date ({user_date.strftime("%Y-%m-%d")})')
                 # Optional: Add point markers for the predictions on the selected date
                 #plt.scatter(user_date_dt, y_modal_user_pred_plot[0], color='purple', s=50, zorder=5)
                 #plt.scatter(user_date_dt, y_min_user_pred_plot[0], color='purple', s=50, zorder=5)
                 #plt.scatter(user_date_dt, y_max_user_pred_plot[0], color='purple', s=50, zorder=5)


        plt.xlabel("Date")
        plt.ylabel("Price (INR/quintal)")
        plt.title("Tomato Price Forecast: Historical Data & Model Prediction")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please try again or check the data if you modified it.")


# --- Main App Layout ---
# The main structure is handled by the flow above, no separate main() needed for this simple app
