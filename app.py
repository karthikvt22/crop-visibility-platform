import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# Dataset you provided
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

st.title("Tomato Price Analysis and Prediction")

# --- Prepare Data for Training/Evaluation (using available data) ---

# Determine the minimum length of the lists
min_len = min(len(lst) for lst in data.values())

# Create a new dictionary with lists truncated to the minimum length
data_aligned = {key: value[:min_len] for key, value in data.items()}

# Create DataFrame from the aligned data (this will have 68 rows)
df_available = pd.DataFrame(data_aligned)

# Convert the Date column to datetime format
df_available['Date'] = pd.to_datetime(df_available['Date'], format='%d/%m/%Y')

# Feature Engineering: Extract Date features for the available data
df_available['Day_of_Year'] = df_available['Date'].dt.dayofyear
df_available['Month'] = df_available['Date'].dt.month
df_available['Year'] = df_available['Date'].dt.year # Year is constant 2021 here

st.warning(f"Note: The dataset contains dates for {len(data['Date'])} days, but only {min_len} days have complete price data. Analysis and model training are based on the {min_len} available data points.")

# --- Train and Evaluate Model on AVAILABLE Data ---
if not df_available.empty:
    st.subheader("Model Training and Evaluation (using available data)")

    # Prepare the dataset for training using available data
    # Using 'Day_of_Year' and 'Month' as 'Year' is constant in this dataset
    X_available = df_available[['Day_of_Year', 'Month']]
    y_available = df_available['Avg_Price']

    # Train-test split (on available data)
    if len(df_available) > 1: # Need at least 2 samples for train_test_split
        # Adjust test_size if data is very small, or use LOOCV/cross-validation
        # For simplicity with this small dataset, let's ensure a valid split is possible
        test_size = max(0.2, 1/len(df_available)) # Ensure test set has at least 1 sample if possible
        X_train, X_test, y_train, y_test = train_test_split(X_available, y_available, test_size=test_size, random_state=42)

        # Create and train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions on the test set (from available data)
        y_pred = model.predict(X_test)

        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Mean Absolute Error on available data test set: {mae:.2f}")

        # Plot: Actual vs Predicted Prices (Test Set from available data)
        st.subheader("Actual vs Predicted Prices (Test Set - available data)")
        plt.figure(figsize=(10, 5))
        # Ensure both y_test and y_pred align for plotting - plot against index
        plt.plot(y_test.values, label='Actual Prices', color='green')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.title('Actual vs Predicted Prices (Test Set)')
        plt.xlabel('Index of Test Data')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    else:
        st.warning("Not enough data points to perform train-test split for evaluation (need at least 2). Training model on the single available data point.")
        # Train model on the single available data point if split is not possible
        model = LinearRegression()
        # Reshape X_available for fit if it's a single sample
        model.fit(X_available.values.reshape(1, -1), y_available.values.reshape(1, -1))
        st.info("Model trained on the available data point.")


    # --- Visualization ---

    # Plot 1: Line Plot of Historical Data (Available data only)
    st.subheader("Historical Tomato Price Trend (Available Data)")
    plt.figure(figsize=(10, 5))
    plt.plot(df_available['Date'], df_available['Avg_Price'], marker='o', color='b', label='Average Price')
    plt.title('Tomato Price Trend Over Time (Available Data)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Plot 2: Correlation Matrix Heatmap (Using available data)
    st.subheader("Correlation Matrix (using available data)")
    plt.figure(figsize=(8, 6))
    # Include only relevant columns from available data
    corr_cols = ['Day_of_Year', 'Month', 'Min_Price', 'Max_Price', 'Avg_Price'] # Exclude Year as it's constant
    sns.heatmap(df_available[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # --- Predict for the FULL 96 dates and plot the trend ---
    st.subheader("Projected Tomato Price Trend (Full Date Range based on model)")
    st.info("This plot shows the model's prediction for the price trend across all dates provided, using a model trained only on the available data.")
    plt.figure(figsize=(10, 5))

    # Create a DataFrame for the full 96 dates (need to recreate dates and features)
    df_full_dates = pd.DataFrame({'Date': data['Date']})
    df_full_dates['Date'] = pd.to_datetime(df_full_dates['Date'], format='%d/%m/%Y')
    df_full_dates['Day_of_Year'] = df_full_dates['Date'].dt.dayofyear
    df_full_dates['Month'] = df_full_dates['Date'].dt.month
    # Assuming the prediction model is based on 'Day_of_Year' and 'Month'
    X_full_dates = df_full_dates[['Day_of_Year', 'Month']]


    # Use the model to predict prices for ALL 96 dates
    if 'model' in locals(): # Check if the model was successfully trained
         y_pred_full = model.predict(X_full_dates)

         plt.plot(df_full_dates['Date'], y_pred_full, color='red', label='Predicted Average Price')
         # Optionally, plot the historical data points on top for comparison
         plt.scatter(df_available['Date'], df_available['Avg_Price'], color='blue', label='Historical Data Points', zorder=5)

         plt.title('Projected Tomato Price Trend')
         plt.xlabel('Date')
         plt.ylabel('Average Price')
         plt.legend()
         plt.grid(True)
         st.pyplot(plt)
    else:
        st.warning("Model could not be trained due to insufficient data, so full-year prediction plot is not available.")


else:
    st.error("No price data available in the dataset (after checking lengths) to train the model.")
