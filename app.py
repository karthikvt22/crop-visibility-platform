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
             '01/12/2021', '02/12/2021', '03/12/2021', '04/12/2021', '05/12/2021', '06/12/2021', '07/12/2021', '08/12/2021'],
    'Min_Price': [400, 600, 600, 600, 500, 600, 500, 600, 800, 1000, 800, 1000, 1200, 1000, 1200, 1200, 1000, 1000, 900, 1000, 1200,
                  1000, 1200, 1000, 1200, 800, 1000, 1200, 1000, 1200, 800, 1200, 1000, 1200, 1500, 1500, 1500, 1200, 1200, 1500,
                  2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 2000, 2000, 2000, 2000, 2000, 2000, 2500, 2000, 2000, 2000,
                  4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], # Has 68 entries
    'Max_Price': [1000, 1200, 1000, 1200, 1200, 1000, 1200, 1000, 1200, 1400, 1200, 1400, 1600, 1400, 1600, 1600, 1400, 1400,
                  1300, 1400, 1600, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1700, 1800,
                  1800, 1600, 1600, 1800, 2000, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 2000, 2000, 2000, 2000, 2000, 2000,
                  2500, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], # Has 68 entries
    'Avg_Price': [800, 900, 800, 900, 900, 800, 900, 800, 900, 1100, 900, 1100, 1400, 1100, 1400, 1400, 1100, 1100, 1000, 1100,
                  1400, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1350, 1400, 1400, 1300,
                  1300, 1400, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1600, 1600, 1600, 1600, 1600, 1600, 2000, 1600,
                  1600, 1600, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000] # Has 68 entries
}

# Create DataFrame from the data
# Pandas will automatically use the longest list for the index length (96)
# and fill shorter lists with NaN.
df = pd.DataFrame(data)

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Feature Engineering: Extract Date features for ALL 96 dates
df['Day_of_Year'] = df['Date'].dt.dayofyear
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

st.title("Tomato Price Analysis and Prediction")

# Acknowledge the data inconsistency
st.warning(f"Note: The dataset contains {len(df['Date'])} dates but only {df['Avg_Price'].count()} entries for price data.")

# --- Train and Evaluate Model on AVAILABLE Data ---
# Use only the rows where price data is available for training/evaluation
df_available = df.dropna(subset=['Avg_Price']).copy() # Use copy() to avoid SettingWithCopyWarning

if not df_available.empty:
    st.subheader("Model Training and Evaluation (using available data)")

    # Prepare the dataset for training using available data
    X_available = df_available[['Day_of_Year', 'Month', 'Year']]
    y_available = df_available['Avg_Price']

    # Train-test split (on available data)
    # Check if there's enough data to split
    if len(df_available) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_available, y_available, test_size=0.2, random_state=42)

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
        # Ensure both y_test and y_pred have the same index or plot against index
        plt.plot(y_test.values, label='Actual Prices', color='green')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.title('Actual vs Predicted Prices (Test Set)')
        plt.xlabel('Index of Test Data')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    else:
        st.warning("Not enough data points (only 68 available) to perform train-test split for evaluation.")
        # Train model on all available data if split is not possible
        model = LinearRegression()
        model.fit(X_available, y_available)
        st.info("Model trained on all 68 available data points.")


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
    corr_cols = ['Day_of_Year', 'Month', 'Year', 'Min_Price', 'Max_Price', 'Avg_Price']
    sns.heatmap(df_available[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # --- Predict for the FULL 96 dates and plot the trend ---
    st.subheader("Predicted Tomato Price Trend (Full Year based on model)")
    plt.figure(figsize=(10, 5))

    # Use the model to predict prices for ALL 96 dates
    X_full_dates = df[['Day_of_Year', 'Month', 'Year']]
    y_pred_full = model.predict(X_full_dates)

    plt.plot(df['Date'], y_pred_full, color='red', label='Predicted Average Price (Full Year)')
    # Optionally, plot the historical data points on top for comparison
    plt.scatter(df_available['Date'], df_available['Avg_Price'], color='blue', label='Historical Data Points', zorder=5)

    plt.title('Projected Tomato Price Trend (based on model trained on historical data)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


else:
    st.error("No price data available in the dataset to train the model.")
