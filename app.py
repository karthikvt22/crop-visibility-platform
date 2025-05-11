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
                  4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000],
    'Max_Price': [1000, 1200, 1000, 1200, 1200, 1000, 1200, 1000, 1200, 1400, 1200, 1400, 1600, 1400, 1600, 1600, 1400, 1400,
                  1300, 1400, 1600, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1600, 1000, 1400, 1600, 1400, 1700, 1800,
                  1800, 1600, 1600, 1800, 2000, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 2000, 2000, 2000, 2000, 2000, 2000,
                  2500, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000],
    'Avg_Price': [800, 900, 800, 900, 900, 800, 900, 800, 900, 1100, 900, 1100, 1400, 1100, 1400, 1400, 1100, 1100, 1000, 1100,
                  1400, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1400, 900, 1100, 1400, 1100, 1350, 1400, 1400, 1300,
                  1300, 1400, 1600, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1600, 1600, 1600, 1600, 1600, 1600, 2000, 1600,
                  1600, 1600, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
}

# Check if the lengths are equal for all lists
if all(len(value) == len(data['Date']) for key, value in data.items()):
    # Convert the Date column to datetime format
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Feature Engineering: Extract Date features
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Prepare the dataset for training
    X = df[['Day_of_Year', 'Month', 'Year']]
    y = df['Avg_Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae}")

    # Plot 1: Line Plot of Historical Data
    st.subheader("Tomato Price Trend Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Avg_Price'], marker='o', color='b', label='Average Price')
    plt.title('Tomato Price Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Plot 2: Correlation Matrix Heatmap
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['Day_of_Year', 'Month', 'Year', 'Min_Price', 'Max_Price', 'Avg_Price']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # Plot 3: Forecasted Price vs Actual Price
    st.subheader("Actual vs Predicted Prices (Test Set)")
    plt.figure(figsize=(10, 5))
    # Ensure both y_test and y_pred have the same index or plot against index
    plt.plot(y_test.values, label='Actual Prices', color='green') # Using .values is safer here as y_test is a Series
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Index of Test Data')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt) # Use st.pyplot here to display in Streamlit

else:
    st.error("Data lists have unequal lengths. Please check the dataset.")
