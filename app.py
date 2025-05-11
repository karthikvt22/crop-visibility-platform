import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# Sample data for testing
data = {
    'Date': ['01/01/2020', '02/01/2020', '03/01/2020', '04/01/2020', '05/01/2020'],
    'Avg_Price': [25, 28, 26, 29, 27],
    'Min_Price': [20, 22, 21, 23, 21],
    'Max_Price': [30, 32, 31, 33, 32]
}

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
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Avg_Price'], marker='o', color='b', label='Average Price')
plt.title('Tomato Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Plot 2: Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Day_of_Year', 'Month', 'Year', 'Min_Price', 'Max_Price', 'Avg_Price']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
st.pyplot(plt)

# Plot 3: Forecasted Price vs Actual Price
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices', color='green')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Summary of model
X2 = sm.add_constant(X_train)  # Add constant to the independent variables for OLS
model_sm = sm.OLS(y_train, X2)  # OLS (Ordinary Least Squares) regression model
results = model_sm.fit()  # Fit the model
st.write(results.summary())  # Print the model summary

# Plot 4: Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.grid(True)
st.pyplot(plt)

# Plot 5: Histogram of Residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
st.pyplot(plt)

# Final prediction plot with trendline
plt.figure(figsize=(10, 5))
plt.scatter(X_test['Day_of_Year'].values, y_test.values, label='Actual Prices', color='green')  # Use .values
plt.plot(X_test['Day_of_Year'].values, y_pred, label='Predicted Prices', color='red')  # Use .values
plt.title('Prediction Trendline vs Actual Prices')
plt.xlabel('Day of Year')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
st.pyplot(plt)
