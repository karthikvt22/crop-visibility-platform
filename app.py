import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  # Correct import for MAE
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

# Check lengths of columns in data
data_lengths = {key: len(value) for key, value in data.items()}
print("Column lengths:", data_lengths)

# Ensure all columns are the same length
if len(set(data_lengths.values())) != 1:
    print("Warning: Columns have different lengths. Adjusting the data...")
    # You can either trim the extra rows or fill missing values
    min_length = min(data_lengths.values())  # Get the shortest column length
    for key in data:
        data[key] = data[key][:min_length]  # Trim to the shortest length

# Now create the DataFrame
df = pd.DataFrame(data)

# Convert the Date column to datetime format
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
print(f"Mean Absolute Error: {mae}")

# Plot 1: Line Plot of Historical Data
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Avg_Price'], marker='o', color='b', label='Average Price')
plt.title('Tomato Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Day_of_Year', 'Month', 'Year', 'Min_Price', 'Max_Price', 'Avg_Price']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Plot 3: Forecasted Price vs Actual Price
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices', color='green')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Summary of model using statsmodels
X2 = sm.add_constant(X_train)  # Add constant to the independent variables for OLS
model_sm = sm.OLS(y_train, X2)  # OLS (Ordinary Least Squares) regression model
results = model_sm.fit()  # Fit the model
print(results.summary())  # Print the model summary

# Plot 4: Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Plot 5: Histogram of Residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Final prediction plot with trendline
plt.figure(figsize=(10, 5))
plt.scatter(X_test['Day_of_Year'], y_test, label='Actual Prices', color='green')
plt.plot(X_test['Day_of_Year'], y_pred, label='Predicted Prices', color='red')
plt.title('Prediction Trendline vs Actual Prices')
plt.xlabel('Day of Year')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
plt.show()
