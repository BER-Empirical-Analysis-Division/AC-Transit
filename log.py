import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load survey data here
data = pd.read_csv('Survey Data/final_bersurvey.csv')

# Define price points and corresponding average demand column names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Get average demands, prices, and n observations
prices = []
avg_demand = []
n_observations = []
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    n_col = f'n_observations_{price}'
    if avg_col in data.columns and n_col in data.columns:
        avg_value = data[avg_col].mean()  # Mean of avg_demand_{price}
        n_value = data[n_col].sum()  # Total n observations for this price point
        if not np.isnan(avg_value) and n_value > 0:
            prices.append(value)
            avg_demand.append(avg_value)
            n_observations.append(n_value)

# Create DataFrame
df = pd.DataFrame({
    'avg_demand': avg_demand,
    'prices': prices,
    'n_observations': n_observations
}).dropna()

# Quadratic Model (Price in Levels)
x_quad = np.hstack((df['avg_demand'].values.reshape(-1, 1), (df['avg_demand'].values**2).reshape(-1, 1)))
y = df['prices'].values.reshape(-1, 1)

quad_model = LinearRegression()
quad_model.fit(x_quad, y)
quad_predictions = quad_model.predict(x_quad)
quad_rmse = np.sqrt(mean_squared_error(y, quad_predictions))
quad_r2 = quad_model.score(x_quad, y)

# Apply log transformation only for log-log model
df['log_qd'] = np.log(df['avg_demand'] + 0.25)  # Prevents log(0)
df['log_p'] = np.log(df['prices'])

# Identify rows with invalid log values (in both log_qd and log_p)
invalid_rows = df[(df['log_qd'] == -np.inf) | (df['log_p'] == -np.inf) | df['log_qd'].isna() | df['log_p'].isna()]
num_invalid = len(invalid_rows)

print(f"\nNumber of invalid (-inf or NaN) rows removed before log-log regression: {num_invalid}")

# Drop invalid rows (simultaneously for both log_qd and log_p)
df_cleaned = df.drop(invalid_rows.index)

# Ensure the x and y values match in length
x_loglog = sm.add_constant(df_cleaned[['log_qd']])  # Independent variable
y_loglog = df_cleaned['log_p']  # Dependent variable

# Fit Log-Log Model
loglog_model = sm.OLS(y_loglog, x_loglog).fit()
loglog_predictions = loglog_model.predict(x_loglog)
loglog_rmse = np.sqrt(mean_squared_error(y_loglog, loglog_predictions))
loglog_r2 = loglog_model.rsquared

# Residual Plot for Quadratic Model
quad_residuals = y.flatten() - quad_predictions.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(df['avg_demand'], quad_residuals, color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot for Quadratic Model (Price in Levels)')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Residual Plot for Log-Log Model
loglog_residuals = y_loglog - loglog_predictions
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['log_qd'], loglog_residuals, color='green', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot for Log-Log Model')
plt.xlabel('Log(Quantity Demanded)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Print RMSE and R² Comparison
comparison_results = pd.DataFrame({
    'Model': ['Quadratic Model (Price in Levels)', 'Log-Log Model'],
    'RMSE': [quad_rmse, loglog_rmse],
    'R²': [quad_r2, loglog_r2]
})

print("\nModel Comparison: Quadratic vs Log-Log")
print(comparison_results)
