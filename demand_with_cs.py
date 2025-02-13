import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
import scipy.stats as stats

# Load data and clean column names
data = pd.read_csv('Survey Data/final_bersurvey.csv')
data.columns = data.columns.str.strip()

# Define price points/column names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Extract average demands and corresponding prices
prices, avg_demand = [], []
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns:
        avg_value = data[avg_col].mean()
        if not np.isnan(avg_value):
            prices.append(value)
            avg_demand.append(avg_value)

# Create DataFrame and remove N/A values
df = pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()

# Set up quadratic regression model for more accuracy
x_quad = df['avg_demand'].values.reshape(-1, 1)
x_quad = np.column_stack((x_quad, x_quad**2))  # Add quadratic term

# Add constant for OLS
x_with_const = add_constant(x_quad) 

# Perform OLS regression
ols_model = OLS(df['prices'], x_with_const).fit()

# Extract coefficients and standard errors
beta_0, beta_1, beta_2 = ols_model.params
std_error_beta_0, std_error_beta_1, std_error_beta_2 = ols_model.bse

# Calculate 95% confidence intervals for the coefficients
z_score = stats.norm.ppf(0.975)  # For 95% confidence interval
ci_lower_beta_1 = beta_1 - z_score * std_error_beta_1
ci_upper_beta_1 = beta_1 + z_score * std_error_beta_1
ci_lower_beta_2 = beta_2 - z_score * std_error_beta_2
ci_upper_beta_2 = beta_2 + z_score * std_error_beta_2

# Print results of the hypothesis test for quadratic specification
print("\nHypothesis Test for Quadratic Specification:")
print(f"Coefficient for linear term (beta_1): {beta_1:.4f}")
print(f"Coefficient for quadratic term (beta_2): {beta_2:.4f}")
print(f"95% Confidence Interval for beta_1: [{ci_lower_beta_1:.4f}, {ci_upper_beta_1:.4f}]")
print(f"95% Confidence Interval for beta_2: [{ci_lower_beta_2:.4f}, {ci_upper_beta_2:.4f}]")

# Calculate consumer surplus using the quadratic demand curve
def calculate_consumer_surplus_quad(beta_0, beta_1, beta_2, max_quantity):
    # Consumer surplus for quadratic model is the integral of the demand curve
    return (beta_0 * max_quantity) + (beta_1 * max_quantity**2 / 2) + (beta_2 * max_quantity**3 / 3)

# Calculate the quantity at price = 0 (intercept)
max_quantity = -beta_0 / beta_1  # Using the quadratic model formula Q = -beta_0 / beta_1

# Calculate consumer surplus (full area under the curve from 0 to Q_max)
consumer_surplus = calculate_consumer_surplus_quad(beta_0, beta_1, beta_2, max_quantity)
print(f"\nConsumer Surplus Estimate: ${consumer_surplus:.2f}")

# Generate demand curve predictions (from 0 to max_quantity)
extended_demand = np.linspace(0, max_quantity, 100)
predicted_prices = beta_0 + beta_1 * extended_demand + beta_2 * extended_demand**2

# Calculate confidence intervals for the consumer surplus
ci_lower = calculate_consumer_surplus_quad(beta_0 - z_score * std_error_beta_0,
                                           beta_1 - z_score * std_error_beta_1,
                                           beta_2 - z_score * std_error_beta_2, max_quantity)
ci_upper = calculate_consumer_surplus_quad(beta_0 + z_score * std_error_beta_0,
                                           beta_1 + z_score * std_error_beta_1,
                                           beta_2 + z_score * std_error_beta_2, max_quantity)

print(f"95% Confidence Interval for Consumer Surplus: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# Plot the demand curve with confidence intervals
plt.figure(figsize=(8, 6))
plt.plot(df['avg_demand'], df['prices'], 'bo', label='Observed Demand Points')
plt.plot(extended_demand, predicted_prices, 'black', label='Fitted Demand Curve')

# Annotate consumer surplus on the plot
plt.fill_between(extended_demand, 0, predicted_prices, color='lightgreen', alpha=0.5, label=f'Consumer Surplus = ${consumer_surplus:.2f}')

plt.title('Demand Curve with Confidence Intervals for Consumer Surplus')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Price ($ per Ride)')
plt.ylim(bottom=-0.5)
plt.legend()
plt.grid(True)
plt.show()
