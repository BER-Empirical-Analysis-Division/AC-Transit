import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.api import OLS, add_constant

# Load updated survey data
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

# Standard deviation for entire survey
overall_std = data[[f'avg_demand_{price}' for price in price_points.keys()]].stack().std()

z_score = 1.960  # Critical z-score for 95% confidence

# Calculate standard error and confidence intervals using z-score
df['std_error'] = overall_std / np.sqrt(df['n_observations'])  # Standard Error
df['ci_lower'] = df['avg_demand'] - z_score * df['std_error']
df['ci_upper'] = df['avg_demand'] + z_score * df['std_error']

# Print confidence intervals
print("\nConfidence Interval Values for Each Price Point:")
for _, row in df.iterrows():
    print(
        f"Price: ${row['prices']:.2f}, "
        f"Avg. Demand: {row['avg_demand']:.2f}, "
        f"CI Lower: {row['ci_lower']:.2f}, "
        f"CI Upper: {row['ci_upper']:.2f}"
    )

# Linear regression
x = df['avg_demand'].values.reshape(-1, 1)
y = df['prices'].values.reshape(-1, 1)
linear_model = LinearRegression()
linear_model.fit(x, y)
linear_predictions = linear_model.predict(x)
linear_rmse = np.sqrt(mean_squared_error(y, linear_predictions))

# Quadratic regression
x_quad = np.hstack((x, x**2))
quad_model = LinearRegression()
quad_model.fit(x_quad, y)
quad_predictions = quad_model.predict(x_quad)
quad_rmse = np.sqrt(mean_squared_error(y, quad_predictions))

# Generate predictions for visualization
extended_demand = np.linspace(0, max(x), 100).reshape(-1, 1)
extended_demand_quad = np.hstack((extended_demand, extended_demand**2))
linear_curve = linear_model.predict(extended_demand)
quad_curve = quad_model.predict(extended_demand_quad)

# Plot 1: Demand Curve with RMSE and Corrected Confidence Intervals
plt.figure(figsize=(10, 6))
plt.errorbar(
    df['avg_demand'], 
    df['prices'], 
    xerr=z_score * df['std_error'],  # Confidence Interval for QD
    fmt='o', 
    label='Avg. Demand with CI', 
    capsize=5, 
    color='blue'
)
plt.plot(extended_demand, linear_curve, label=f'Linear Model (RMSE: {linear_rmse:.2f})', linestyle='--', color='red')
plt.plot(extended_demand, quad_curve, label=f'Quadratic Model (RMSE: {quad_rmse:.2f})', color='black')
plt.title('Demand Curve with RMSE and Confidence Intervals')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Price ($ per Ride)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Demand Distribution Analysis (Boxplots)
plt.figure(figsize=(10, 6))
avg_demand_columns = [f'avg_demand_{price}' for price in price_points.keys()]
demand_data = data[avg_demand_columns]

plt.boxplot(
    [demand_data[col].dropna() for col in avg_demand_columns],
    tick_labels=price_points.keys(),
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue"),
    flierprops=dict(markerfacecolor="green", marker="o")
)
plt.title("Distribution of Average Demand Across Price Points")
plt.xlabel("Price Points")
plt.ylabel("Quantity Demanded (Rides per Week)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Hypothesis Test for Quadratic Coefficient
x_with_const = add_constant(x_quad)  # Add constant for OLS
ols_model = OLS(df['prices'], x_with_const).fit()

# Extract the coefficient and standard error for x^2
beta_2 = ols_model.params.iloc[2] 
std_error_beta_2 = ols_model.bse.iloc[2]

beta_1 = ols_model.params.iloc[1]

# Calculate 95% confidence interval
ci_lower = beta_2 - z_score * std_error_beta_2
ci_upper = beta_2 + z_score * std_error_beta_2

# Print hypothesis test results
print("\nHypothesis Test for Quadratic Specification:")
print(f"Coefficient for x^2: {beta_2:.4f}")
print(f"Coefficient for x: {beta_1: .4f}")
print(f"Standard Error: {std_error_beta_2:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower <= 0 <= ci_upper:
    print("Result: Fail to reject the null hypothesis (H0). The quadratic specification is not necessary (linear model is sufficient).")
else:
    print("Result: Reject the null hypothesis (H0). The quadratic specification is appropriate.")


all_avg_demand = data[[f'avg_demand_{price}' for price in price_points.keys()]].stack().dropna()

plt.figure(figsize=(10, 6))
plt.hist(all_avg_demand, bins=20, density=True, color='skyblue', alpha=0.7)
plt.title("Overall Distribution of Average Demand")
plt.xlabel("Average Demand (Rides per Week)")
plt.ylabel("Density")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Filter out any nonpositive values (since log(x) is only defined for x>0)
all_avg_demand = all_avg_demand[all_avg_demand > 0]
log_avg_demand = np.log(all_avg_demand)

plt.figure(figsize=(10, 6))
plt.hist(log_avg_demand, bins=20, density=True, color='lightgreen', alpha=0.7)
plt.title("Distribution of Log(Quantity Demanded)")
plt.xlabel("Log(Quantity Demanded)")
plt.ylabel("Density")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()