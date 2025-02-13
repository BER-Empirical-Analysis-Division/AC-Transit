import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data and clean column names
data = pd.read_csv('Survey Data/v2survey_11-22-24.csv')
data.columns = data.columns.str.strip()

# Define price points and corresponding average demand column names
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

# Create DF and remove N/A values
df = pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()

# Set up linear regression model
x = df['avg_demand'].values.reshape(-1,1)
y = df['prices'].values.reshape(-1,1)
model = LinearRegression()
model.fit(x, y)

# Calculate the x-intercept (quantity where price is $0)
a = model.intercept_[0]  # Intercept (WTP at D=0)
b = model.coef_[0][0]    # Slope
max_quantity = -a / b    # Quantity at $0

# Calculate consumer surplus at $2.25
initial_price = 2.25
initial_quantity = (initial_price - a) / b 
initial_cs = 0.5 * initial_quantity * (a - initial_price)

# Calculate consumer surplus at $0
free_price_cs = 0.5 * max_quantity * a

# Compensating variation (difference in consumer surplus)
compensating_variation = free_price_cs - initial_cs

# Demand Curve Visualization with Compensating Variation annotation
extended_demand = np.linspace(0, max_quantity, 100).reshape(-1,1)
predicted_prices = model.predict(extended_demand)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label='Avg. Demand at Price Point')
plt.plot(extended_demand, predicted_prices, 'black', label='Demand Curve')

# Annotate consumer surplus at both price points
# Fill in CS at $2.25
plt.fill_between(
    extended_demand.flatten(), initial_price, predicted_prices.flatten(), 
    where=(extended_demand.flatten() <= initial_quantity), color='yellow', alpha=0.5, 
    label=f'Consumer Surplus at $2.25 = ${initial_cs:.2f}')
plt.hlines(initial_price, 0, initial_quantity, color='blue', linestyle='--', linewidth=1)

# CS at $0
plt.fill_between(
    extended_demand.flatten(), 0, predicted_prices.flatten(), 
    where=(extended_demand.flatten() <= max_quantity), color='lightgreen', alpha=0.3, 
    label=f'Consumer Surplus at $0 = ${free_price_cs:.2f}')

plt.title(f'Change in CS/Compensating Variation = ${compensating_variation:.2f}')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Price ($ per Ride)')
plt.ylim(bottom=-0.5)
plt.legend()
plt.grid(True)
plt.show()
