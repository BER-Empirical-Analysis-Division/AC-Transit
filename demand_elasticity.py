import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load updated data with average demand columns
data = pd.read_csv('Survey Data/final_bersurvey.csv')

# Define price points/columns names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Get average demands from CSV
prices = []
avg_demand = []
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns:
        avg_value = data[avg_col].mean()
        if not np.isnan(avg_value):
            prices.append(value)
            avg_demand.append(avg_value)

# Create DF and drop N/A values
df = pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()

# Calculate elasticity of demand
df['pct_change_demand'] = df['avg_demand'].pct_change()
df['pct_change_price'] = df['prices'].pct_change()
df['elasticity'] = df['pct_change_demand'] / df['pct_change_price']

# Set up linear regression model
x = df['avg_demand'].values.reshape(-1, 1)
y = df['prices'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)

# Generate predictions for visualization
extended_demand = np.linspace(0, max(x), 100).reshape(-1, 1)
predicted_prices = model.predict(extended_demand)

# Visualization
plt.figure(figsize=(10, 8))

# Plot the demand curve
plt.plot(x, y, 'bo', label='Avg. Demand at Price Point')  # Actual data points
plt.plot(extended_demand, predicted_prices, 'black', label='Demand Curve')

# Annotate elasticity values at each price point
for i, row in df.iterrows():
    plt.annotate(f'ε={row["elasticity"]:.2f}',
                 (row['avg_demand'], row['prices']),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 fontsize=9,
                 color='darkgreen')


plt.title('Demand Curve with Elasticity for Berkeley AC Transit Buses')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Price ($ per Ride)')
plt.ylim(bottom=-0.5)
plt.legend()
plt.grid(True)
plt.show()
