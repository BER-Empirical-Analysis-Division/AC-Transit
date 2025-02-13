import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.linear_model import LinearRegression

# Load updated data with average demand columns
data = pd.read_csv('Survey Data/final_bersurvey.csv')

# Define price points and corresponding average demand column names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Get average demands and corresponding prices from the updated CSV based on filter arguments
def priceDemandByFilter(trueData, filter_col):
    filter_col_values = list(trueData[filter_col].unique())
    for val in filter_col_values:
        print(val)
    
    plt.figure(figsize=(8,6))
    color_list = list(mcolors.TABLEAU_COLORS.values())
    colors = iter(color_list)
    
    for filter_col_value in filter_col_values:
        data = trueData[trueData[filter_col] == filter_col_value]
        prices = []
        avg_demand = []
        for price, value in price_points.items():
            avg_col = f'avg_demand_{price}'
            if avg_col in data.columns:
                avg_value = data[avg_col].mean()
                if not np.isnan(avg_value):
                    prices.append(value)
                    avg_demand.append(avg_value)
                    
        # Remove N/A values
        df = pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()


        # Set up linear regression model
        x = df['avg_demand'].values.reshape(-1,1)
        y = df['prices'].values.reshape(-1,1)
        model = LinearRegression()
        model.fit(x, y)

        # Generate predictions
        extended_demand = np.linspace(0, max(x), 100).reshape(-1,1)
        predicted_prices = model.predict(extended_demand)

        # Demand Curve Visualization
        filter_value_color = next(colors)
        plt.plot(x, y, 'bo', label=f'Avg. Demand at Price Point for {filter_col_value}', color = filter_value_color)  # Actual data points
        plt.plot(extended_demand, predicted_prices, label=f'Demand Curve for {filter_col_value}', color = filter_value_color)

    plt.title(f'Demand Curve for Berkeley AC Transit Buses by {filter_col}')
    plt.xlabel('Quantity Demanded (Rides per Week)')
    plt.ylabel('Price ($ per Ride)')
    plt.ylim(bottom=-0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
    
priceDemandByFilter(data, 'Year')
priceDemandByFilter(data, 'Transfer')
priceDemandByFilter(data, 'Gender')
priceDemandByFilter(data, 'Housing')
priceDemandByFilter(data, 'Accessibility')
priceDemandByFilter(data, 'Days on Campus ')
priceDemandByFilter(data, 'Bus Pass Impact')
priceDemandByFilter(data, 'Class Pass Fee Know ')
priceDemandByFilter(data, 'Support Class Pass ')
priceDemandByFilter(data, 'Clipper Card Value ')

