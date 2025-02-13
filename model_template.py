import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data(csv_path):
    """
    Load the CSV file that contains the survey data with average demand columns.
    """
    data = data.columns.str.strip()
    return data

def extract_avg_demand(data, conditions, group_by_column=None, group_value=None, price_column_prefix='avg_demand_'):
    """
    Extract average demand and corresponding price values based on specific conditions.
    - data: the DataFrame with the survey data
    - conditions: a dictionary mapping condition labels (e.g., 'free', 'half') to price values
    - group_by_column: column name to group/filter data by (optional)
    - group_value: the value of the group_by_column to filter the data (optional)
    - price_column_prefix: the prefix used for average demand columns in the CSV
    """
    # Apply filtering based on group_by_column and group_value
    if group_by_column and group_value is not None:
        data = data[data[group_by_column] == group_value]

    prices = []
    avg_demand = []
    for condition, price in conditions.items():
        avg_col = f'{price_column_prefix}{condition}'  # Build the column name
        if avg_col in data.columns:
            avg_value = data[avg_col].mean()  # Average of the selected condition column
            if not np.isnan(avg_value):
                prices.append(price)
                avg_demand.append(avg_value)
    return pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()

def perform_linear_regression(df):
    """
    Perform linear regression on the average demand vs. price data.
    - df: DataFrame containing 'avg_demand' and 'prices'
    """
    x = df['avg_demand'].values.reshape(-1, 1)
    y = df['prices'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model

def plot_demand_curve(model, df, group_label=None):
    """
    Plot the demand curve using the fitted linear regression model.
    - model: the trained LinearRegression model
    - df: DataFrame containing 'avg_demand' and 'prices'
    - group_label: an optional label to include in the plot title if a group filter was applied
    """
    extended_demand = np.linspace(0, max(df['avg_demand']), 100).reshape(-1, 1)  # Start from 0
    predicted_prices = model.predict(extended_demand)

    # Plot demand curve
    plt.figure(figsize=(8, 6))
    plt.plot(df['avg_demand'], df['prices'], 'bo', label='Avg. Demand at Price Point')  # Actual data points
    plt.plot(extended_demand, predicted_prices, 'black', label='Demand Curve')

    title = 'Demand Curve'
    if group_label:
        title += f' for {group_label}'

    plt.title(title)
    plt.xlabel('Quantity Demanded (Rides per Week)')
    plt.ylabel('Price ($ per Ride)')
    plt.ylim(bottom=-0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_demand_curve_analysis(csv_path, conditions, group_by_column=None, group_value=None):
    """
    Run the full analysis pipeline: load data, extract values, perform regression, and plot the demand curve.
    - csv_path: the path to the CSV file with the survey data
    - conditions: dictionary mapping labels to price values
    - group_by_column: column name to group/filter data by (optional)
    - group_value: the value of the group_by_column to filter the data (optional)
    """
    data = load_data(csv_path)  # Load the CSV data
    df = extract_avg_demand(data, conditions, group_by_column, group_value)  # Extract avg_demand and price info
    model = perform_linear_regression(df)  # Perform regression

    # Create a label for the group if filtering was applied
    group_label = f"{group_by_column} = {group_value}" if group_by_column and group_value is not None else None
    plot_demand_curve(model, df, group_label)  # Plot the demand curve



# USAGE

# Define the conditions (e.g., different price points)
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}


# ****MAKE ALL YOUR CHANGES HERE****:

# Run the analysis with a specific grouping condition (e.g., filtering by 'class' column)
run_demand_curve_analysis(
    'Survey Data/v2survey_11-06-24.csv', # this is the path for the CSV here, change if needed
    price_points, 
    group_by_column = 'Year',  # Change this to the column you want to filter by
    group_value = 'Senior (4th Year)'  # The specific value in the column to filter by
)