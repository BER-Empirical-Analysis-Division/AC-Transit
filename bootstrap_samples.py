import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Survey Data/final_bersurvey.csv')

# Define price points and corresponding average demand column names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Bootstrap parameters
n_iterations = 1000  # Number of bootstrap samples

# Store results for each price point
bootstrap_results = {}

# Perform bootstrap sampling for each price point
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns:
        # Filter non-NaN values for the price point
        avg_demand_data = data[avg_col].dropna()
        
        # Perform bootstrap resampling
        bootstrap_means = []
        for _ in range(n_iterations):
            resample_data = resample(avg_demand_data, replace=True)
            bootstrap_means.append(resample_data.mean())
        
        # Calculate bootstrap confidence intervals
        ci_lower = np.percentile(bootstrap_means, 2.5)  # Lower bound of 95% CI
        ci_upper = np.percentile(bootstrap_means, 97.5)  # Upper bound of 95% CI
        mean_estimate = np.mean(bootstrap_means)
        
        # Save results for the price point
        bootstrap_results[price] = {
            'mean': mean_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_means': bootstrap_means
        }
        
        # Print results for the price point
        print(f"\nPrice Point: ${value:.2f}")
        print(f"Estimated Mean: {mean_estimate:.2f}")
        print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Visualize bootstrap distributions for each price point
for price, value in price_points.items():
    if price in bootstrap_results:
        results = bootstrap_results[price]
        plt.figure(figsize=(10, 6))
        plt.hist(results['bootstrap_means'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(results['ci_lower'], color='red', linestyle='--', label=f'Lower CI: {results["ci_lower"]:.2f}')
        plt.axvline(results['ci_upper'], color='green', linestyle='--', label=f'Upper CI: {results["ci_upper"]:.2f}')
        plt.axvline(results['mean'], color='blue', linestyle='-', label=f'Bootstrap Mean: {results["mean"]:.2f}')
        plt.title(f'Bootstrap Distribution of Sample Means (Price Point: ${value:.2f})')
        plt.xlabel('Mean Quantity Demanded')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
