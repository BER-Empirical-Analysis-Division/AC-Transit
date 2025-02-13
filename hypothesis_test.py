import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.utils import resample

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
n_boots = 1000  # Number of bootstrap samples
bootstrap_results = {}

# Perform bootstrap sampling for each price point to estimate population mean
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns:
        avg_demand_data = data[avg_col].dropna()
        
        # Perform bootstrap resampling
        bootstrap_means = []
        for _ in range(n_boots):
            resample_data = resample(avg_demand_data, replace=True)
            bootstrap_means.append(resample_data.mean())
        
        # Calculate bootstrap population mean
        population_mean = np.mean(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, 2.5)  # Lower bound of 95% CI
        ci_upper = np.percentile(bootstrap_means, 97.5)  # Upper bound of 95% CI
        
        # Save results for the price point
        bootstrap_results[price] = {
            'population_mean': population_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        # Print bootstrap results
        print(f"\nPrice Point: ${value:.2f}")
        print(f"Bootstrap Population Mean: {population_mean:.2f}")
        print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Hypothesis testing for each price point
print("\nHypothesis Testing: Comparing Survey Data to Bootstrap Population Means")
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns and price in bootstrap_results:
        # Use the observed data for this price point
        observed_data = data[avg_col].dropna()
        
        # Get the bootstrap population mean for this price point
        population_mean = bootstrap_results[price]['population_mean']
        
        # Perform one-sample t-test
        t_stat, p_value = ttest_1samp(observed_data, population_mean)
        
        # Print hypothesis test results
        print(f"\nPrice Point: ${value:.2f}")
        print(f"Observed Mean: {observed_data.mean():.2f}")
        print(f"Bootstrap Population Mean: {population_mean:.2f}")
        print(f"t-Statistic: {t_stat:.4f}, p-Value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject the null hypothesis. The survey data differs significantly from the bootstrap mean.")
        else:
            print("Result: Accept the null hypothesis. The survey data does not differ significantly from the bootstrap mean.")
