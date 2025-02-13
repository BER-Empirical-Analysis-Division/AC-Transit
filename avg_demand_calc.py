import pandas as pd
import numpy as np

# Load survey
data = pd.read_csv('Survey Data/bersurvey.csv')

# Filter out respondents who didn't pass the Attention Check 
data = data[data['Q136'] == '2']

# Define price points
price_points = {
    'free': {'value': 0, 'cols': []},
    'half': {'value': 1.13, 'cols': []},
    'normal': {'value': 2.25, 'cols': []},
    'extra': {'value': 3.37, 'cols': []},
    'double': {'value': 5.50, 'cols': []}
}

# Sort columns by price point
for col in data.columns:
    if 'conf' not in col:
        for key in price_points.keys():
            if col.startswith(f'{key}_'):
                price_points[key]['cols'].append(col)

# Add average demand columns and number of observations for each price point
for price, info in price_points.items():
    data[info['cols']] = data[info['cols']].apply(pd.to_numeric, errors='coerce')
    data[f'avg_demand_{price}'] = data[info['cols']].mean(axis=1)
    data[f'n_observations_{price}'] = data[info['cols']].notnull().sum(axis=1)

# Save updated data to a new survey
data.to_csv('Survey Data/final_bersurvey.csv', index=False)
print(f"CSV updated with average demand columns and observation counts for each price point.")
