import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# ================================
# 1. Data Import and Preparation
# ================================

# Load survey data here
data = pd.read_csv('Survey Data/final_bersurvey.csv')

# Define price points
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Compute average demand values and total data points per price point
prices = []
avg_demand = []
n_observations = []
for price, price_val in price_points.items():
    avg_col = f'avg_demand_{price}'
    n_col = f'n_observations_{price}'
    if avg_col in data.columns and n_col in data.columns:
        avg_val = data[avg_col].mean()
        n_val = data[n_col].sum()
        if not np.isnan(avg_val) and n_val > 0:
            prices.append(price_val)
            avg_demand.append(avg_val)
            n_observations.append(n_val)

# Create a DataFrame
df = pd.DataFrame({
    'avg_demand': avg_demand,
    'prices': prices,
    'n_observations': n_observations
}).dropna()

# =======================================
# 2. Fit the Quadratic Demand Specification
# =======================================

# The inverse demand function is: P(Q) = beta0 + beta1*Q + beta2*Q^2.
# Build regressors: Q and Q^2, plus an intercept.
X = np.column_stack([df['avg_demand'], df['avg_demand']**2])
X = add_constant(X)  # Adds an intercept column named 'const'
y = df['prices']

# Fit the OLS regression to obtain coefficient estimates and their covariance.
ols_model = OLS(y, X).fit()
beta = ols_model.params
cov_params = ols_model.cov_params().values

beta0 = beta['const']
beta1 = beta.iloc[1]
beta2 = beta.iloc[2]

print("Quadratic Regression Coefficients:")
print(f"  beta0 = {beta0:.4f}")
print(f"  beta1 = {beta1:.4f}")
print(f"  beta2 = {beta2:.4f}")

# =======================================
# 3. Determine the Effective Choke Quantity (y=0 value)
# =======================================

def get_choke_quantity(b0, b1, b2, Q_upper_bound):
    """
    Returns an effective choke quantity (Q*): the quantity at which the demand curve reaches y=0
    (minimum price point):
      b0 + b1*Q + b2*Q^2 = 0
    has a positive real solution, that value is used. Otherwise, numerical optimization is used.
    """
    P = lambda Q: b0 + b1 * Q + b2 * Q**2
    disc = b1**2 - 4 * b2 * b0
    if disc < 0:
        print(f"Discriminant is negative ({disc:.4f}); using numerical optimization for choke quantity.")
        res = minimize_scalar(lambda Q: np.abs(P(Q)), bounds=(0, Q_upper_bound), method='bounded')
        Q_choke = res.x
        print(f"Effective choke quantity (optimization): {Q_choke:.4f}, with P(Q) = {P(Q_choke):.4f}")
    else:
        # If the quadratic has real roots, choose the one that is positive.
        if b2 > 0:
            Q_choke = (-b1 + np.sqrt(disc)) / (2 * b2)
        else:
            Q_choke = (-b1 - np.sqrt(disc)) / (2 * b2)

        # Trigger numerical optimization 
        if Q_choke <= 0:
            print("Calculated choke quantity is not positive; using optimization fallback.")
            res = minimize_scalar(lambda Q: np.abs(P(Q)), bounds=(0, Q_upper_bound), method='bounded')
            Q_choke = res.x
            print(f"Effective choke quantity (fallback optimization): {Q_choke:.4f}, with P(Q) = {P(Q_choke):.4f}")
    return Q_choke


# Set upper bound for Q in optimization (creates an interval)
Q_upper_bound = 10 * max(df['avg_demand'])

# =======================================
# 4. Consumer Surplus Calculations
# =======================================

# Numerical Integration Method
def consumer_surplus_integral(b0, b1, b2):
    P = lambda Q: b0 + b1 * Q + b2 * Q**2
    Q_choke = get_choke_quantity(b0, b1, b2, Q_upper_bound)
    cs, err = quad(P, 0, Q_choke)
    return cs, Q_choke

consumer_surplus_numeric, Q_star = consumer_surplus_integral(beta0, beta1, beta2)
print(f"\nNumerical Consumer Surplus Estimate (via integration): ${consumer_surplus_numeric:.2f}")
print(f"Effective Choke Quantity (Q*): {Q_star:.4f}")


# Closed-Form Consumer Surplus (Using Antiderivative) & Delta Method
def delta_method_cs(beta_vec):
    """
    Computes consumer surplus using the antiderivative:
      CS = b0*Q* + 0.5*b1*Q*^2 + (1/3)*b2*Q*^3,
    where Q* is the effective choke quantity.
    If the quadratic does not cross the x-axis (or yields a non-positive Q*), numerical optimization is used.
    """
    b0, b1, b2 = beta_vec
    Q_choke = get_choke_quantity(b0, b1, b2, Q_upper_bound)
    cs = b0 * Q_choke + 0.5 * b1 * Q_choke**2 + (1/3) * b2 * Q_choke**3
    return cs

beta_vec = np.array([beta0, beta1, beta2])
cs_exact = delta_method_cs(beta_vec)
print(f"\nExact (closed-form) Consumer Surplus: ${cs_exact:.2f}")

def numerical_gradient(func, beta_vec, h=1e-6):
    grad = np.zeros_like(beta_vec)
    for i in range(len(beta_vec)):
        beta_plus = beta_vec.copy()
        beta_minus = beta_vec.copy()
        beta_plus[i] += h
        beta_minus[i] -= h
        grad[i] = (func(beta_plus) - func(beta_minus)) / (2 * h)
    return grad

grad_cs = numerical_gradient(delta_method_cs, beta_vec)
var_cs = grad_cs.dot(cov_params).dot(grad_cs)
se_cs = np.sqrt(var_cs)
z = 1.96
ci_lower = cs_exact - z * se_cs
ci_upper = cs_exact + z * se_cs

print(f"Delta Method 95% Confidence Interval for Consumer Surplus: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# =======================================
# 5. Plotting the Demand Curve and Consumer Surplus
# =======================================

# Generate a range of quantities for plotting the fitted curve.
Q_plot = np.linspace(0, Q_star * 1.1, 200)
price_curve = beta0 + beta1 * Q_plot + beta2 * Q_plot**2

plt.figure(figsize=(10, 6))
plt.plot(Q_plot, price_curve, label="Quadratic Demand Curve", color='black')
plt.fill_between(Q_plot, 0, price_curve, where=(Q_plot <= Q_star), color='lightgreen', alpha=0.5,
                 label=f'Consumer Surplus = ${cs_exact:.2f}')
plt.axvline(Q_star, color='red', linestyle='--', label=f'Choke Quantity Q* = {Q_star:.2f}')

plt.xlabel("Quantity Demanded (Q)")
plt.ylabel("Price ($ per unit)")
plt.title("Quadratic Demand Curve with Consumer Surplus")
plt.legend()
plt.grid(True)
plt.show()



# Integral at 4, 5, and 20 cents (approx. 11 cents off)
# Find delta values and see if the difference is significant