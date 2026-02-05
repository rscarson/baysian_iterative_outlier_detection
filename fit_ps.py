import numpy as np
from numpy.random import multivariate_normal

import pandas as pd

from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.optimize import OptimizeWarning

import matplotlib.pyplot as plt

import warnings
import sys

def hill_equation_derivative(x, n, k, ymin=0, ymax=1):
    """
    2nd derivative of Hill equation:
    y = y_{\\min} + (y_{\\max} - y_{\\min}) \\frac{x^n}{k^n + x^n}
    A = y_{\\max} - y_{\\min}, y = y_{\\min} + A \\frac{x^n}{k^n + x^n}
    d/dx(x^n/(k^n + x^n)) = (n k^n x^(n - 1))/(k^n + x^n)^2

    So the derivative is:
    A (n k^n x^(n - 1))/(k^n + x^n)^2
    """
    n = np.clip(n, 0, 1e10)
    A = ymax - ymin
    denom = np.power(np.power(k, n) + np.power(x, n), 2) # (k^n + x^n)^2
    num = n * np.power(k, n) * np.power(x, n - 1)
    return A * num / denom

def hill_equation_2nd_derivative(x, n, k, ymin=0, ymax=1):
    """
    2nd derivative of Hill equation:
    y = y_{\\min} + (y_{\\max} - y_{\\min}) \\frac{x^n}{k^n + x^n}
    A = y_{\\max} - y_{\\min}, y = y_{\\min} + A \\frac{x^n}{k^n + x^n}
    d/dx(x^n/(k^n + x^n)) = (n k^n x^(n - 1))/(k^n + x^n)^2
    d/dx((n k^n x^(n - 1))/(k^n + x^n)^2) = (n k^n x^(n - 2) ((n - 1) k^n - (n + 1) x^n))/(k^n + x^n)^3

    So the 2nd derivative is:
    A (n k^n x^(n - 2) ((n - 1) k^n - (n + 1) x^n))/(k^n + x^n)^3
    """
    n = np.clip(n, 0, 1e10)
    A = ymax - ymin
    denom = np.power(np.power(k, n) + np.power(x, n), 3) # (k^n + x^n)^3
    num_1 = n * np.power(k, n) * np.power(x, n - 2)
    num_2 = (n - 1) * np.power(k, n) - (n + 1) * np.power(x, n)
    return A * num_1 * num_2 / denom    

def hill_equation(x, n, k, ymin=0, ymax=1):
    """
    Standard Hill equation.
    y = y_{\\min} + (y_{\\max} - y_{\\min}) \\frac{x^n}{k^n + x^n}

    :param x: Independent variable
    :param n: Hill coefficient
    :param k: Half-maximal effective concentration
    :param ymin: Minimum value
    :param ymax: Maximum value

    :return: Dependent variable
    """
    n = np.clip(n, 0, 1e10)
    x_n, k_n = np.power(x, n), np.power(k, n)
    return ymin + (ymax - ymin) * (x_n / (k_n + x_n))

def hill_from_fit(x, fit):
    """
    Evaluate Hill equation from fit dict.
    """
    return hill_equation(x, fit['n'], fit['k'], fit.get('ymin', 0), fit.get('ymax', 1))

def hill_fit(X, Y, stdev=None):
    """
    Fit Hill equation to data.

    y bounds are fixed from data.

    Returns a dict with keys 'n', 'k', 'cov' for fitted parameters.
    """
    weights = None
    if stdev is not None: 
        weights = 1 / (stdev + 1e-8)  # avoid division by zero

    ylo, yhi = min(Y), max(Y)
    mid_idx = np.argmin(np.abs(Y - (ylo + yhi) / 2))
    p0 = [1.0, X[mid_idx]]

    bounds = ([0, 0], [10, np.inf])

    if Y[0] > Y[-1]:
        # Descending curve - swap ymin and ymax
        ylo, yhi = yhi, ylo

    def model(x, n, k):
        return hill_equation(x, n, k, ymin=ylo, ymax=yhi)

    params, cov = curve_fit(
        model, X, Y,
        p0=p0, bounds=bounds,
        sigma=weights, absolute_sigma=True
    )

    n, k = params
    return {'n': n, 'k': k, 'ymin': ylo, 'ymax': yhi, 'cov': cov}

def hill_pareto_model(p_s, accuracy_fit, compute_fit, balance_factor):
    """
    Model for objective minimization
    It uses accuracy_hill - b * compute_hill
    where b is a balance factor
    """
    d_accuracy = hill_from_fit(p_s, accuracy_fit)
    d_compute = hill_from_fit(p_s, compute_fit)
    return d_accuracy - balance_factor * d_compute

#######################################################
# First we get the data with error bars from the experiments
data_src = pd.read_csv("results/bias_experiment2_3_errors.csv")

#######################################################
# Experiment 2 - Fit Hill to bias stdev vs p_s
X, Y = data_src["p_s"].values, data_src["mean_stdev_bias"].values

# We normalize the data such that they are all fractions of the maximum observed stdev
maxy = max(Y)
Y = Y / maxy

X2, Y2 = X, Y  # Save for plotting later
stdev2 = data_src["stdev_stdev_bias"].values

# Perform curve fitting
fit2 =  (X, Y, stdev2)
fit2 = hill_fit(X, Y, stdev=stdev2)
print(f"Fitted parameters for experiment 2 (bias): n={fit2['n']:.2e}, k={fit2['k']:.2e}, ymin={fit2['ymin']:.2}, ymax={fit2['ymax']:.2}")

# get R^2 value
Y_fit = hill_from_fit(X, fit2)
ss_res = np.sum((Y - Y_fit) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R^2 for experiment 2 fit: {r_squared:.4f}")

####################################################
# Experiment 3 - Fit Hill to mean iterations vs p_s
X, Y = data_src["p_s"].values, data_src["mean_mean_iterations"].values
stdev = data_src["stdev_mean_iterations"].values

X3, Y3, stdev3 = X, Y, stdev  # Save for plotting later

# Perform curve fitting
fit3 = hill_fit(X, Y, stdev=stdev)
print(f"Fitted parameters for experiment 3 (compute): n={fit3['n']:.2e}, k={fit3['k']:.2e}, ymin={fit3['ymin']:.2}, ymax={fit3['ymax']:.2}")

# get R^2 value
Y_fit = hill_from_fit(X, fit3)
ss_res = np.sum((Y - Y_fit) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R^2 for experiment 3 fit: {r_squared:.4f}")

########
# Find a balance point between the two curves
# We will minimize hill_pareto_model (sort of)

balances = np.logspace(-5, 1, 200)
results = []

for b in balances:
    try:
        sol = root_scalar(
            hill_pareto_model,
            args=(fit2, fit3, b),
            bracket=(1e3, 1e7),
            method='brentq'
        )
        if sol.converged:
            ps = sol.root
            acc = hill_from_fit(ps, fit2)
            comp = hill_from_fit(ps, fit3)
            results.append((b, ps, acc, comp))
    except ValueError:
        continue  # skip invalid brackets

# Transform results to be worst_compute - compute, bias - best_bias
best_bias = min(Y2)
prev_comp = None
prev_bias = None
transformed_results = []
for r in results:
    b, ps, acc, comp = r
    acc = acc - best_bias

    delta_bias = None if prev_bias is None else acc - prev_bias
    delta_comp = None if prev_comp is None else -(comp - prev_comp)

    prev_bias = acc
    prev_comp = comp

    # We will propagate error with the delta method
    # So, the error on p_s is ~ sqrt( err_acc^2 + (b * err_comp)^2 ) / |d/dp_s (acc - b * comp)|
    d_acc = hill_equation_derivative(ps, fit2['n'], fit2['k'], fit2['ymin'], fit2['ymax'])
    d_comp = hill_equation_derivative(ps, fit3['n'], fit3['k'], fit3['ymin'], fit3['ymax'])
    err_acc = stdev2[np.argmin(np.abs(X2 - ps))]
    err_comp = stdev3[np.argmin(np.abs(X3 - ps))]
    d_p_s = np.abs(d_acc - b * d_comp) + 1e-8  # avoid division by zero
    err_p_s = np.sqrt(err_acc**2 + (b * err_comp)**2) / d_p_s

    transformed_results.append((b, ps, comp, acc, delta_comp, delta_bias, err_p_s))
    
# Print results table
print("\nBalance Factor | P_s at Balance | Compute | Bias  | Compute Delta | Bias Delta | P_s Error")
for r in transformed_results:
    b, ps, comp, acc, delta_comp, delta_bias, err_p_s = r

    delta_comp = '-----' if delta_comp is None else f"{delta_comp:.3f}"
    delta_bias = '-----' if delta_bias is None else f"{delta_bias:.3f}"

    print(f"{b:.2e}       | {ps:.4e}     | {comp:.3f}   | {acc:.3f} | {delta_comp}         | {delta_bias}      | {err_p_s:.2e}")

# Find the root 

###################
# Plot both curves
# If user pased --show-plots, show plots
if "--show-plots" in sys.argv:
    P_fit = np.logspace(np.log10(min(X2.min(), X3.min())), np.log10(max(X2.max(), X3.max())), 500)

    plt.figure(figsize=(8,5))
    # Raw data
    plt.scatter(X2, Y2, color='blue', label='Stdev of bias (normalized)')
    plt.errorbar(X3, Y3, yerr=stdev3, fmt='o', color='red', label='Mean iterations')

    # Fitted curves
    plt.plot(P_fit, hill_from_fit(P_fit, fit2), color='blue', lw=2, label='Hill fit (stdev)')
    plt.plot(P_fit, hill_from_fit(P_fit, fit3), color='red', lw=2, label='Hill fit (mean iterations)')

    # Balance point as a vertical line
   # plt.axvline(res.root, color='green', ls='--', lw=2, label=f'Balance point (p_s={res.root:.4f})')
   # plt.axvline(res2.root, color='purple', ls='--', lw=2, label=f'Balance point (p_s={res2.root:.4f})')

    plt.xscale('log')
    plt.xlabel("P_s")
    plt.ylabel("Fraction / Standard deviation")
    plt.legend()
    plt.title("Performance vs Bias Trade-off")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()

###########################
# Export results for latex

# Define halflog points: 10^0, 10^0.5, 10^1, ..., up to max P_s
powers = np.arange(0, np.log10(max(X2.max(), X3.max())) + 0.5, 0.5)
P_halog = 10**powers

# Compute fractions for both curves
perf_fraction = hill_from_fit(P_halog, fit3)
bias_fraction = hill_from_fit(P_halog, fit2)

# Export Performance
df_perf = pd.DataFrame({
    "p_s": P_halog,
    "fraction": perf_fraction,
    "error": np.interp(P_halog, X3, stdev3)
})
df_perf.to_csv("results/bias_experiment4_perf.csv", index=False)

# Export Bias Variance
df_bias = pd.DataFrame({
    "p_s": P_halog,
    "fraction": bias_fraction,
    "error": np.interp(P_halog, X2, stdev2)
})
df_bias.to_csv("results/bias_experiment4_bias.csv", index=False)

print("Exported results/bias_experiment4_perf.csv and results/bias_experiment4_bias.csv")

# We also want a table of [balancepoint, p_s, compute, bias, delta, delta] for the two balance points and extreme case from transformed_results
df_balance = pd.DataFrame(transformed_results, columns=["balance_factor", "p_s", "compute", "bias", "compute_delta", "bias_delta", "p_s_error"])
df_balance.to_csv("results/bias_experiment4_balance_points.csv", index=False)
print("Exported results/bias_experiment4_balance_points.csv")