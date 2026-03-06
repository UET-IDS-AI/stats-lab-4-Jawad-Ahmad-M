"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation

    RETURN

        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
    """

    lambda_ = 1

    p_greater_5 = math.exp(-lambda_ * 5)
    p_less_5 = 1 - math.exp(-lambda_ * 5)
    _3_greater_p_less_7 = math.exp(-lambda_ * 3) - math.exp(-lambda_ * 7)

    samples = np.random.exponential(scale = 1.0, size = 100000)

    p_greater_5_simulated = np.sum(samples > 5) / len(samples)

    return p_greater_5, p_less_5, _3_greater_p_less_7, p_greater_5_simulated


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0

    STEP 1
    Verify non-negativity

    STEP 2
    Compute

        integral_0^∞ f(x) dx

    STEP 3
    Determine if valid PDF

    STEP 4
    Plot f(x) on [0,3]

    RETURN

        integral_value
        is_valid_pdf
    """
    def f(x):
        return 2 * x * math.exp(-x**2)

    integral_value, error = quad(f, 0, np.inf)
    is_valid_pdf = bool(np.isclose(integral_value, 1, atol=1e-6))

    x = np.linspace(0, 3, 300)       # 300 points between 0 and 3
    y = 2 * x * np.exp(-x**2)        # evaluate f(x) at each point
    
    plt.plot(x, y, color='blue', label=r'$f(x) = 2x\,e^{-x^2}$')
    plt.fill_between(x, y, alpha=0.2)  # shade under the curve
    plt.title("Candidate PDF: f(x) = 2x·e^(-x²)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    return integral_value, is_valid_pdf

# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)

    STEP 1
    Compute analytically

        P(X > 5)
        P(1 < X < 3)

    STEP 2
    Simulate 100000 samples

    STEP 3
    Estimate probabilities using simulation

    RETURN

        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """
    lambda_ = 1

    p_greater_5 = math.exp(-lambda_ * 5)
    _1_greater_x_less_3 =  math.exp(-lambda_ * 1) -  math.exp(-lambda_ * 3)

    samples = np.random.exponential(scale = 1.0, size = 100000)

    simulated_gt5 = np.sum(samples > 5) / len(samples)
    simulated_interval = np.sum((samples > 1) & (samples < 3)) / len(samples)

    return p_greater_5, _1_greater_x_less_3, simulated_gt5, simulated_interval

# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

    # Analytical probabilities using scipy
    analytic_le12 = norm.cdf(12, loc=10, scale=2)

    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # Simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    simulated_le12 = np.mean(samples <= 12)

    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
