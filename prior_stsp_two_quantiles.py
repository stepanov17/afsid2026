"""
Fit STSP prior: two quantile constraints.

The measurand x lies in [0, 1] with a prior belief concentrated near
the upper bound. The prior family is the standard two-sided power
distribution (van Dorp & Kotz, 2002):

    f(x | theta, n) = n * (x / theta)^{n-1},          0 < x <= theta,
                    = n * ((1-x)/(1-theta))^{n-1},    theta <= x < 1,

where theta in (0, 1) is the mode and n > 0 is the shape parameter.

The prior is fitted to four conditions:
    E[x]    = mu           (nominal value)
    sd(x)   = u            (standard uncertainty of the prior)
    F(C_1)  = P_1          (first quantile constraint, lower level)
    F(C_2)  = P_2          (second quantile constraint, higher level)

Four conditions for two parameters -> overdetermined system,
solved by weighted least squares.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares, brentq


# ----------------------------------------------------------------------
# Prior information
# ----------------------------------------------------------------------
MU_PRIOR = 0.98       # E[x]
U_PRIOR  = 0.015      # sd(x)

C1, P1 = 0.95, 0.90   # F(C1) = P1  (lower quantile)
C2, P2 = 0.97, 0.99   # F(C2) = P2  (upper quantile)

# Weights: w_i ~ 1 / sigma_i^2. Here sigma_i are the assumed tolerances
# on the corresponding conditions.
SIGMA_MU = 0.005
SIGMA_U  = 0.005
SIGMA_Q1 = 0.010
SIGMA_Q2 = 0.005

# Measurement model
Y_MEAS = 0.97
U_MEAS = 0.02


# ----------------------------------------------------------------------
# STSP pdf / cdf / moments
# ----------------------------------------------------------------------
def stsp_pdf(x, theta, n):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    left  = (x > 0.0) & (x <= theta)
    right = (x >= theta) & (x < 1.0)
    out[left]  = n * (x[left]  / theta) ** (n - 1.0)
    out[right] = n * ((1.0 - x[right]) / (1.0 - theta)) ** (n - 1.0)
    return out


def stsp_cdf(x, theta, n):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x <= theta:
        return theta * (x / theta) ** n
    return 1.0 - (1.0 - theta) * ((1.0 - x) / (1.0 - theta)) ** n


def stsp_moments(theta, n):
    """Analytic mean and standard deviation (van Dorp & Kotz, 2002)."""
    mean = ((n - 1.0) * theta + 1.0) / (n + 1.0)
    var  = (n - 2.0 * (n - 1.0) * theta * (1.0 - theta)) / \
           ((n + 2.0) * (n + 1.0) ** 2)
    return mean, np.sqrt(var)


def stsp_quantile(p, theta, n):
    """Solve F(x) = p for x in [0, 1]."""
    return brentq(lambda x: stsp_cdf(x, theta, n) - p, 0.0, 1.0)


# ----------------------------------------------------------------------
# WLS fit of the STSP prior with two quantile constraints
# ----------------------------------------------------------------------
def _stsp_residuals(params):
    theta, n = params
    if not (1e-6 < theta < 1.0 - 1e-6) or n <= 1e-6:
        return np.array([1e6, 1e6, 1e6, 1e6])
    mean, sd = stsp_moments(theta, n)
    q1 = stsp_quantile(P1, theta, n)
    q2 = stsp_quantile(P2, theta, n)
    return np.array([
        (mean - MU_PRIOR) / SIGMA_MU,
        (sd   - U_PRIOR)  / SIGMA_U,
        (q1   - C1)       / SIGMA_Q1,
        (q2   - C2)       / SIGMA_Q2,
    ])


def fit_stsp_prior(x0=(0.97, 5.0)):
    sol = least_squares(_stsp_residuals, x0=x0,
                        bounds=([1e-6, 1e-6], [1.0 - 1e-6, np.inf]))
    return sol.x, sol.cost


# ----------------------------------------------------------------------
# Posterior and coverage interval
# ----------------------------------------------------------------------
def posterior_pdf(x, theta, n, y=Y_MEAS, u=U_MEAS):
    return stsp_pdf(x, theta, n) * norm.pdf(y, loc=x, scale=u)


def posterior_moments(theta, n, y=Y_MEAS, u=U_MEAS, n_grid=4001):
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, theta, n, y, u)
    p = p / np.trapezoid(p, x)
    mean = np.trapezoid(x * p, x)
    var  = np.trapezoid((x - mean) ** 2 * p, x)
    return mean, np.sqrt(var), x, p


def coverage_interval_right_at_one(theta, n, level=0.95,
                                   y=Y_MEAS, u=U_MEAS, n_grid=4001):
    """
    Left end L such that P(x in [L, 1] | y) = level.
    The right end is fixed at the physical bound x = 1.
    """
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, theta, n, y, u)
    p = p / np.trapezoid(p, x)

    def tail(L):
        mask = x >= L
        return np.trapezoid(p[mask], x[mask]) - level

    L = brentq(tail, 0.0, 1.0)
    return L, 1.0


if __name__ == "__main__":
    (theta, n), cost = fit_stsp_prior()
    print(f"STSP prior: theta = {theta:.4f}, n = {n:.4f}")
    print(f"Residual WLS cost  = {cost:.4e}")

    mean_fit, sd_fit = stsp_moments(theta, n)
    q1_fit = stsp_quantile(P1, theta, n)
    q2_fit = stsp_quantile(P2, theta, n)
    print(f"E[x]   : fit {mean_fit:.5f}  target {MU_PRIOR:.5f}")
    print(f"sd(x)  : fit {sd_fit:.5f}  target {U_PRIOR:.5f}")
    print(f"F(C1)  : C1_fit {q1_fit:.5f}  target {C1:.5f}  (P1 = {P1})")
    print(f"F(C2)  : C2_fit {q2_fit:.5f}  target {C2:.5f}  (P2 = {P2})")

    x_hat, u_hat, _, _ = posterior_moments(theta, n)
    L, R = coverage_interval_right_at_one(theta, n)

    print(f"\nPosterior mean  x_hat    = {x_hat:.5f}")
    print(f"Posterior sd    u(x_hat) = {u_hat:.5f}")
    print(f"95% coverage interval    = [{L:.5f}, {R:.5f}]")
