"""
WLS fit of the STSP prior

STSP(theta, n) is the standard two-sided power distribution
(van Dorp & Kotz, 2002):

    f(x | theta, n) = n * (x / theta)^{n-1},        0 < x <= theta,
                    = n * ((1-x)/(1-theta))^{n-1},  theta <= x < 1,

with mode theta in (0, 1) and shape parameter n > 0.

The prior is fitted to the same triple of conditions as in prior_beta.py:
E[x] = mu, sd(x) = u, F(C_95) = 0.95, via weighted least squares.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares, brentq
from scipy.integrate import quad


# ----------------------------------------------------------------------
# Prior information
# ----------------------------------------------------------------------
MU_PRIOR = 0.98
U_PRIOR  = 0.015
C_95     = 0.95
P_95     = 0.95

SIGMA_MU = 0.005
SIGMA_U  = 0.005
SIGMA_C  = 0.010

Y_MEAS   = 0.97
U_MEAS   = 0.02


# ----------------------------------------------------------------------
# STSP pdf / cdf and moments
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
    """Analytic expressions from van Dorp & Kotz (2002)."""
    mean = ((n - 1.0) * theta + 1.0) / (n + 1.0)
    var  = (n - 2.0 * (n - 1.0) * theta * (1.0 - theta)) / \
           ((n + 2.0) * (n + 1.0) ** 2)
    return mean, np.sqrt(var)


# ----------------------------------------------------------------------
# WLS fit of the STSP prior
# ----------------------------------------------------------------------
def _stsp_residuals(theta_params):
    theta, n = theta_params
    if not (1e-6 < theta < 1.0 - 1e-6) or n <= 1e-6:
        return np.array([1e6, 1e6, 1e6])
    mean, sd = stsp_moments(theta, n)
    q = brentq(lambda x: stsp_cdf(x, theta, n) - P_95, 0.0, 1.0)
    return np.array([
        (mean - MU_PRIOR) / SIGMA_MU,
        (sd   - U_PRIOR)  / SIGMA_U,
        (q    - C_95)     / SIGMA_C,
    ])


def fit_stsp_prior(x0=(0.98, 3.0)):
    sol = least_squares(_stsp_residuals, x0=x0,
                        bounds=([1e-6, 1e-6], [1.0 - 1e-6, np.inf]))
    return sol.x


# ----------------------------------------------------------------------
# Posterior and coverage interval
# ----------------------------------------------------------------------
def posterior_pdf(x, theta, n, y=Y_MEAS, u=U_MEAS):
    return stsp_pdf(x, theta, n) * norm.pdf(y, loc=x, scale=u)


def posterior_moments(theta, n, y=Y_MEAS, u=U_MEAS, n_grid=4001):
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, theta, n, y, u)
    Z = np.trapezoid(p, x)
    p = p / Z
    mean = np.trapezoid(x * p, x)
    var  = np.trapezoid((x - mean) ** 2 * p, x)
    return mean, np.sqrt(var), x, p


def coverage_interval_right_at_one(theta, n, level=0.95,
                                   y=Y_MEAS, u=U_MEAS, n_grid=4001):
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, theta, n, y, u)
    p = p / np.trapezoid(p, x)

    def tail(L):
        mask = x >= L
        return np.trapezoid(p[mask], x[mask]) - level

    L = brentq(tail, 0.0, 1.0)
    return L, 1.0


if __name__ == "__main__":

    theta, n = fit_stsp_prior()
    print(f"STSP prior: theta = {theta:.4f}, n = {n:.4f}")

    x_hat, u_hat, _, _ = posterior_moments(theta, n)
    L, R = coverage_interval_right_at_one(theta, n)

    print(f"Posterior mean  x_hat    = {x_hat:.5f}")
    print(f"Posterior sd    u(x_hat) = {u_hat:.5f}")
    print(f"95% coverage interval    = [{L:.5f}, {R:.5f}]")
