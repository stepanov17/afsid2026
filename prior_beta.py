"""
WLS fit of the Beta prior

The measurand x lies in [0, 1] and is expected to be close to 1.
Prior information is given as a triple:
    - nominal value   : E[x] = mu
    - standard uncertainty of the prior : sd(x) = u
    - 95% percentile  : F(C) = 0.95   (i.e. P(x <= C) = 0.95)

After fitting the prior, the measurement model
    y = x + eps,   eps ~ N(0, u_meas^2)
is used to obtain the posterior, the estimate x_hat, its standard
uncertainty u(x_hat), and the 95% coverage interval whose right end
is fixed at the physical bound x = 1.
"""

import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import least_squares, brentq


# ----------------------------------------------------------------------
# Prior information (adjust to the actual problem)
# ----------------------------------------------------------------------
MU_PRIOR = 0.98      # E[x]
U_PRIOR  = 0.015     # sd(x)
C_95     = 0.95      # F(C_95) = 0.95
P_95     = 0.95

# Weights for the WLS fit: w_i ~ 1 / sigma_i^2.
SIGMA_MU = 0.005
SIGMA_U  = 0.005
SIGMA_C  = 0.010

# Measurement model
Y_MEAS   = 0.97      # observed value
U_MEAS   = 0.02      # standard uncertainty of the measurement


# ----------------------------------------------------------------------
# WLS fit of the Beta prior
# ----------------------------------------------------------------------
def _beta_residuals(theta):
    a, b = theta
    if a <= 0 or b <= 0:
        return np.array([1e6, 1e6, 1e6])
    mean = a / (a + b)
    var  = a * b / ((a + b) ** 2 * (a + b + 1.0))
    sd   = np.sqrt(var)
    q    = beta.ppf(P_95, a, b)
    return np.array([
        (mean - MU_PRIOR) / SIGMA_MU,
        (sd   - U_PRIOR)  / SIGMA_U,
        (q    - C_95)     / SIGMA_C,
    ])


def fit_beta_prior(x0=(80.0, 2.0)):
    sol = least_squares(_beta_residuals, x0=x0,
                        bounds=([1e-6, 1e-6], [np.inf, np.inf]))
    return sol.x


# ----------------------------------------------------------------------
# Posterior and coverage interval
# ----------------------------------------------------------------------
def posterior_pdf(x, a, b, y=Y_MEAS, u=U_MEAS):
    """Unnormalized posterior; normalization is done numerically below."""
    return beta.pdf(x, a, b) * norm.pdf(y, loc=x, scale=u)


def posterior_moments(a, b, y=Y_MEAS, u=U_MEAS, n_grid=4001):
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, a, b, y, u)
    Z = np.trapezoid(p, x)
    p = p / Z
    mean = np.trapezoid(x * p, x)
    var  = np.trapezoid((x - mean) ** 2 * p, x)
    return mean, np.sqrt(var), x, p


def coverage_interval_right_at_one(a, b, level=0.95,
                                   y=Y_MEAS, u=U_MEAS, n_grid=4001):
    """
    Left end L such that P(x in [L, 1] | y) = level.
    The right end is fixed at the physical bound x = 1.
    """
    x = np.linspace(0.0, 1.0, n_grid)
    p = posterior_pdf(x, a, b, y, u)
    p = p / np.trapezoid(p, x)

    def tail(L):
        mask = x >= L
        return np.trapezoid(p[mask], x[mask]) - level

    L = brentq(tail, 0.0, 1.0)
    return L, 1.0


if __name__ == "__main__":

    a, b = fit_beta_prior()
    print(f"Beta prior: alpha = {a:.4f}, beta = {b:.4f}")

    x_hat, u_hat, _, _ = posterior_moments(a, b)
    L, R = coverage_interval_right_at_one(a, b)

    print(f"Posterior mean  x_hat  = {x_hat:.5f}")
    print(f"Posterior sd    u(x_hat) = {u_hat:.5f}")
    print(f"95% coverage interval  = [{L:.5f}, {R:.5f}]")
