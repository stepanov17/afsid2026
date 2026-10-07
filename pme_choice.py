"""
Selection between two prior families (Beta and STSP) by residual misfit
and, when the misfits are close, by the maximum entropy principle (PME).

Both priors are fitted to the same triple of conditions:
    E[x] = mu, sd(x) = u, F(C_95) = 0.95.
The residual WLS cost is reported for each family. If the two costs are
within a relative tolerance, the family with the larger differential
entropy of the prior is preferred, since it encodes less additional
information beyond the imposed constraints.

Requires prior_beta.py and prior_stsp.py in the same directory.
"""

import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import brentq

from prior_beta import (
    fit_beta_prior,
    posterior_moments as beta_post_moments,
    coverage_interval_right_at_one as beta_ci,
    SIGMA_MU, SIGMA_U, SIGMA_C,
    MU_PRIOR, U_PRIOR, C_95, P_95,
    Y_MEAS, U_MEAS,
)
from prior_stsp import (
    fit_stsp_prior,
    stsp_pdf, stsp_cdf, stsp_moments,
    posterior_moments as stsp_post_moments,
    coverage_interval_right_at_one as stsp_ci,
)


# Tolerance for treating two residual costs as "close".
# If the relative difference is below this threshold, PME decides.
REL_TOL = 0.15


# ----------------------------------------------------------------------
# Residual WLS costs for each family
# ----------------------------------------------------------------------
def beta_residual_cost(a, b):
    mean = a / (a + b)
    sd   = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
    q    = beta.ppf(P_95, a, b)
    r = np.array([
        (mean - MU_PRIOR) / SIGMA_MU,
        (sd   - U_PRIOR)  / SIGMA_U,
        (q    - C_95)     / SIGMA_C,
    ])
    return float(r @ r)


def stsp_residual_cost(theta, n):
    mean, sd = stsp_moments(theta, n)
    q = brentq(lambda x: stsp_cdf(x, theta, n) - P_95, 0.0, 1.0)
    r = np.array([
        (mean - MU_PRIOR) / SIGMA_MU,
        (sd   - U_PRIOR)  / SIGMA_U,
        (q    - C_95)     / SIGMA_C,
    ])
    return float(r @ r)


# ----------------------------------------------------------------------
# Differential entropy of each prior
# ----------------------------------------------------------------------
def entropy_beta(a, b, n_grid=4001):
    """
    Differential entropy of Beta(a, b) on [0, 1].
    Zero-density grid points are excluded: their contribution to
    p * log(p) is zero by the limit p * log(p) -> 0 as p -> 0+.
    """
    x = np.linspace(1e-6, 1.0 - 1e-6, n_grid)
    p = beta.pdf(x, a, b)
    p = p / np.trapezoid(p, x)
    mask = p > 0.0
    return -np.trapezoid(p[mask] * np.log(p[mask]), x[mask])


def entropy_stsp(theta, n, n_grid=4001):
    """
    Differential entropy of STSP(theta, n) on [0, 1].
    Same masking as in entropy_beta.
    """
    x = np.linspace(1e-6, 1.0 - 1e-6, n_grid)
    p = stsp_pdf(x, theta, n)
    p = p / np.trapezoid(p, x)
    mask = p > 0.0
    return -np.trapezoid(p[mask] * np.log(p[mask]), x[mask])


if __name__ == "__main__":

    # --- fit both families ---
    a, b = fit_beta_prior()
    theta, n = fit_stsp_prior()

    cost_beta = beta_residual_cost(a, b)
    cost_stsp = stsp_residual_cost(theta, n)

    print(f"Beta  fit: alpha = {a:.4f}, beta = {b:.4f}, "
          f"residual cost = {cost_beta:.4e}")
    print(f"STSP  fit: theta = {theta:.4f}, n = {n:.4f}, "
          f"residual cost = {cost_stsp:.4e}")

    # --- selection rule ---
    rel_diff = abs(cost_beta - cost_stsp) / max(cost_beta, cost_stsp, 1e-12)
    if rel_diff > REL_TOL:
        chosen = "Beta" if cost_beta < cost_stsp else "STSP"
        print(f"Residual costs differ by more than {REL_TOL:.0%}; "
              f"chosen family: {chosen}")
    else:
        H_beta = entropy_beta(a, b)
        H_stsp = entropy_stsp(theta, n)
        print("Residual costs are close; PME applied.")
        print(f"  H[Beta prior] = {H_beta:.5f}")
        print(f"  H[STSP prior] = {H_stsp:.5f}")
        chosen = "Beta" if H_beta > H_stsp else "STSP"
        print(f"Chosen family by PME: {chosen}")

    # --- final posterior summary for the chosen family ---
    if chosen == "Beta":
        x_hat, u_hat, _, _ = beta_post_moments(a, b)
        L, R = beta_ci(a, b)
    else:
        x_hat, u_hat, _, _ = stsp_post_moments(theta, n)
        L, R = stsp_ci(theta, n)

    print(f"Posterior mean  x_hat    = {x_hat:.5f}")
    print(f"Posterior sd    u(x_hat) = {u_hat:.5f}")
    print(f"95% coverage interval    = [{L:.5f}, {R:.5f}]")
