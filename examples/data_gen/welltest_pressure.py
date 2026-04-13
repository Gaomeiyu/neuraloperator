"""
Wellbore Pressure Data Generator
================================

Generates dimensionless wellbore pressure (pwD) vs. dimensionless time (tD)
data for well-testing problems with wellbore storage and skin effects.

The governing PDE in radial coordinates is:

    ∂²pD/∂rD² + (1/rD)(∂pD/∂rD) = ∂pD/∂tD

with:
    - Initial condition:  pD(rD, 0) = 0
    - Outer boundary:     pD(∞, tD) = 0
    - Inner boundary:
        CD * dpwD/dtD - (∂pD/∂rD)|_{rD=1} = 1   (wellbore storage)
        pwD = [pD - S * (∂pD/∂rD)]|_{rD=1}       (skin effect)

The Laplace-domain analytical solution for the wellbore pressure is:

    p̄wD(u) = [K0(√u) + S·√u·K1(√u)]
              / [u · (√u·K1(√u) + CD·u·(K0(√u) + S·√u·K1(√u)))]

where K0 and K1 are modified Bessel functions of the second kind,
and u is the Laplace variable.

Inversion to the time domain uses the Stehfest algorithm.
"""

import math
import numpy as np
from scipy.special import kv  # modified Bessel functions K_nu


def _stehfest_weights(n_stehfest=12):
    """Compute Stehfest algorithm weights for numerical Laplace inversion.

    Parameters
    ----------
    n_stehfest : int
        Number of terms in the Stehfest expansion (must be even).

    Returns
    -------
    weights : np.ndarray of shape (n_stehfest,)
    """
    if n_stehfest % 2 != 0:
        raise ValueError("n_stehfest must be even")

    n_half = n_stehfest // 2
    weights = np.zeros(n_stehfest)

    for i in range(1, n_stehfest + 1):
        total = 0.0
        k_min = (i + 1) // 2
        k_max = min(i, n_half)
        for k in range(k_min, k_max + 1):
            numer = k ** n_half * math.factorial(2 * k)
            denom = (
                math.factorial(n_half - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            total += numer / denom
        weights[i - 1] = (-1) ** (n_half + i) * total

    return weights


def _pwd_laplace(u, cd, s):
    """Evaluate the Laplace-domain wellbore pressure solution.

    Parameters
    ----------
    u : float
        Laplace variable.
    cd : float
        Dimensionless wellbore storage coefficient.
    s : float
        Skin factor.

    Returns
    -------
    pwd_bar : float
        Laplace-domain wellbore pressure.
    """
    sqrt_u = np.sqrt(u)
    k0 = kv(0, sqrt_u)
    k1 = kv(1, sqrt_u)

    numerator = k0 + s * sqrt_u * k1
    denominator = u * (sqrt_u * k1 + cd * u * (k0 + s * sqrt_u * k1))

    return numerator / denominator


def compute_pwd(td_array, cd, s, n_stehfest=12):
    """Compute dimensionless wellbore pressure pwD at given times.

    Uses the Stehfest algorithm to numerically invert the Laplace-domain
    analytical solution.

    Parameters
    ----------
    td_array : np.ndarray
        Array of dimensionless times at which to evaluate pwD.
    cd : float
        Dimensionless wellbore storage coefficient.
    s : float
        Skin factor.
    n_stehfest : int
        Number of terms in Stehfest inversion (must be even).

    Returns
    -------
    pwd : np.ndarray
        Dimensionless wellbore pressure values at the given times.
    """
    weights = _stehfest_weights(n_stehfest)
    ln2 = math.log(2)
    pwd = np.zeros_like(td_array, dtype=np.float64)

    for j, t in enumerate(td_array):
        if t <= 0:
            pwd[j] = 0.0
            continue
        total = 0.0
        for i in range(1, n_stehfest + 1):
            u = i * ln2 / t
            total += weights[i - 1] * _pwd_laplace(u, cd, s)
        pwd[j] = ln2 / t * total

    return pwd


def generate_dataset(
    cd_values,
    s_values,
    n_time_points=256,
    td_min=0.01,
    td_max=1e6,
    n_stehfest=12,
):
    """Generate a dataset of wellbore pressure curves.

    For each (CD, S) combination, computes pwD(tD) over a log-spaced
    time grid.

    Parameters
    ----------
    cd_values : array-like
        List of dimensionless wellbore storage coefficients.
    s_values : array-like
        List of skin factor values.
    n_time_points : int
        Number of time grid points (log-spaced).
    td_min : float
        Minimum dimensionless time.
    td_max : float
        Maximum dimensionless time.
    n_stehfest : int
        Number of terms for Stehfest inversion.

    Returns
    -------
    td_array : np.ndarray of shape (n_time_points,)
        Log-spaced dimensionless time grid.
    pressures : np.ndarray of shape (n_samples, n_time_points)
        Wellbore pressure curves, one per (CD, S) combination.
    params : np.ndarray of shape (n_samples, 2)
        Corresponding [CD, S] parameters.
    """
    td_array = np.logspace(np.log10(td_min), np.log10(td_max), n_time_points)

    pressures = []
    params = []
    for cd in cd_values:
        for s in s_values:
            pwd = compute_pwd(td_array, cd, s, n_stehfest)
            pressures.append(pwd)
            params.append([cd, s])

    pressures = np.array(pressures)
    params = np.array(params)
    return td_array, pressures, params


if __name__ == "__main__":
    # Quick test / demo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cd_vals = [1.0, 10.0, 100.0, 1000.0]
    s_vals = [0.0, 5.0, 10.0]
    td, pressures, params = generate_dataset(
        cd_vals, s_vals, n_time_points=128, td_min=0.01, td_max=1e6
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cd, s) in enumerate(params):
        ax.semilogx(td, pressures[i], label=f"CD={cd:.0f}, S={s:.0f}")
    ax.set_xlabel("tD (dimensionless time)")
    ax.set_ylabel("pwD (dimensionless pressure)")
    ax.set_title("Wellbore Pressure Curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("welltest_pressure_curves.png", dpi=150)
    print("Saved welltest_pressure_curves.png")
