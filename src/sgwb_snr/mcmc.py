"""Posterior on G mu from the separated amplitudes (arXiv:2502.12828 Eq. 37).

Parameters theta = (log10 G mu, log10 Omega_CBC, log10 Omega_CBHE) with
log-uniform priors (the paper uses [-18, -8] for all three). The model vector is
(Omega_CS(f*; G mu), Omega_CBC, Omega_CBHE) and the likelihood is Gaussian,

    log L = -1/2 (Omega(theta) - Omega_hat)^T Sigma^-1 (Omega(theta) - Omega_hat),

with Sigma the amplitude covariance from fisher.py (the normalization term of
Eq. 37 does not depend on theta). Sampled with emcee (Foreman-Mackey et al. 2013,
as in the paper).

Omega_CS(f*; G mu) is tabulated on a fine grid in log10 G mu and interpolated, so
that whole walker ensembles are evaluated at once.
"""

import emcee
import numpy as np

from sgwb_snr.spectra import REFERENCE_FREQUENCY_HZ, omega_cosmic_strings


class AmplitudePosterior:
    """log posterior of theta for one network, equation of state and Omega_hat."""

    def __init__(self, omega_hat, covariance, equation_of_state, reheating_temperature_gev, params, prior_bounds_log10):
        self.omega_hat = np.asarray(omega_hat, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        assert self.omega_hat.shape == (3,) and covariance.shape == (3, 3)
        self.inverse_covariance = np.linalg.inv(covariance)
        low, high = prior_bounds_log10
        if not low < high:
            raise ValueError("prior bounds must satisfy low < high")
        self.bounds = (float(low), float(high))
        # Omega_CS(f*) on a grid fine enough that interpolation error is < 1e-6.
        self.grid_log10_tension = np.linspace(low, high, 20001)
        f_star = np.array([REFERENCE_FREQUENCY_HZ])
        self.grid_log10_omega = np.log10([
            omega_cosmic_strings(f_star, 10.0**x, equation_of_state, reheating_temperature_gev, params)[0]
            for x in self.grid_log10_tension
        ])

    def omega_cs(self, log10_tension):
        return 10.0 ** np.interp(log10_tension, self.grid_log10_tension, self.grid_log10_omega)

    def __call__(self, thetas):
        """Vectorized over walkers: thetas has shape (n_walkers, 3)."""
        thetas = np.atleast_2d(thetas)
        inside = np.all((thetas >= self.bounds[0]) & (thetas <= self.bounds[1]), axis=1)
        model = np.column_stack([self.omega_cs(thetas[:, 0]), 10.0 ** thetas[:, 1], 10.0 ** thetas[:, 2]])
        residual = model - self.omega_hat
        log_likelihood = -0.5 * np.einsum("wi,ij,wj->w", residual, self.inverse_covariance, residual)
        return np.where(inside, log_likelihood, -np.inf)


class NotConverged(RuntimeError):
    """The chain did not converge within max_steps."""


def sample(posterior, start_theta, n_walkers, max_steps, seed, check_every=5000):
    """emcee samples of theta after burn-in, shape (n_samples, 3).

    Runs in chunks of check_every steps until the chain is longer than 50
    integrated autocorrelation times tau and tau changed by < 1% since the last
    check (the convergence test in emcee's documentation), then discards 5 tau
    as burn-in. Raises NotConverged if that does not happen within max_steps.
    """
    start_theta = np.asarray(start_theta, dtype=float)
    assert start_theta.shape == (3,)
    rng = np.random.default_rng(seed)
    state = start_theta + 1e-3 * rng.standard_normal((n_walkers, 3))
    sampler = emcee.EnsembleSampler(n_walkers, 3, posterior, vectorize=True)
    sampler.random_state = np.random.RandomState(seed).get_state()  # reproducible chains
    previous_tau = np.inf
    while sampler.iteration < max_steps:
        state = sampler.run_mcmc(state, check_every, progress=False)
        tau = sampler.get_autocorr_time(tol=0)
        if sampler.iteration > 50.0 * tau.max() and np.all(np.abs(previous_tau - tau) < 0.01 * tau):
            return sampler.get_chain(discard=int(5.0 * tau.max()), flat=True)
        previous_tau = tau
    raise NotConverged(f"not converged after {max_steps} steps (tau = {previous_tau})")


def relative_uncertainty(log10_tension_samples):
    """Delta(G mu)/(G mu): half the 16-84 percentile width of G mu over its median."""
    tension = 10.0 ** np.asarray(log10_tension_samples)
    p16, p50, p84 = np.percentile(tension, [16.0, 50.0, 84.0])
    return 0.5 * (p84 - p16) / p50


def threshold_from_scan(string_tensions, uncertainties, target):
    """G mu where Delta(G mu)/(G mu) first falls to `target` along an increasing scan.

    Linear interpolation in (log10 G mu, log Delta) between the bracketing points.
    """
    log_tension = np.log10(np.asarray(string_tensions, dtype=float))
    log_delta = np.log(np.asarray(uncertainties, dtype=float))
    assert log_tension.shape == log_delta.shape and np.all(np.diff(log_tension) > 0)
    below = np.where(log_delta <= np.log(target))[0]
    if below.size == 0 or below[0] == 0:
        raise ValueError("target not bracketed by the scan")
    k = below[0]
    if not np.all(np.isfinite(log_delta[k - 1 : k + 1])):
        raise ValueError("an unconverged (non-finite) scan point brackets the target")
    x0, x1, y0, y1 = log_tension[k - 1], log_tension[k], log_delta[k - 1], log_delta[k]
    return 10.0 ** (x0 + (np.log(target) - y0) * (x1 - x0) / (y1 - y0))
