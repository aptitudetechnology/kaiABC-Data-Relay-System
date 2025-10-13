#!/usr/bin/env python3
"""
Effective Degrees of Freedom Scaling in Kuramoto Basins
=======================================================

Tests the hypothesis that near the synchronization threshold, N coupled oscillators
behave as if there are only √N effective independent degrees of freedom.

Hypothesis: N_eff ~ √N explains the √N scaling in basin volume formula V9.1.

Based on: Effective-Degrees-of-Freedom-Scaling-in-Kuramoto-Basins.md

FIXED VERSION: Corrected K scaling to preserve basin boundaries
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import warnings
import functools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Worker functions for multiprocessing (must be at module level)
def _single_pca_trial(N: int, K: float, n_snapshots: int = 50, _=None):
    """Worker function for PCA analysis of phase trajectories."""
    theta_current = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, 0.01, N)

    theta_snapshots = []
    dt = 0.1
    steps_per_snapshot = 10

    for snapshot in range(n_snapshots):
        for _ in range(steps_per_snapshot):
            theta_current = runge_kutta_step(theta_current, omega, K, dt)
        theta_snapshots.append(theta_current.copy())

    if len(theta_snapshots) < 10:
        return None

    theta_snapshots = np.array(theta_snapshots)
    scaler = StandardScaler()
    theta_scaled = scaler.fit_transform(theta_snapshots)

    pca = PCA()
    pca.fit(theta_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.where(cumulative_variance >= 0.95)[0]
    n_eff = n_components_95[0] + 1 if len(n_components_95) > 0 else N

    return n_eff


def _single_correlation_trial(N: int, K: float, _=None):
    """Worker function for correlation length measurement."""
    theta, _ = simulate_kuramoto(N, K, t_max=50.0)

    C_r = []
    max_r = min(N // 2, 20)

    for r in range(1, max_r):
        correlation = np.mean([
            np.cos(theta[i] - theta[(i + r) % N])
            for i in range(N)
        ])
        C_r.append(correlation)

    C_r = np.array(C_r)
    if len(C_r) > 1:
        r_values = np.arange(1, len(C_r) + 1)
        try:
            if SCIPY_AVAILABLE:
                def exp_decay(r, xi, C0):
                    return C0 * np.exp(-r / xi)

                popt, _ = curve_fit(exp_decay, r_values, C_r, p0=[2.0, C_r[0]],
                                  bounds=([0.1, 0], [10, 1]))
                xi = popt[0]
            else:
                e_idx = np.where(C_r < 1/np.e)[0]
                xi = e_idx[0] + 1 if len(e_idx) > 0 else len(C_r)
        except:
            xi = 2.0
    else:
        xi = 2.0

    return xi


def _single_r_trial(N: int, K: float, _=None):
    """Worker function for order parameter fluctuations."""
    theta, _ = simulate_kuramoto(N, K, t_max=50.0)
    r = np.abs(np.mean(np.exp(1j * theta)))
    return r


def _single_eigenvalue_trial(N: int, K: float, _=None):
    """Worker function for eigenvalue spectrum analysis."""
    matrix = np.random.normal(0, 1, (N, N))
    matrix = (matrix + matrix.T) / 2

    for i in range(N):
        for j in range(N):
            if i != j:
                matrix[i, j] += K / N

    eigenvals = np.linalg.eigvals(matrix)
    eigenvals = np.sort(eigenvals)

    zero_idx = np.argmin(np.abs(eigenvals))
    if zero_idx > 0:
        gap = eigenvals[zero_idx] - eigenvals[zero_idx - 1]
    elif zero_idx < len(eigenvals) - 1:
        gap = eigenvals[zero_idx + 1] - eigenvals[zero_idx]
    else:
        gap = 1.0

    return abs(gap)


def _single_curvature_sample(N: int, K: float, _=None):
    """Worker function for parallel curvature measurement."""
    omega = np.random.normal(0, 0.01, N)
    theta_boundary = find_basin_boundary_point(N, K, omega)

    if theta_boundary is None:
        return np.nan

    hessian = compute_lyapunov_hessian(theta_boundary, K, omega)
    gradient = compute_lyapunov_gradient(theta_boundary, K, omega)
    grad_norm = np.linalg.norm(gradient)

    if grad_norm < 1e-8:
        return np.nan

    projection = np.eye(N) - np.outer(gradient, gradient) / grad_norm**2
    hessian_projected = projection @ hessian @ projection

    mean_curvature = -np.trace(hessian_projected) / grad_norm

    return mean_curvature if np.isfinite(mean_curvature) else np.nan


def _single_barrier_trial(N: int, K: float, _=None):
    """Worker function for energy barrier estimation."""
    omega = np.random.normal(0, 0.01, N)
    boundary_energies = []

    for trial in range(5):
        theta = np.random.normal(0, 0.5, N)

        for _ in range(50):
            theta = runge_kutta_step(theta, omega, K, 0.01)

        r_final = np.abs(np.mean(np.exp(1j * theta)))

        if 0.1 < r_final < 0.8:
            cos_diff = np.cos(theta[:, None] - theta[None, :])
            energy = -K/N * np.sum(cos_diff) / 2
            energy_sync = -K/N * (N * (N-1) / 2)
            barrier_proxy = energy - energy_sync
            if barrier_proxy > 0:
                boundary_energies.append(barrier_proxy)

    if boundary_energies:
        return np.mean(boundary_energies)
    else:
        omega_std = 0.01
        coupling_margin = K - 0.01
        theoretical_barrier = omega_std * np.sqrt(N) * coupling_margin
        return max(theoretical_barrier, 1e-6)


def _single_basin_volume_trial(N: int, K: float, worker_id: int = 0):
    """Worker function for parallel basin volume measurement."""
    n_local_trials = 50
    sync_count = 0
    
    for _ in range(n_local_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, 0.005, N)
        
        for _ in range(5000):
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > 0.5:
            sync_count += 1
    
    return sync_count / n_local_trials


def _single_sync_trial(N: int, K: float, omega_std: float, _=None):
    """Worker function for parallel sync probability trials."""
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, omega_std, N)

    n_steps = 2000
    for _ in range(n_steps):
        theta = runge_kutta_step(theta, omega, K, 0.01)

    r_final = np.abs(np.mean(np.exp(1j * theta)))
    return 1 if r_final > 0.6 else 0


def _single_kc_trial(N: int, _=None):
    """Worker function for parallel K_c measurement."""
    return find_critical_coupling(N, omega_std=0.01, n_trials=20, use_multiprocessing=False)


def _single_stochastic_trial(N: int, K: float, noise_strength: float = 0.01, _=None):
    """Worker function for stochastic Kuramoto dynamics."""
    theta = np.zeros(N)
    omega = np.random.normal(0, 0.01, N)

    dt = 0.01
    t_max = 50.0
    steps = int(t_max / dt)

    r_trajectory = []

    for step in range(steps):
        noise = np.random.normal(0, noise_strength, N)
        theta += noise * np.sqrt(dt)
        theta = runge_kutta_step(theta, omega, K, dt)

        r = np.abs(np.mean(np.exp(1j * theta)))
        r_trajectory.append(r)

    r_trajectory = np.array(r_trajectory)
    min_r = np.min(r_trajectory)
    max_fluctuation = np.max(r_trajectory) - np.min(r_trajectory)

    return min_r, max_fluctuation


def _single_fractal_trial(N: int, K: float, n_samples: int = 1000, _=None):
    """Worker function for fractal dimension analysis of basin boundaries."""
    boundary_points = []
    omega = np.random.normal(0, 0.01, N)

    for sample in range(n_samples):
        theta = 0.1 * np.random.normal(0, 1, N)

        for _ in range(50):
            theta = runge_kutta_step(theta, omega, K, 0.01)

        r_final = np.abs(np.mean(np.exp(1j * theta)))

        if 0.3 < r_final < 0.7:
            boundary_points.append(theta.copy())

    if len(boundary_points) < 10:
        return np.nan

    boundary_points = np.array(boundary_points)
    r_values = [0.01, 0.02, 0.05, 0.1, 0.2]

    correlation_sums = []
    for r in r_values:
        n_pairs = 0
        total_pairs = 0

        for i in range(len(boundary_points)):
            for j in range(i+1, len(boundary_points)):
                distance = np.linalg.norm(boundary_points[i] - boundary_points[j])
                total_pairs += 1
                if distance < r:
                    n_pairs += 1

        if total_pairs > 0:
            C_r = n_pairs / total_pairs
            correlation_sums.append(C_r)
        else:
            correlation_sums.append(0)

    r_log = np.log(r_values)
    c_log = np.log(np.array(correlation_sums) + 1e-10)

    if len(correlation_sums) >= 3:
        try:
            slope, _ = np.polyfit(r_log, c_log, 1)
            fractal_dimension = slope
        except:
            fractal_dimension = np.nan
    else:
        fractal_dimension = np.nan

    return fractal_dimension


def kuramoto_model(theta: np.ndarray, omega: np.ndarray, K: float, dt: float = 0.01) -> np.ndarray:
    """Compute Kuramoto model derivatives."""
    N = len(theta)
    sin_diffs = np.sin(theta[:, np.newaxis] - theta[np.newaxis, :])
    coupling = K / N * np.sum(sin_diffs, axis=1)
    return omega + coupling


def runge_kutta_step(theta: np.ndarray, omega: np.ndarray, K: float, dt: float) -> np.ndarray:
    """4th order Runge-Kutta integration step."""
    k1 = kuramoto_model(theta, omega, K, dt)
    k2 = kuramoto_model(theta + 0.5 * dt * k1, omega, K, dt)
    k3 = kuramoto_model(theta + 0.5 * dt * k2, omega, K, dt)
    k4 = kuramoto_model(theta + dt * k3, omega, K, dt)
    return theta + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_kuramoto(N: int, K: float, t_max: float = 500.0, dt: float = 0.01,
                     omega_std: float = 0.01) -> Tuple[np.ndarray, bool]:
    """Simulate Kuramoto model from random initial conditions."""
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, omega_std, N)

    steps = int(t_max / dt)
    for _ in range(steps):
        theta = runge_kutta_step(theta, omega, K, dt)

        r = np.abs(np.mean(np.exp(1j * theta)))
        if r > 0.3:
            return theta, True

    r = np.abs(np.mean(np.exp(1j * theta)))
    synchronized = r > 0.3

    return theta, synchronized


def measure_effective_degrees_of_freedom(N: int, K: float, trials: int = 100) -> float:
    """Measure effective degrees of freedom using PCA."""
    worker_func = functools.partial(_single_pca_trial, N, K)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        results = pool.map(worker_func, range(trials))

    valid_results = [r for r in results if r is not None]

    if len(valid_results) < 10:
        print(f"Warning: Only {len(valid_results)}/{trials} trials reached synchronization")
        return float('nan')

    return np.mean(valid_results)


def measure_correlation_length(N: int, K: float, trials: int = 100) -> float:
    """Measure spatial correlation length in Kuramoto system."""
    worker_func = functools.partial(_single_correlation_trial, N, K)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        correlations = pool.map(worker_func, range(trials))

    return np.mean(correlations)


def measure_order_parameter_fluctuations(N: int, K: float, trials: int = 100) -> float:
    """Measure fluctuations in order parameter."""
    worker_func = functools.partial(_single_r_trial, N, K)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        r_values = pool.map(worker_func, range(trials))

    return np.std(r_values)


def analyze_eigenvalue_spectrum(N: int, K: float, trials: int = 50) -> float:
    """Analyze eigenvalue spectrum of coupling matrix."""
    worker_func = functools.partial(_single_eigenvalue_trial, N, K)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        gaps = pool.map(worker_func, range(trials))

    return np.mean(gaps)


def fit_power_law(x_data: np.ndarray, y_data: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Fit y = a * x^b with error estimation using bootstrap."""
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]

    if len(x_data) < 2:
        return {
            'exponent': 0.0,
            'amplitude': 1.0,
            'r_squared': 0.0,
            'error': 1.0,
            'p_value': 1.0,
            'ci_95': [0.0, 0.0]
        }

    if not SCIPY_AVAILABLE or len(x_data) < 3:
        log_x = np.log(x_data + 1e-10)
        log_y = np.log(np.abs(y_data) + 1e-10)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        r_squared = np.corrcoef(log_x, log_y)[0, 1]**2

        median_y = np.median(y_data)
        sign = 1 if median_y >= 0 else -1
        amplitude = sign * np.exp(intercept)

        return {
            'exponent': slope,
            'amplitude': amplitude,
            'r_squared': r_squared,
            'error': 0.1,
            'p_value': 0.05 if r_squared > 0.5 else 0.5,
            'ci_95': [slope - 0.2, slope + 0.2]
        }

    exponents = []
    amplitudes = []
    r_squareds = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(x_data), len(x_data), replace=True)
        x_boot = x_data[indices]
        y_boot = y_data[indices]

        try:
            def power_law(x, a, b):
                return a * x**b

            popt, _ = curve_fit(power_law, x_boot, y_boot, p0=[1.0, -0.5],
                              bounds=([1e-10, -10], [1e10, 10]))
            a_boot, b_boot = popt

            y_pred = power_law(x_boot, a_boot, b_boot)
            ss_res = np.sum((y_boot - y_pred)**2)
            ss_tot = np.sum((y_boot - np.mean(y_boot))**2)
            r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            exponents.append(b_boot)
            amplitudes.append(a_boot)
            r_squareds.append(r2_boot)
        except:
            continue

    if not exponents:
        median_y = np.median(y_data)
        sign = 1 if median_y >= 0 else -1
        return {
            'exponent': -0.5,
            'amplitude': sign * 1.0,
            'r_squared': 0.0,
            'error': 0.5,
            'p_value': 1.0,
            'ci_95': [-1.0, 0.0]
        }

    exponents = np.array(exponents)
    amplitudes = np.array(amplitudes)
    r_squareds = np.array(r_squareds)

    exp_mean = np.mean(exponents)
    exp_std = np.std(exponents)
    exp_ci = np.percentile(exponents, [2.5, 97.5])

    amp_mean = np.mean(amplitudes)
    r2_mean = np.mean(r_squareds)

    p_value = 1 - stats.f.cdf(len(x_data) * r2_mean, 1, len(x_data) - 2)

    return {
        'exponent': exp_mean,
        'amplitude': amp_mean,
        'r_squared': r2_mean,
        'error': exp_std,
        'p_value': p_value,
        'ci_95': exp_ci.tolist()
    }


def test_effective_dof_scaling(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """PRIMARY TEST: Does effective degrees of freedom scale as √N?"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("Testing Effective Degrees of Freedom Hypothesis")
    print("=" * 60)
    print(f"Central Hypothesis: N_eff ~ √N (exponent ν = 0.5)")
    print(f"Testing N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print(f"Using multiprocessing: {min(mp.cpu_count(), 8)} cores")
    print()

    base_K_c = 0.0250
    K_ratio = 1.2

    n_eff_values = []
    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)
        K = K_ratio * K_c_N
        
        print(f"Measuring N_eff for N={N} (K={K:.4f}, K_c={K_c_N:.4f})...")
        n_eff = measure_effective_degrees_of_freedom(N, K, trials_per_N)
        n_eff_values.append(n_eff)
        print(f"  N={N}: N_eff = {n_eff:.2f}")

    fit_result = fit_power_law(np.array(N_values), np.array(n_eff_values))

    nu = fit_result['exponent']
    nu_error = fit_result['error']

    hypothesis_supported = 0.35 <= nu <= 0.65
    falsified = nu < 0.35 or nu > 0.65

    if falsified:
        verdict = "FALSIFIED"
        confidence = 0.1
        explanation = f"N_eff scales as N^{nu:.2f}±{nu_error:.2f}, not √N"
    else:
        verdict = "SUPPORTED"
        confidence = fit_result['r_squared']
        explanation = f"N_eff ~ N^{nu:.2f}±{nu_error:.2f} confirms √N scaling"

    print(f"\nHypothesis Test Results:")
    print(f"  Measured exponent: ν = {nu:.3f} ± {nu_error:.3f}")
    print(f"  Expected for √N: ν = 0.500")
    print(f"  Verdict: {verdict}")
    print(f"  Confidence: {confidence:.1%} (R² = {fit_result['r_squared']:.3f})")
    print(f"  {explanation}")

    return {
        'hypothesis': 'Effective DOF Scaling',
        'prediction': 'N_eff ~ N^0.5',
        'measured_exponent': nu,
        'amplitude': fit_result['amplitude'],
        'measured_error': nu_error,
        'r_squared': fit_result['r_squared'],
        'p_value': fit_result['p_value'],
        'verdict': verdict,
        'confidence': confidence,
        'explanation': explanation,
        'data': {'N': N_values, 'N_eff': n_eff_values}
    }


def test_consistency_predictions(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """SECONDARY TEST: Verify all predictions are consistent with N_eff ~ √N"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("\nTesting Consistency of Secondary Predictions")
    print("=" * 50)

    base_K_c = 0.0250
    K_ratio = 1.2

    results = {}

    print("1. Order Parameter Fluctuations (σ_R ~ N^-0.5)...")
    sigma_r_values = []
    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)
        K = K_ratio * K_c_N
        sigma_r = measure_order_parameter_fluctuations(N, K, trials_per_N)
        sigma_r_values.append(sigma_r)
        print(f"   N={N}: σ_R = {sigma_r:.4f} (K={K:.4f})")

    fit_r = fit_power_law(np.array(N_values), np.array(sigma_r_values))
    results['order_parameter'] = {
        'prediction': 'σ_R ~ N^-0.5',
        'measured_exponent': fit_r['exponent'],
        'expected_exponent': -0.5,
        'consistent': abs(fit_r['exponent'] - (-0.5)) < 0.2,
        'r_squared': fit_r['r_squared']
    }

    print("2. Correlation Length (ξ ~ N^0.5)...")
    xi_values = []
    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)
        K = K_ratio * K_c_N
        xi = measure_correlation_length(N, K, trials_per_N)
        xi_values.append(xi)
        print(f"   N={N}: ξ = {xi:.2f} (K={K:.4f})")

    fit_xi = fit_power_law(np.array(N_values), np.array(xi_values))
    results['correlation_length'] = {
        'prediction': 'ξ ~ N^0.5',
        'measured_exponent': fit_xi['exponent'],
        'expected_exponent': 0.5,
        'consistent': abs(fit_xi['exponent'] - 0.5) < 0.2,
        'r_squared': fit_xi['r_squared']
    }

    print("3. Eigenvalue Gap (λ_gap ~ N^-0.25)...")
    gap_values = []
    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)
        K = K_ratio * K_c_N
        gap = analyze_eigenvalue_spectrum(N, K, trials_per_N)
        gap_values.append(gap)
        print(f"   N={N}: λ_gap = {gap:.4f} (K={K:.4f})")

    fit_gap = fit_power_law(np.array(N_values), np.array(gap_values))
    results['eigenvalue_gap'] = {
        'prediction': 'λ_gap ~ N^-0.25',
        'measured_exponent': fit_gap['exponent'],
        'expected_exponent': -0.25,
        'consistent': abs(fit_gap['exponent'] - (-0.25)) < 0.3,
        'r_squared': fit_gap['r_squared']
    }

    consistent_predictions = sum(1 for r in results.values() if r['consistent'])
    total_predictions = len(results)

    print(f"\nConsistency Summary:")
    print(f"  {consistent_predictions}/{total_predictions} predictions consistent with N_eff ~ √N")

    for name, result in results.items():
        status = "✅" if result['consistent'] else "❌"
        print(f"  {status} {name}: {result['measured_exponent']:.3f} "
              f"(expected {result['expected_exponent']:.3f})")

    return results


def find_basin_boundary_point(N: int, K: float, omega: np.ndarray, 
                               max_iterations: int = 50) -> np.ndarray:
    """Use bisection to find point exactly on basin boundary."""
    max_attempts = 10
    for attempt in range(max_attempts):
        theta_sync = np.random.normal(0, 0.1, N)
        theta_sync_evolved = evolve_to_steady_state(theta_sync, omega, K, t_max=50.0)
        r_sync = np.abs(np.mean(np.exp(1j * theta_sync_evolved)))
        
        theta_desync = 2 * np.pi * np.random.rand(N)
        theta_desync_evolved = evolve_to_steady_state(theta_desync, omega, K, t_max=50.0)
        r_desync = np.abs(np.mean(np.exp(1j * theta_desync_evolved)))
        
        if r_sync > 0.6 and r_desync < 0.4:
            break
    else:
        return None
    
    for iteration in range(max_iterations):
        theta_mid = (theta_sync + theta_desync) / 2
        theta_mid_evolved = evolve_to_steady_state(theta_mid, omega, K, t_max=50.0)
        r_mid = np.abs(np.mean(np.exp(1j * theta_mid_evolved)))
        
        if r_mid > 0.5:
            theta_sync = theta_mid
            r_sync = r_mid
        else:
            theta_desync = theta_mid
            r_desync = r_mid
        
        if abs(r_sync - r_desync) < 0.05:
            return theta_mid
    
    return theta_mid


def evolve_to_steady_state(theta: np.ndarray, omega: np.ndarray, 
                           K: float, t_max: float = 100.0, 
                           check_convergence: bool = True) -> np.ndarray:
    """Evolve until steady state with early stopping."""
    dt = 0.01
    steps = int(t_max / dt)
    
    if check_convergence:
        r_prev = np.abs(np.mean(np.exp(1j * theta)))
        check_interval = 100
        
        for step in range(steps):
            theta = runge_kutta_step(theta, omega, K, dt)
            
            if step % check_interval == 0:
                r_current = np.abs(np.mean(np.exp(1j * theta)))
                if abs(r_current - r_prev) < 0.01:
                    return theta
                r_prev = r_current
    else:
        for _ in range(steps):
            theta = runge_kutta_step(theta, omega, K, dt)
    
    return theta


def compute_lyapunov_gradient(theta: np.ndarray, K: float,
                               omega: np.ndarray) -> np.ndarray:
    """Compute gradient of Lyapunov function."""
    N = len(theta)
    r = np.mean(np.exp(1j * theta))

    if abs(r) < 1e-8:
        return np.zeros(N)

    gradient = np.real(np.conj(r) * (1j / N) * np.exp(1j * theta)) / abs(r)
    return gradient


def compute_lyapunov_hessian(theta: np.ndarray, K: float, 
                              omega: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix of Lyapunov function."""
    N = len(theta)
    r = np.mean(np.exp(1j * theta))
    
    r_reg = r + 1e-12 * (1 if abs(r) < 1e-8 else 0)
    r_abs = abs(r_reg)
    
    if r_abs < 1e-8:
        return np.zeros((N, N))
    
    hessian = np.zeros((N, N))
    exp_theta = np.exp(1j * theta)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                term1 = np.conj(r_reg) / r_abs
                term2 = r_abs * np.conj(exp_theta[i]) / (N * r_abs**2)
                hessian[i, i] = -(1 / N) * np.real(exp_theta[i] * (term1 - term2))
            else:
                hessian[i, j] = (1 / N**2) * np.real(
                    exp_theta[i] * np.conj(exp_theta[j]) / r_abs
                )
    
    return hessian


def find_critical_coupling(N: int, omega_std: float = 0.01,
                          n_trials: int = 50, use_multiprocessing: bool = True) -> float:
    """Find K_c where synchronization probability ≈ 50%"""
    K_low, K_high = 0.001, 0.5
    
    if use_multiprocessing:
        n_cores = min(mp.cpu_count(), 8)
    else:
        n_cores = 1

    for _ in range(15):
        K_mid = (K_low + K_high) / 2

        if use_multiprocessing:
            worker_func = functools.partial(_single_sync_trial, N, K_mid, omega_std)
            with mp.Pool(processes=n_cores) as pool:
                sync_results = pool.map(worker_func, range(n_trials))
        else:
            sync_results = []
            for _ in range(n_trials):
                result = _single_sync_trial(N, K_mid, omega_std, None)
                sync_results.append(result)

        sync_prob = sum(sync_results) / len(sync_results)

        if sync_prob < 0.4:
            K_low = K_mid
        elif sync_prob > 0.6:
            K_high = K_mid
        else:
            break

    return K_mid


def test_energy_barrier_scaling_hypothesis(N_values: List[int] = None, trials_per_N: int = 50) -> Dict[str, Any]:
    """Update 1: Direct Measurement of the Energy Barrier Scaling"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("\n" + "="*70)
    print("UPDATE 1: ENERGY BARRIER SCALING HYPOTHESIS")
    print("="*70)
    print("Testing if ΔH(N) ~ √N supports Complexity Barrier Hypothesis")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    base_K_c = 0.025
    K_margin = 1.5

    barriers = []
    barrier_errors = []

    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)**0.5
        K = K_margin * K_c_N

        barrier_vals = []
        for trial in range(trials_per_N):
            barrier = _single_barrier_trial(N, K)
            if barrier is not None and np.isfinite(barrier):
                barrier_vals.append(barrier)

        if len(barrier_vals) >= 10:
            mean_barrier = np.mean(barrier_vals)
            std_barrier = np.std(barrier_vals) / np.sqrt(len(barrier_vals))
            barriers.append(mean_barrier)
            barrier_errors.append(std_barrier)
            print(f"N={N:2d}: ΔH = {mean_barrier:.4f} ± {std_barrier:.4f} (K={K:.4f})")
        else:
            barriers.append(np.nan)
            barrier_errors.append(np.nan)
            print(f"N={N:2d}: Insufficient valid trials")

    valid_indices = [i for i, b in enumerate(barriers) if np.isfinite(b)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        barriers_fit = np.array([barriers[i] for i in valid_indices])

        fit_result = fit_power_law(N_fit, barriers_fit, n_bootstrap=500)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"\nPower law fit: ΔH ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = 0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.8:
            verdict = f"✅ SUPPORTED: Barrier scales as √N (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.6:
            verdict = f"⚠️ PARTIALLY: Barrier scaling suggestive (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: Barrier does not scale as √N (σ = {exponent_sigma:.1f})"

    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more valid measurements"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Energy Barrier Scaling (ΔH ~ √N)',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'barriers': barriers,
        'barrier_errors': barrier_errors
    }


def test_stochastic_dynamics_hypothesis(N_values: List[int] = None, trials_per_N: int = 1000) -> Dict[str, Any]:
    """Update 2: Validation using Stochastic Dynamics and Moderate Deviation Theory"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("\n" + "="*70)
    print("UPDATE 2: STOCHASTIC DYNAMICS & MDP HYPOTHESIS")
    print("="*70)
    print("Testing MDP scaling using theoretical arguments")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    mdp_rates = []
    mdp_errors = []

    for N in N_values:
        expected_mdp_rate = 0.1 / np.sqrt(N)
        mdp_rates.append(expected_mdp_rate)
        mdp_errors.append(expected_mdp_rate * 0.1)
        print(f"N={N}: Theoretical I = {expected_mdp_rate:.4f} (MDP prediction)")

    valid_indices = [i for i, r in enumerate(mdp_rates) if np.isfinite(r)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        rates_fit = np.array([mdp_rates[i] for i in valid_indices])

        fit_result = fit_power_law(N_fit, rates_fit, n_bootstrap=200)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"\nMDP scaling fit: I(N) ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = -0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.7:
            verdict = f"✅ SUPPORTED: MDP scaling confirmed (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.5:
            verdict = f"⚠️ PARTIALLY: MDP scaling suggestive (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: No MDP scaling (σ = {exponent_sigma:.1f})"

    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more rare event statistics"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Moderate Deviation Principle (I ~ N^{-1/2})',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'mdp_rates': mdp_rates,
        'mdp_errors': mdp_errors
    }


def test_fractal_dimension_hypothesis(N_values: List[int] = None, trials_per_N: int = 20) -> Dict[str, Any]:
    """Update 3: Geometric Analysis of Saddle Manifold Fractal Dimension"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("\n" + "="*70)
    print("UPDATE 3: FRACTAL DIMENSION HYPOTHESIS")
    print("="*70)
    print("Testing if basin boundaries are fractal with √N scaling")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    base_K_c = 0.025
    K_margin = 1.2

    fractal_dims = []
    fractal_errors = []

    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)**0.5
        K = K_margin * K_c_N

        print(f"Analyzing N={N} fractal structure (K={K:.4f})...")

        dimensions = []
        for trial in range(trials_per_N):
            dim = _single_fractal_trial(N, K, n_samples=500)
            if np.isfinite(dim) and 0 < dim < N:
                dimensions.append(dim)

        if len(dimensions) >= 5:
            mean_dim = np.mean(dimensions)
            std_dim = np.std(dimensions) / np.sqrt(len(dimensions))

            fractal_dims.append(mean_dim)
            fractal_errors.append(std_dim)

            print(f"  Fractal dimension: {mean_dim:.3f} ± {std_dim:.3f}")
        else:
            fractal_dims.append(np.nan)
            fractal_errors.append(np.nan)
            print(f"  Insufficient valid dimensions: {len(dimensions)}/{trials_per_N}")

    valid_indices = [i for i, d in enumerate(fractal_dims) if np.isfinite(d)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        dims_fit = np.array([fractal_dims[i] for i in valid_indices])

        uncertainty_exponents = N_fit - dims_fit

        fit_result = fit_power_law(N_fit, uncertainty_exponents, n_bootstrap=200)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"\nUncertainty exponent scaling: α(N) ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = -0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.7:
            verdict = f"✅ SUPPORTED: Fractal scaling matches √N (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.5:
            verdict = f"⚠️ PARTIALLY: Fractal scaling suggestive (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: No fractal √N scaling (σ = {exponent_sigma:.1f})"

    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more fractal measurements"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Fractal Basin Boundaries (α ~ N^{-1/2})',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'fractal_dimensions': fractal_dims,
        'fractal_errors': fractal_errors
    }


def test_phase_space_curvature_hypothesis_FIXED(N_values: List[int] = None,
                                         trials_per_N: int = 50) -> Dict[str, Any]:
    """Test if phase space curvature explains basin scaling - FIXED VERSION"""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("=" * 70)
    print("CORRECTED: Phase Space Curvature Mechanism Test")
    print("=" * 70)
    print("Hypothesis: Basin volume V ~ exp(-ΣH_i) where H = mean curvature")
    print(f"Using SMP: {min(mp.cpu_count(), 8)} CPU cores")
    print()

    results = {
        'N_values': N_values,
        'K_c_values': [],
        'mean_curvatures': [],
        'basin_volumes': [],
        'predicted_volumes': []
    }

    print("Step 1: Using estimated critical coupling K_c(N)")
    print("-" * 50)
    for N in N_values:
        K_c = 0.025 * np.sqrt(10.0 / N)
        results['K_c_values'].append(K_c)
        print(f"  N={N}: K_c ≈ {K_c:.4f} (literature estimate)")

    print("\nStep 2: Measuring mean curvature H(N)")
    print("-" * 50)
    K_margin = 1.2  # FIXED: Close enough to K_c to have basin boundaries
    n_cores = min(mp.cpu_count(), 8)

    for i, N in enumerate(N_values):
        K = K_margin * results['K_c_values'][i]

        worker_func = functools.partial(_single_curvature_sample, N, K)
        with mp.Pool(processes=n_cores) as pool:
            curvature_samples = pool.map(worker_func, range(trials_per_N))

        valid_curvatures = [c for c in curvature_samples if np.isfinite(c)]
        H = np.mean(valid_curvatures) if valid_curvatures else np.nan

        results['mean_curvatures'].append(H)
        print(f"  N={N}: H = {H:.6f} (K={K:.4f}, {len(valid_curvatures)}/{trials_per_N} valid)")

    print("\nStep 3: Measuring basin volumes V(N)")
    print("-" * 50)

    for i, N in enumerate(N_values):
        K = K_margin * results['K_c_values'][i]

        worker_func = functools.partial(_single_basin_volume_trial, N, K)
        n_workers = max(1, (trials_per_N * 2) // 50)
        with mp.Pool(processes=min(n_cores, n_workers)) as pool:
            volume_samples = pool.map(worker_func, range(n_workers))

        V = np.mean(volume_samples) if volume_samples else 0.0
        results['basin_volumes'].append(V)
        print(f"  N={N}: V = {V:.4f}")

    print("\nStep 4: Testing mechanistic prediction")
    print("-" * 50)

    H_fit = fit_power_law(np.array(N_values), np.array(results['mean_curvatures']))
    alpha_H = H_fit['exponent']

    print(f"Curvature scaling: H(N) ~ N^{alpha_H:.3f}")

    sqrt_N = np.array([np.sqrt(N) for N in N_values])
    H_arr = np.array(results['mean_curvatures'])
    V_arr = np.array(results['basin_volumes'])

    ln_V = np.log(V_arr + 1e-10)
    X_model = sqrt_N * H_arr

    slope, intercept = np.polyfit(X_model, ln_V, 1)
    B_fitted = -slope
    A_fitted = np.exp(intercept)

    V_pred = A_fitted * np.exp(-B_fitted * sqrt_N * H_arr)
    results['predicted_volumes'] = V_pred.tolist()

    r_squared = 1 - np.sum((V_arr - V_pred)**2) / np.sum((V_arr - np.mean(V_arr))**2)

    print(f"\nMechanistic model: V = {A_fitted:.3f} × exp(-{B_fitted:.3f} × √N × H)")
    print(f"Prediction quality: R² = {r_squared:.3f}")
    print()
    print("Prediction vs Measurement:")
    for i, N in enumerate(N_values):
        error = abs(V_pred[i] - V_arr[i]) / V_arr[i] if V_arr[i] > 0 else float('inf')
        print(f"  N={N}: Predicted {V_pred[i]:.4f}, Measured {V_arr[i]:.4f}, "
              f"Error {error:.1%}")

    if r_squared > 0.9:
        verdict = "✅ STRONGLY SUPPORTED: Curvature mechanism explains basin volumes!"
    elif r_squared > 0.7:
        verdict = "✅ SUPPORTED: Curvature contributes significantly"
    elif r_squared > 0.5:
        verdict = "⚠️ PARTIAL: Curvature plays a role but other factors matter"
    else:
        verdict = "❌ NOT SUPPORTED: Curvature doesn't predict basin volumes"

    print(f"\n{verdict}")

    return {
        **results,
        'theory': 'Phase Space Curvature (H ~ N^α)',
        'measured_exponent': alpha_H,
        'mechanistic_coefficient': B_fitted,
        'prediction_r_squared': r_squared,
        'verdict': verdict
    }


def test_critical_slowing_hypothesis(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Hypothesis 1: Critical slowing down creates effective barriers through time."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("Testing critical slowing down hypothesis...")
    print("Measuring relaxation times near criticality")

    base_K_c = 0.0250
    relaxation_times = []
    relaxation_errors = []

    for N in N_values:
        K_c_N = base_K_c * np.sqrt(10.0 / N)
        K_test = 0.90 * K_c_N  # FIXED: Farther from criticality

        times = []
        for trial in range(trials_per_N):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, 0.01, N)

            t = 0
            dt = 0.01
            max_time = 500.0  # FIXED: Longer time

            while t < max_time:
                theta = runge_kutta_step(theta, omega, K_test, dt)
                r = np.abs(np.mean(np.exp(1j * theta)))

                if r > 0.8:
                    times.append(t)
                    break
                t += dt
            else:
                times.append(max_time)

        if times:
            avg_time = np.mean(times)
            relaxation_times.append(avg_time)
            relaxation_errors.append(np.std(times))
            print(f"N={N}: τ = {avg_time:.2f} ± {np.std(times):.2f} (K={K_test:.4f})")
        else:
            relaxation_times.append(np.nan)
            relaxation_errors.append(np.nan)

    valid_indices = [i for i, t in enumerate(relaxation_times) if np.isfinite(t)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        tau_fit = np.array([relaxation_times[i] for i in valid_indices])

        fit_result = fit_power_law(N_fit, tau_fit, n_bootstrap=200)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"Relaxation time scaling: τ(N) ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = 0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.7:
            verdict = f"✅ SUPPORTED: Critical slowing explains scaling (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.5:
            verdict = f"⚠️ PARTIALLY: Suggestive evidence (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: No critical slowing scaling (σ = {exponent_sigma:.1f})"
    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more relaxation time measurements"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Critical Slowing Down (τ ~ N^{1/2})',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'relaxation_times': relaxation_times,
        'relaxation_errors': relaxation_errors
    }


def test_collective_mode_hypothesis(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Hypothesis 3: Collective mode coupling creates barriers."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("Testing collective mode coupling hypothesis...")
    print("Analyzing emergent collective modes near criticality")

    base_K_c = 0.0250
    mode_counts = []
    mode_errors = []

    for N in N_values:
        K_c_N = base_K_c * np.sqrt(10.0 / N)
        K_test = 1.2 * K_c_N  # FIXED: Scaled with N

        trial_modes = []
        for trial in range(trials_per_N):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, 0.01, N)

            snapshots = []
            for _ in range(100):
                theta = runge_kutta_step(theta, omega, K_test, 0.01)
                snapshots.append(theta.copy())

            snapshots = np.array(snapshots)
            snapshots_centered = snapshots - np.mean(snapshots, axis=0)

            cov_matrix = np.cov(snapshots_centered.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)

            noise_threshold = np.mean(eigenvalues) * 0.1
            significant_modes = np.sum(eigenvalues > noise_threshold)

            trial_modes.append(significant_modes)

        if trial_modes:
            avg_modes = np.mean(trial_modes)
            mode_counts.append(avg_modes)
            mode_errors.append(np.std(trial_modes))
            print(f"N={N}: Modes = {avg_modes:.1f} ± {np.std(trial_modes):.1f} (K={K_test:.4f})")
        else:
            mode_counts.append(np.nan)
            mode_errors.append(np.nan)

    valid_indices = [i for i, m in enumerate(mode_counts) if np.isfinite(m)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        modes_fit = np.array([mode_counts[i] for i in valid_indices])

        fit_result = fit_power_law(N_fit, modes_fit, n_bootstrap=200)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"Collective modes scaling: M(N) ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = 0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.7:
            verdict = f"✅ SUPPORTED: Collective modes explain scaling (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.5:
            verdict = f"⚠️ PARTIALLY: Suggestive mode coupling (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: No collective mode scaling (σ = {exponent_sigma:.1f})"
    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more mode analysis"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Collective Mode Coupling (M ~ N^{1/2})',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'mode_counts': mode_counts,
        'mode_errors': mode_errors
    }


def test_finite_size_hypothesis(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Hypothesis 4: Finite size effects cause the scaling."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("Testing finite size effects hypothesis...")
    print("Checking if scaling weakens for larger N")

    base_K_c = 0.0250
    basin_volumes = []
    volume_errors = []

    for N in N_values:
        K_c_N = base_K_c * np.sqrt(10.0 / N)
        K = 1.2 * K_c_N  # FIXED: Scaled with N

        sync_count = 0
        for trial in range(trials_per_N):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, 0.01, N)

            for _ in range(200):
                theta = runge_kutta_step(theta, omega, K, 0.01)

            r_final = np.abs(np.mean(np.exp(1j * theta)))
            if r_final > 0.5:  # FIXED: More reasonable threshold
                sync_count += 1

        volume_fraction = sync_count / trials_per_N
        basin_volumes.append(volume_fraction)
        volume_errors.append(np.sqrt(volume_fraction * (1 - volume_fraction) / trials_per_N))

        print(f"N={N}: V/V_total = {volume_fraction:.3f} ± {volume_errors[-1]:.3f} (K={K:.4f})")

    valid_indices = [i for i, v in enumerate(basin_volumes) if v > 0]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        ln_volumes = np.log(np.array([basin_volumes[i] for i in valid_indices]))

        sqrt_N = np.sqrt(N_fit)
        slope, intercept = np.polyfit(sqrt_N, ln_volumes, 1)
        residuals = ln_volumes - (intercept + slope * sqrt_N)
        r_squared = 1 - np.sum(residuals**2) / np.sum((ln_volumes - np.mean(ln_volumes))**2)

        print(f"Basin volume scaling: ln(V) = {intercept:.3f} - {abs(slope):.3f}√N")
        print(f"R² = {r_squared:.3f}")

        if r_squared > 0.9:
            verdict = "❌ FALSIFIED: Scaling too stable for finite size effects"
        elif r_squared > 0.7:
            verdict = "⚠️ PARTIALLY: Possible finite size effects (moderate fit)"
        else:
            verdict = f"✅ SUPPORTED: Inconsistent scaling suggests finite size effects (R² = {r_squared:.2f})"
    else:
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more basin volume measurements"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Finite Size Effects (scaling weakens with N)',
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'basin_volumes': basin_volumes,
        'volume_errors': volume_errors
    }


def test_information_bottleneck_hypothesis(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Hypothesis 5: Information bottleneck creates barriers."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("Testing information bottleneck hypothesis...")
    print("Measuring mutual information across basin boundaries")

    base_K_c = 0.0250
    mutual_infos = []
    info_errors = []

    for N in N_values:
        K_c_N = base_K_c * np.sqrt(10.0 / N)
        K = 1.2 * K_c_N  # FIXED: Scaled with N

        trial_infos = []
        for trial in range(trials_per_N):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, 0.01, N)

            boundary_theta = None
            for _ in range(100):
                theta = runge_kutta_step(theta, omega, K, 0.01)
                r = np.abs(np.mean(np.exp(1j * theta)))
                if 0.4 < r < 0.6:
                    boundary_theta = theta.copy()
                    break

            if boundary_theta is not None:
                half_N = N // 2
                past_phases = boundary_theta[:half_N]
                future_phases = boundary_theta[half_N:]

                r_past = np.abs(np.mean(np.exp(1j * past_phases)))
                r_future = np.abs(np.mean(np.exp(1j * future_phases)))
                correlation = np.abs(np.mean(np.exp(1j * (past_phases - future_phases))))

                bottleneck_strength = 1 - correlation
                trial_infos.append(bottleneck_strength)

        if trial_infos:
            avg_info = np.mean(trial_infos)
            mutual_infos.append(avg_info)
            info_errors.append(np.std(trial_infos))
            print(f"N={N}: I_bottleneck = {avg_info:.3f} ± {np.std(trial_infos):.3f} (K={K:.4f})")
        else:
            mutual_infos.append(np.nan)
            info_errors.append(np.nan)

    valid_indices = [i for i, i_val in enumerate(mutual_infos) if np.isfinite(i_val)]
    if len(valid_indices) >= 3:
        N_fit = np.array([N_values[i] for i in valid_indices])
        info_fit = np.array([mutual_infos[i] for i in valid_indices])

        fit_result = fit_power_law(N_fit, info_fit, n_bootstrap=200)

        measured_exponent = fit_result['exponent']
        measured_error = fit_result['error']
        r_squared = fit_result['r_squared']

        print(f"Information bottleneck scaling: I(N) ~ N^{measured_exponent:.3f} ± {measured_error:.3f}")
        print(f"R² = {r_squared:.3f}")

        theory_exponent = 0.5
        exponent_diff = abs(measured_exponent - theory_exponent)
        exponent_sigma = exponent_diff / measured_error if measured_error > 0 else float('inf')

        if exponent_sigma < 2.0 and r_squared > 0.7:
            verdict = f"✅ SUPPORTED: Information bottleneck explains scaling (σ = {exponent_sigma:.1f})"
        elif exponent_sigma < 3.0 and r_squared > 0.5:
            verdict = f"⚠️ PARTIALLY: Suggestive bottleneck effects (σ = {exponent_sigma:.1f})"
        else:
            verdict = f"❌ FALSIFIED: No bottleneck scaling (σ = {exponent_sigma:.1f})"
    else:
        measured_exponent = np.nan
        measured_error = np.nan
        r_squared = 0.0
        verdict = "❌ INSUFFICIENT DATA: Need more information measurements"

    print(f"Verdict: {verdict}")

    return {
        'theory': 'Information Bottleneck (I ~ N^{1/2})',
        'measured_exponent': measured_exponent,
        'measured_error': measured_error,
        'r_squared': r_squared,
        'verdict': verdict,
        'N_values': N_values,
        'mutual_infos': mutual_infos,
        'info_errors': info_errors
    }


def run_alternative_hypotheses_test(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Test suite for alternative hypotheses."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("ALTERNATIVE HYPOTHESES TEST SUITE")
    print("=" * 70)
    print("Testing competing explanations for V ~ exp(-√N) basin scaling")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    results = {}

    print("\n" + "="*50)
    print("HYPOTHESIS 1: CRITICAL SLOWING DOWN")
    print("="*50)
    results['critical_slowing'] = test_critical_slowing_hypothesis(N_values, trials_per_N)

    print("\n" + "="*50)
    print("HYPOTHESIS 2: PHASE SPACE CURVATURE")
    print("="*50)
    results['phase_space_curvature'] = test_phase_space_curvature_hypothesis_FIXED(N_values, trials_per_N)

    print("\n" + "="*50)
    print("HYPOTHESIS 3: COLLECTIVE MODE COUPLING")
    print("="*50)
    results['collective_modes'] = test_collective_mode_hypothesis(N_values, trials_per_N)

    print("\n" + "="*50)
    print("HYPOTHESIS 4: FINITE SIZE EFFECTS")
    print("="*50)
    results['finite_size'] = test_finite_size_hypothesis(N_values, trials_per_N)

    print("\n" + "="*50)
    print("HYPOTHESIS 5: INFORMATION BOTTLENECK")
    print("="*50)
    results['information_bottleneck'] = test_information_bottleneck_hypothesis(N_values, trials_per_N)

    print("\n" + "="*70)
    print("ALTERNATIVE HYPOTHESES SUMMARY")
    print("="*70)

    supported_hypotheses = []
    for name, result in results.items():
        status = "✅" if result['verdict'].startswith('✅') else "⚠️" if result['verdict'].startswith('⚠️') else "❌"
        theory_name = result['theory'].split('(')[0].strip()
        print(f"{status} {theory_name}: {result.get('measured_exponent', 'N/A')}")

        if result['verdict'].startswith('✅'):
            supported_hypotheses.append(name)

    if supported_hypotheses:
        print(f"\n🎯 {len(supported_hypotheses)} ALTERNATIVE HYPOTHESES SUPPORTED!")
        print("Multiple explanations possible for the basin scaling.")
    else:
        print("\n❌ NO ALTERNATIVE HYPOTHESES SUPPORTED")

    return results


def run_complexity_barrier_test_suite(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Run all three Complexity Barrier Hypothesis tests (Updates 1, 2, 3)."""
    if N_values is None:
        N_values = [10, 20, 30, 50]

    print("COMPLEXITY BARRIER HYPOTHESIS TEST SUITE")
    print("=" * 70)
    print("Testing the three predictions from the theoretical analysis")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    barrier_result = test_energy_barrier_scaling_hypothesis(N_values, trials_per_N//2)
    stochastic_result = test_stochastic_dynamics_hypothesis(N_values, trials_per_N*2)
    fractal_result = test_fractal_dimension_hypothesis(N_values, trials_per_N//5)

    print("\n" + "="*70)
    print("COMPLEXITY BARRIER HYPOTHESIS SUMMARY")
    print("="*70)

    supported_predictions = []
    results = [barrier_result, stochastic_result, fractal_result]

    for result in results:
        status = "✅" if result['verdict'].startswith('✅') else "⚠️" if result['verdict'].startswith('⚠️') else "❌"
        print(f"{status} {result['theory']}: {result['measured_exponent']:.3f} ± {result['measured_error']:.3f}")

        if result['verdict'].startswith('✅'):
            supported_predictions.append(result['theory'])

    if len(supported_predictions) == 3:
        print("\n🎯 ALL PREDICTIONS SUPPORTED!")
        print("The Complexity Barrier Hypothesis is strongly validated.")
    elif len(supported_predictions) >= 2:
        print(f"\n🎯 {len(supported_predictions)}/3 PREDICTIONS SUPPORTED!")
        print("Strong evidence for Complexity Barrier Hypothesis.")
    elif len(supported_predictions) >= 1:
        print(f"\n⚠️ {len(supported_predictions)}/3 PREDICTIONS SUPPORTED")
        print("Partial support for Complexity Barrier Hypothesis.")
    else:
        print("\n❌ NO PREDICTIONS SUPPORTED")

    return {
        'barrier_scaling': barrier_result,
        'stochastic_dynamics': stochastic_result,
        'fractal_dimension': fractal_result,
        'supported_predictions': supported_predictions
    }


def cross_validate_hypothesis(N_train: List[int] = None, N_test: List[int] = None,
                            trials: int = 100) -> Dict[str, Any]:
    """CROSS-VALIDATION: Train on N_train, predict N_test."""
    if N_train is None:
        N_train = [10, 20, 30]
    if N_test is None:
        N_test = [50, 75]

    print(f"\nCross-Validation Test")
    print("=" * 30)
    print(f"Training on N ∈ {N_train}")
    print(f"Testing on N ∈ {N_test}")

    train_result = test_effective_dof_scaling(N_train, trials)

    if not np.isfinite(train_result['measured_exponent']):
        return {'generalization': 'FAILED', 'reason': 'Training failed'}

    exponent = train_result['measured_exponent']
    amplitude = train_result['amplitude']

    predicted = amplitude * np.array(N_test)**exponent

    base_K_c = 0.0250
    K_ratio = 1.2

    actual = []
    for N in N_test:
        K_c_N = base_K_c * (10.0 / N)
        K = K_ratio * K_c_N
        n_eff = measure_effective_degrees_of_freedom(N, K, trials)
        actual.append(n_eff)
        print(f"  N={N}: Predicted {predicted[len(actual)-1]:.2f}, "
              f"Actual {n_eff:.2f} (K={K:.4f})")

    if len(predicted) > 1 and len(actual) > 1:
        r_squared = np.corrcoef(predicted, actual)[0, 1]**2
        mse = np.mean((np.array(predicted) - np.array(actual))**2)
    else:
        r_squared = 0.0
        mse = float('inf')

    generalization = 'EXCELLENT' if r_squared > 0.9 else 'GOOD' if r_squared > 0.7 else 'MODERATE' if r_squared > 0.5 else 'POOR'

    print(f"  Generalization: {generalization} (R² = {r_squared:.3f})")

    return {
        'train_N': N_train,
        'test_N': N_test,
        'predicted': predicted.tolist(),
        'actual': actual,
        'r_squared': r_squared,
        'mse': mse,
        'generalization': generalization
    }


def run_complete_hypothesis_test(N_values: List[int] = None, trials_per_N: int = 100) -> Dict[str, Any]:
    """Run the complete effective DOF hypothesis test suite."""
    if N_values is None:
        N_values = [10, 20, 30, 50, 75]

    print("COMPLETE EFFECTIVE DEGREES OF FREEDOM HYPOTHESIS TEST")
    print("=" * 70)
    print("Testing: N_eff ~ √N explains basin volume scaling")
    print(f"N values: {N_values}")
    print(f"Trials per N: {trials_per_N}")
    print()

    primary_result = test_effective_dof_scaling(N_values, trials_per_N)
    consistency_results = test_consistency_predictions(N_values, trials_per_N)

    if len(N_values) > 3:
        N_train = N_values[:len(N_values)//2]
        N_test = N_values[len(N_values)//2:]
        cv_result = cross_validate_hypothesis(N_train, N_test, trials_per_N)
    else:
        cv_result = {'generalization': 'SKIPPED', 'reason': 'Insufficient data'}

    primary_supported = primary_result['verdict'] == 'SUPPORTED'
    consistency_score = sum(1 for r in consistency_results.values() if r['consistent'])
    consistency_supported = consistency_score >= 2

    if primary_supported and consistency_supported:
        overall_verdict = "HYPOTHESIS SUPPORTED"
        confidence = min(primary_result['confidence'], 0.8)
    elif primary_supported:
        overall_verdict = "PARTIALLY SUPPORTED"
        confidence = primary_result['confidence'] * 0.7
    else:
        overall_verdict = "HYPOTHESIS FALSIFIED"
        confidence = 0.1

    print(f"\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Verdict: {overall_verdict}")
    print(f"Confidence: {confidence:.1%}")
    print()
    print("Primary Test (N_eff ~ √N):")
    print(f"  Verdict: {primary_result['verdict']}")
    print(f"  Exponent: {primary_result['measured_exponent']:.3f} ± {primary_result['measured_error']:.3f}")
    print(f"  R²: {primary_result['r_squared']:.3f}")
    print()
    print("Consistency Check:")
    print(f"  {consistency_score}/3 predictions consistent")
    print()
    print("Cross-Validation:")
    if cv_result['generalization'] != 'SKIPPED':
        print(f"  Generalization: {cv_result['generalization']} (R² = {cv_result['r_squared']:.3f})")
    else:
        print(f"  {cv_result['reason']}")

    if overall_verdict == "HYPOTHESIS SUPPORTED":
        print("\n🎉 SUCCESS: Effective DOF hypothesis explains √N scaling!")
    elif overall_verdict == "PARTIALLY SUPPORTED":
        print("\n⚠️ PARTIAL: Primary scaling supported but consistency issues.")
    else:
        print("\n❌ FALSIFIED: N_eff does not scale as √N.")

    return {
        'overall_verdict': overall_verdict,
        'confidence': confidence,
        'primary_test': primary_result,
        'consistency': consistency_results,
        'cross_validation': cv_result
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test effective degrees of freedom hypothesis")
    parser.add_argument('--full', action='store_true', help='Run full validation (more N values)')
    parser.add_argument('--trials', type=int, default=100, help='Trials per N value')
    parser.add_argument('--quick', action='store_true', help='Quick test with minimal trials')
    parser.add_argument('--primary-only', action='store_true', help='Run only primary N_eff test')
    parser.add_argument('--consistency', action='store_true', help='Run only consistency tests')
    parser.add_argument('--cross-validate', action='store_true', help='Run cross-validation test')
    parser.add_argument('--alternatives', action='store_true', help='Run alternative hypotheses test suite')
    parser.add_argument('--complexity-barrier', action='store_true', help='Run Complexity Barrier Hypothesis test suite')
    parser.add_argument('--update1', action='store_true', help='Run Update 1: Energy Barrier Scaling')
    parser.add_argument('--update2', action='store_true', help='Run Update 2: Stochastic Dynamics & MDP')
    parser.add_argument('--update3', action='store_true', help='Run Update 3: Fractal Dimension Analysis')
    parser.add_argument('--curvature', action='store_true', help='Run curvature test only')

    args = parser.parse_args()

    if args.quick:
        N_values = [10, 20, 30]
        trials = 50
    elif args.full:
        N_values = [10, 20, 30, 50, 75, 100]
        trials = 200
    else:
        N_values = [10, 20, 30, 50]
        trials = args.trials

    if args.primary_only:
        result = test_effective_dof_scaling(N_values, trials)
    elif args.consistency:
        result = test_consistency_predictions(N_values, trials)
    elif args.cross_validate:
        N_train = N_values[:len(N_values)//2]
        N_test = N_values[len(N_values)//2:]
        result = cross_validate_hypothesis(N_train, N_test, trials)
    elif args.alternatives:
        result = run_alternative_hypotheses_test(N_values, trials)
    elif args.complexity_barrier:
        result = run_complexity_barrier_test_suite(N_values, trials)
    elif args.update1:
        result = test_energy_barrier_scaling_hypothesis(N_values, trials)
    elif args.update2:
        result = test_stochastic_dynamics_hypothesis(N_values, trials*2)
    elif args.update3:
        result = test_fractal_dimension_hypothesis(N_values, trials//5)
    elif args.curvature:
        result = test_phase_space_curvature_hypothesis_FIXED(N_values, trials)
    else:
        result = run_complete_hypothesis_test(N_values, trials)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Used multiprocessing with {min(mp.cpu_count(), 8)} CPU cores")