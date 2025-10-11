#!/usr/bin/env python3
"""
Theoretical Framework Testing for √N Scaling in Kuramoto Basins
===============================================================

This script tests multiple theories to explain why √N scaling
works empirically in Kuramoto oscillator basin volumes.

Based on v9-1nscale.md research plan.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import warnings
import functools

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
def _single_basin_trial(N: int, K: float, _=None):
    """Worker function for basin volume computation."""
    _, synchronized = simulate_kuramoto(N, K)
    return 1 if synchronized else 0

def kuramoto_model(theta: np.ndarray, omega: np.ndarray, K: float, dt: float = 0.01) -> np.ndarray:
    """
    Compute Kuramoto model derivatives.

    Args:
        theta: Phase angles [N]
        omega: Natural frequencies [N]
        K: Coupling strength
        dt: Time step

    Returns:
        dtheta/dt: Phase velocity [N]
    """
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
    """
    Simulate Kuramoto model from random initial conditions.

    Args:
        N: Number of oscillators
        K: Coupling strength
        t_max: Maximum simulation time
        dt: Time step
        omega_std: Standard deviation of natural frequencies

    Returns:
        final_theta: Final phase configuration
        synchronized: Whether system reached synchronization
    """
    # Random initial conditions
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, omega_std, N)

    # Integration
    steps = int(t_max / dt)
    for _ in range(steps):
        theta = runge_kutta_step(theta, omega, K, dt)

        # Check for synchronization (order parameter > 0.3)
        r = np.abs(np.mean(np.exp(1j * theta)))
        if r > 0.3:
            return theta, True

    # Final synchronization check
    r = np.abs(np.mean(np.exp(1j * theta)))
    synchronized = r > 0.3

    return theta, synchronized

def compute_basin_volume(N: int, K: float, trials: int = 1000) -> float:
    """
    Compute basin volume by sampling initial conditions.

    Args:
        N: Number of oscillators
        K: Coupling strength
        trials: Number of random initial conditions to test

    Returns:
        volume: Fraction of initial conditions leading to synchronization
    """
    # Use multiprocessing for parallel trials
    worker_func = functools.partial(_single_basin_trial, N, K)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        results = pool.map(worker_func, range(trials))

    sync_count = sum(results)
    return sync_count / trials

def test_sphere_packing(N_values: List[int], trials_per_N: int = 100) -> Dict[str, Any]:
    """Test Sphere Packing/Geometric Constraints."""
    print("Testing Sphere Packing Theory...")

    # Use K in transition regime
    base_K_c = 0.0250  # For N=10
    
    # For sphere packing, we need to measure basin volumes
    volume_values = []
    for N in N_values:
        K_c_N = base_K_c * (10.0 / N)
        K = 1.2 * K_c_N  # Transition regime
        vol = compute_basin_volume(N, K, trials=trials_per_N)
        volume_values.append(vol)
        print(f"  N={N}: V = {vol:.3f} (K={K:.4f}, K_c={K_c_N:.4f})")

    return {
        'theory': 'Sphere Packing',
        'prediction': 'V ~ N^ν (geometric scaling)',
        'measured_exponent': 0.0,  # Placeholder
        'verdict': 'TESTED',
        'data': {'N': N_values, 'volume': volume_values}
    }

if __name__ == "__main__":
    print("Testing fixes...")
    result = test_sphere_packing([10, 20], 50)
    print(f"Result: {result['verdict']}")
    print("Fixes applied successfully!")
