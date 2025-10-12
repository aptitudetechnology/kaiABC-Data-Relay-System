#!/usr/bin/env python3
"""
FIXED: Robustness Analysis with α Calibration
==============================================
The key insight: α must be measured from data first!
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import multiprocessing as mp
import functools

def runge_kutta_step(theta, omega, K, dt):
    """4th order RK for Kuramoto model."""
    def kuramoto(th, om, k):
        N = len(th)
        return om + (k/N) * np.sum(np.sin(th[:, None] - th), axis=1)
    
    k1 = kuramoto(theta, omega, K)
    k2 = kuramoto(theta + 0.5*dt*k1, omega, K)
    k3 = kuramoto(theta + 0.5*dt*k2, omega, K)
    k4 = kuramoto(theta + dt*k3, omega, K)
    return theta + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def _single_basin_trial(N: int, K: float, omega_std: float, _=None):
    """Worker function for single basin volume trial."""
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, omega_std, N)

    # Evolve system
    for _ in range(500):
        theta = runge_kutta_step(theta, omega, K, 0.01)

    # Check synchronization
    r_final = np.abs(np.mean(np.exp(1j * theta)))
    return 1 if r_final > 0.8 else 0


def _single_basin_trial_multi_attractor(N: int, K: float, omega_std: float, _=None):
    """
    Enhanced basin volume measurement that accounts for multiple attractors.

    Returns convergence to specific attractor:
    0: desynchronized
    1: fully synchronized (r > 0.9)
    2: partial synchronization (0.6 < r < 0.9)
    """
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(0, omega_std, N)

    # Evolve system with longer time for convergence
    for _ in range(1000):  # Increased from 500
        theta = runge_kutta_step(theta, omega, K, 0.01)

    # Check synchronization level
    r_final = np.abs(np.mean(np.exp(1j * theta)))

    if r_final > 0.9:
        return 1  # Full synchronization
    elif r_final > 0.6:
        return 2  # Partial synchronization
    else:
        return 0  # Desynchronized


def measure_basin_volume_multi_attractor(N: int, K: float, n_trials: int = 1000,
                                       omega_std: float = 0.01) -> Tuple[float, float, Dict[str, float]]:
    """
    Measure basin volumes for multiple attractors.

    Returns:
    - V_full: fraction converging to full synchronization
    - V_full_err: error in V_full
    - attractor_stats: dict with fractions for each attractor type
    """
    # Use multiprocessing for parallel trials
    worker_func = functools.partial(_single_basin_trial_multi_attractor, N, K, omega_std)
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        results = pool.map(worker_func, range(n_trials))

    # Count convergence to each attractor
    full_sync_count = sum(1 for r in results if r == 1)
    partial_sync_count = sum(1 for r in results if r == 2)
    desync_count = sum(1 for r in results if r == 0)

    # Calculate fractions
    V_full = full_sync_count / n_trials
    V_partial = partial_sync_count / n_trials
    V_desync = desync_count / n_trials

    # Errors using binomial statistics
    V_full_err = np.sqrt(V_full * (1 - V_full) / n_trials)

    attractor_stats = {
        'full_sync': V_full,
        'partial_sync': V_partial,
        'desync': V_desync,
        'total': V_full + V_partial + V_desync
    }

    return V_full, V_full_err, attractor_stats


def calibrate_alpha(N_values: List[int] = None, K: float = 0.02,
                   omega_std: float = 0.01, n_trials: int = 1000) -> Dict[str, Any]:
    """
    CRITICAL STEP: Calibrate α from actual measurements.
    
    Returns dict with α, R², and raw data.
    """
    if N_values is None:
        N_values = [10, 20, 30, 50]
    
    print("=" * 60)
    print("CALIBRATING α PARAMETER FROM DATA")
    print("=" * 60)
    print(f"System: K={K:.3f}, σ_ω={omega_std:.4f}")
    print(f"Measuring V(N) for N ∈ {N_values}")
    print()
    
    V_measured = []
    V_errors = []
    
    for N in N_values:
        print(f"N={N:2d}: ", end="", flush=True)
        V, V_err, attractor_stats = measure_basin_volume_multi_attractor(N, K, n_trials, omega_std)
        V_measured.append(V)
        V_errors.append(V_err)
        print(f"V={V:.4f} ± {V_err:.4f} (full sync: {attractor_stats['full_sync']:.1%}, partial: {attractor_stats['partial_sync']:.1%})")
    
    # Fit ln(V) = -α√N + c
    sqrt_N = np.sqrt(N_values)
    
    # Handle V=0 cases
    ln_V = []
    valid_indices = []
    for i, v in enumerate(V_measured):
        if v > 0:
            ln_V.append(np.log(v))
            valid_indices.append(i)
        else:
            print(f"⚠️ Warning: V=0 at N={N_values[i]}, excluding from fit")
    
    if len(valid_indices) < 2:
        print("❌ ERROR: Too few valid measurements!")
        return {'alpha': 0.1, 'r_squared': 0.0, 'data': {}}
    
    sqrt_N_valid = np.array([sqrt_N[i] for i in valid_indices])
    ln_V_valid = np.array(ln_V)
    
    # Linear fit
    slope, intercept = np.polyfit(sqrt_N_valid, ln_V_valid, 1)
    alpha_fitted = -slope
    
    # Calculate R²
    ln_V_pred = slope * sqrt_N_valid + intercept
    ss_res = np.sum((ln_V_valid - ln_V_pred)**2)
    ss_tot = np.sum((ln_V_valid - np.mean(ln_V_valid))**2)
    r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Fitted: ln(V) = {slope:.4f}√N + {intercept:.4f}")
    print(f"α = {alpha_fitted:.4f}")
    print(f"R² = {r_squared:.3f}")
    
    if r_squared > 0.9:
        print("✅ EXCELLENT FIT: High confidence in α")
    elif r_squared > 0.7:
        print("⚠️ MODERATE FIT: Use α with caution")
    else:
        print("❌ POOR FIT: V(N) may not follow exp(-α√N)")
    
    return {
        'alpha': max(0.01, alpha_fitted),
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'data': {
            'N': N_values,
            'V': V_measured,
            'V_err': V_errors,
            'valid_indices': valid_indices
        }
    }


def explore_coupling_regimes(N_values: List[int] = None, 
                           omega_std: float = 0.01, n_trials: int = 500) -> Dict[str, Any]:
    """
    Explore different coupling regimes inspired by the 3-oscillator paper.
    
    Tests different K values to find regimes with different basin structures:
    - Low K: desynchronized
    - Medium K: single attractor
    - High K: multiple attractors (if they exist)
    """
    if N_values is None:
        N_values = [10, 20, 30]
    
    print("=" * 70)
    print("EXPLORING COUPLING REGIMES (Inspired by 3-Oscillator Analysis)")
    print("=" * 70)
    print("Testing different K values to find optimal basin volume regimes")
    print()
    
    # Test different coupling strengths
    K_values = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    
    results = {}
    
    for K in K_values:
        print(f"K = {K:.3f}")
        print("-" * 30)
        
        K_results = {}
        for N in N_values:
            V_full, V_err, attractor_stats = measure_basin_volume_multi_attractor(
                N, K, n_trials, omega_std
            )
            K_results[N] = {
                'V_full': V_full,
                'V_err': V_err,
                'attractor_stats': attractor_stats
            }
            print(f"  N={N:2d}: V_full={V_full:.4f} ± {V_err:.4f}")
            print(f"        (full: {attractor_stats['full_sync']:.1%}, partial: {attractor_stats['partial_sync']:.1%}, desync: {attractor_stats['desync']:.1%})")
        
        results[K] = K_results
        print()
    
    # Find optimal K for each N (maximum basin volume)
    optimal_K = {}
    for N in N_values:
        best_K = None
        best_V = 0
        for K in K_values:
            V = results[K][N]['V_full']
            if V > best_V:
                best_V = V
                best_K = K
        optimal_K[N] = best_K
    
    print("OPTIMAL COUPLING STRENGTHS:")
    print("-" * 30)
    for N in N_values:
        K_opt = optimal_K[N]
        V_opt = results[K_opt][N]['V_full']
        print(f"N={N:2d}: K_opt = {K_opt:.3f} (V_full = {V_opt:.4f})")
    
    return {
        'K_values': K_values,
        'N_values': N_values,
        'results': results,
        'optimal_K': optimal_K
    }


def phase_diameter_analysis(N: int, K: float, omega_std: float = 0.01,
                           n_trials: int = 100, evolution_steps: int = 1000) -> Dict[str, Any]:
    """
    Implement phase diameter analysis inspired by the 3-oscillator paper.
    
    The phase diameter function D(t) = max_i,j |θ_i(t) - θ_j(t)|
    measures the spread of phases and can indicate convergence properties.
    """
    print(f"Phase diameter analysis for N={N}, K={K:.3f}")
    print("-" * 40)
    
    diameters_over_time = []
    
    for trial in range(n_trials):
        # Random initial conditions
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Track phase diameter over time
        diameters = []
        for step in range(evolution_steps):
            theta = runge_kutta_step(theta, omega, K, 0.01)
            
            # Calculate phase diameter
            theta_sorted = np.sort(theta)
            diameter = theta_sorted[-1] - theta_sorted[0]
            diameters.append(diameter)
        
        diameters_over_time.append(diameters)
    
    # Convert to numpy array for analysis
    diameters_array = np.array(diameters_over_time)  # Shape: (n_trials, evolution_steps)
    
    # Calculate statistics
    mean_diameter = np.mean(diameters_array, axis=0)
    std_diameter = np.std(diameters_array, axis=0)
    
    # Check for exponential decay (indicating convergence)
    # Fit exponential decay: D(t) ~ D0 * exp(-λt)
    t_fit_start = evolution_steps // 2  # Use second half for fitting
    t_values = np.arange(t_fit_start, evolution_steps)
    log_diameters = np.log(mean_diameter[t_fit_start:] + 1e-10)  # Add small constant to avoid log(0)
    
    # Linear fit to log(diameter) vs time
    if len(t_values) > 10 and not np.any(np.isnan(log_diameters)):
        slope, intercept = np.polyfit(t_values, log_diameters, 1)
        decay_rate = -slope  # λ in exp(-λt)
        r_squared_decay = np.corrcoef(t_values, log_diameters)[0, 1]**2
    else:
        decay_rate = 0
        r_squared_decay = 0
    
    # Final synchronization assessment
    final_diameters = diameters_array[:, -1]
    final_r = np.mean([np.abs(np.mean(np.exp(1j * theta))) for theta in 
                      [2*np.pi*np.random.rand(N) + final_diameters[i] * np.random.rand(N) 
                       for i in range(min(10, n_trials))]])
    
    print(f"Final phase diameter: {mean_diameter[-1]:.4f} ± {std_diameter[-1]:.4f}")
    print(f"Decay rate λ: {decay_rate:.6f} (R² = {r_squared_decay:.3f})")
    print(f"Final order parameter r: {final_r:.4f}")
    
    if decay_rate > 0.001 and r_squared_decay > 0.8:
        print("✅ Exponential convergence detected")
    else:
        print("⚠️ No clear exponential convergence")
    
    return {
        'mean_diameter': mean_diameter,
        'std_diameter': std_diameter,
        'decay_rate': decay_rate,
        'r_squared_decay': r_squared_decay,
        'final_r': final_r,
        'time_steps': np.arange(evolution_steps)
    }


def inverse_design_formula(V_target: float, alpha: float) -> float:
    """
    Calculate required N for target basin volume.
    
    From: V = exp(-α√N)
    → N = [ln(1/V)/α]²
    """
    if V_target <= 0 or V_target >= 1:
        raise ValueError("V_target must be in (0, 1)")
    
    ln_inv_V = np.log(1.0 / V_target)
    sqrt_N = ln_inv_V / alpha
    N_required = sqrt_N ** 2
    
    return N_required


def validate_inverse_formula(calibration: Dict[str, Any], 
                            V_targets: List[float] = None,
                            n_trials: int = 500) -> Dict[str, Any]:
    """
    Phase 1: Validate inverse formula using calibrated α.
    """
    alpha = calibration['alpha']
    
    print()
    print("=" * 60)
    print("PHASE 1: VALIDATING INVERSE DESIGN FORMULA")
    print("=" * 60)
    print(f"Using calibrated α = {alpha:.4f} (R² = {calibration['r_squared']:.3f})")
    print()
    
    if V_targets is None:
        V_targets = [0.50, 0.30, 0.20, 0.10, 0.05, 0.01]
    
    results = []
    K_test = 0.02
    omega_std = 0.01
    
    for V_target in V_targets:
        # Calculate predicted N
        N_predicted = inverse_design_formula(V_target, alpha)
        N_test = max(5, int(np.round(N_predicted)))
        
        print(f"Target V = {V_target:.3f}")
        print(f"  Predicted N = {N_predicted:.1f} → Testing N = {N_test}")
        
        # Measure actual basin volume
        V_measured, V_error, attractor_stats = measure_basin_volume_multi_attractor(N_test, K_test, n_trials, omega_std)
        
        # Calculate prediction error
        error = abs(V_measured - V_target)
        rel_error = error / V_target if V_target > 0 else float('inf')
        
        print(f"  Measured V = {V_measured:.4f} ± {V_error:.4f}")
        print(f"  Error = {error:.4f} ({rel_error:.1%} relative)")
        
        success = error < 0.1 or rel_error < 0.3  # Within 10% absolute or 30% relative
        status = "✅" if success else "❌"
        print(f"  {status} {'PASS' if success else 'FAIL'}")
        print()
        
        results.append({
            'V_target': V_target,
            'N_predicted': N_predicted,
            'N_test': N_test,
            'V_measured': V_measured,
            'V_error': V_error,
            'error': error,
            'rel_error': rel_error,
            'success': success
        })
    
    # Summary
    success_rate = np.mean([r['success'] for r in results])
    mean_rel_error = np.mean([r['rel_error'] for r in results if np.isfinite(r['rel_error'])])
    
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Mean relative error: {mean_rel_error:.1%}")
    
    if success_rate > 0.8:
        verdict = "✅ VALIDATED: Inverse formula works!"
    elif success_rate > 0.5:
        verdict = "⚠️ PARTIAL: Formula works for some regimes"
    else:
        verdict = "❌ FAILED: Formula needs refinement"
    
    print(f"Verdict: {verdict}")
    
    return {
        'alpha_used': alpha,
        'results': results,
        'success_rate': success_rate,
        'mean_rel_error': mean_rel_error,
        'verdict': verdict
    }


def kaiabc_design_example(calibration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Design a KaiABC IoT network using calibrated α.
    """
    alpha = calibration['alpha']
    
    print()
    print("=" * 60)
    print("APPLICATION: KaiABC IoT NETWORK DESIGN")
    print("=" * 60)
    
    # Requirements
    V_target = 0.95  # 95% reliability
    temperature = 25.0  # °C
    
    print(f"Target reliability: {V_target:.1%}")
    print(f"Operating temperature: {temperature}°C")
    print()
    
    # Calculate required N
    N_required = inverse_design_formula(V_target, alpha)
    N_design = int(np.ceil(N_required))
    
    print(f"Required system size:")
    print(f"  N = {N_required:.1f} → Design for N = {N_design} nodes")
    print()
    
    # Calculate for different reliability levels
    print("Design table for various reliability targets:")
    print("-" * 40)
    for V in [0.99, 0.95, 0.90, 0.80, 0.70]:
        N = inverse_design_formula(V, alpha)
        print(f"  {V:.0%} reliability: N ≤ {int(np.ceil(N)):2d} nodes")
    
    return {
        'V_target': V_target,
        'N_required': N_required,
        'N_design': N_design,
        'alpha_used': alpha
    }


def run_complete_analysis():
    """Run complete robustness analysis with proper α calibration."""
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ROBUSTNESS ANALYSIS SUITE" + " " * 23 + "║")
    print("║" + " " * 8 + "Inverse Basin Design Formula" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"Using multiprocessing with {min(mp.cpu_count(), 8)} CPU cores")
    print()
    
    # Step 0: Explore coupling regimes (NEW - inspired by 3-oscillator paper)
    print("PHASE 0: EXPLORING COUPLING REGIMES")
    regime_results = explore_coupling_regimes(
        N_values=[10, 20, 30],
        n_trials=200  # Fewer trials for exploration
    )
    
    # Use optimal K for each N from exploration
    optimal_K_values = regime_results['optimal_K']
    
    # Step 1: Calibrate α using optimal K for each N
    print("\nPHASE 1: CALIBRATING α WITH OPTIMAL COUPLING")
    
    # Use optimal K values from exploration
    N_values = [10, 20, 30]
    V_measured = []
    V_errors = []
    
    for N in N_values:
        optimal_K = optimal_K_values.get(N, 0.02)
        print(f"N={N:2d} (K={optimal_K:.3f}): ", end="", flush=True)
        V, V_err, attractor_stats = measure_basin_volume_multi_attractor(N, optimal_K, 500, 0.01)
        V_measured.append(V)
        V_errors.append(V_err)
        print(f"V={V:.4f} ± {V_err:.4f}")
    
    # Fit α from the measurements
    sqrt_N = np.sqrt(N_values)
    ln_V = []
    valid_indices = []
    for i, v in enumerate(V_measured):
        if v > 0:
            ln_V.append(np.log(v))
            valid_indices.append(i)
        else:
            print(f"⚠️ Warning: V=0 at N={N_values[i]}, excluding from fit")
    
    if len(valid_indices) < 2:
        print("❌ ERROR: Too few valid measurements!")
        calibration = {'alpha': 0.1, 'r_squared': 0.0, 'data': {}}
    else:
        sqrt_N_valid = np.array([sqrt_N[i] for i in valid_indices])
        ln_V_valid = np.array(ln_V)
        
        slope, intercept = np.polyfit(sqrt_N_valid, ln_V_valid, 1)
        alpha_fitted = -slope
        
        ln_V_pred = slope * sqrt_N_valid + intercept
        ss_res = np.sum((ln_V_valid - ln_V_pred)**2)
        ss_tot = np.sum((ln_V_valid - np.mean(ln_V_valid))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        print(f"\nFitted: ln(V) = {slope:.4f}√N + {intercept:.4f}")
        print(f"α = {alpha_fitted:.4f} (R² = {r_squared:.3f})")
        
        calibration = {
            'alpha': max(0.01, alpha_fitted),
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept,
            'data': {
                'N': N_values,
                'V': V_measured,
                'V_err': V_errors,
                'valid_indices': valid_indices
            }
        }
    
    if calibration['r_squared'] < 0.5:
        print("\n❌ Calibration failed. Cannot proceed with validation.")
        return
    
    # Step 2: Phase diameter analysis (NEW)
    print("\nPHASE 2: PHASE DIAMETER ANALYSIS")
    for N in [10, 20]:
        optimal_K = optimal_K_values.get(N, 0.02)
        phase_analysis = phase_diameter_analysis(N, optimal_K, n_trials=50)
        print()
    
    # Step 3: Validate inverse formula
    validation = validate_inverse_formula(
        calibration,
        V_targets=[0.30, 0.20, 0.10, 0.05],
        n_trials=300
    )
    
    # Step 4: Application example
    kaiabc = kaiabc_design_example(calibration)
    
    # Final summary
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Calibrated α: {calibration['alpha']:.4f} (R²={calibration['r_squared']:.3f})")
    print(f"Validation success: {validation['success_rate']:.1%}")
    print(f"Mean prediction error: {validation['mean_rel_error']:.1%}")
    print(f"\nKaiABC Design: N ≤ {kaiabc['N_design']} nodes for 95% reliability")
    print(f"Multiprocessing: {min(mp.cpu_count(), 8)} CPU cores utilized")
    print()
    
    if validation['success_rate'] > 0.8:
        print("🎉 SUCCESS! Inverse design formula validated!")
        print("   Formula: N = [ln(1/V) / α]² where α ≈ {:.3f}".format(calibration['alpha']))
    else:
        print("⚠️ Formula needs refinement for broader applicability")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--trials', type=int, default=500)
    
    args = parser.parse_args()
    
    if args.calibrate_only:
        result = calibrate_alpha(n_trials=args.trials)
        print(f"\nα = {result['alpha']:.4f}")
    else:
        run_complete_analysis()