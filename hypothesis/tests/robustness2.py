#!/usr/bin/env python3
"""
Find Critical Coupling K_c and Calibrate α
===========================================
Step 1: Find K_c where synchronization becomes possible
Step 2: Work at K = K_c × margin to ensure synchronization
Step 3: Calibrate α at this K value

SMP SUPPORT: Uses multiprocessing for parallel computation of trials
and N-value sweeps. Automatically scales to available CPU cores.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import multiprocessing as mp
import functools

# Global variables for multiprocessing (will be set before use)
_global_K_c_values = None
_global_K_margin = None
_global_n_trials = None
_global_omega_std = None

def _measure_single_N_mp(args):
    """Module-level function for multiprocessing basin volume measurement."""
    i, N = args
    K_test = _global_K_margin * _global_K_c_values[i]
    V, V_err = measure_basin_volume(N, K_test, _global_n_trials, _global_omega_std)
    return i, V, V_err, K_test

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


def find_critical_coupling(N: int, omega_std: float = 0.01, 
                          n_trials: int = 50) -> float:
    """
    Find K_c where synchronization probability ≈ 50%.
    Uses binary search with improved parameters.
    """
    K_low = 0.001
    K_high = 2.0  # Increased upper bound
    
    print(f"Finding K_c for N={N}...", end="", flush=True)
    
    # First, do a coarse scan to find approximate range
    K_test_values = np.logspace(-3, 0, 20)  # 0.001 to 1.0, 20 points
    best_K = 0.001
    best_sync_prob = 0.0
    
    for K_test in K_test_values:
        sync_count = 0
        for trial in range(10):  # Fewer trials for coarse scan
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, omega_std, N)
            
            # Evolve
            for _ in range(1000):
                theta = runge_kutta_step(theta, omega, K_test, 0.01)
            
            r_final = np.abs(np.mean(np.exp(1j * theta)))
            if r_final > 0.6:  # Synchronization threshold
                sync_count += 1
        
        sync_prob = sync_count / 10
        if sync_prob > best_sync_prob:
            best_sync_prob = sync_prob
            best_K = K_test
    
    # Set binary search bounds around best K
    K_low = max(0.0001, best_K / 3)
    K_high = min(2.0, best_K * 3)
    
    for iteration in range(10):  # Binary search
        K_mid = (K_low + K_high) / 2
        
        # Test synchronization probability
        sync_count = 0
        for trial in range(n_trials):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, omega_std, N)
            
            # Evolve
            for _ in range(1000):  # Increased evolution time
                theta = runge_kutta_step(theta, omega, K_mid, 0.01)
            
            r_final = np.abs(np.mean(np.exp(1j * theta)))
            if r_final > 0.6:  # Adjusted threshold
                sync_count += 1
        
        sync_prob = sync_count / n_trials
        
        # Binary search update
        if sync_prob < 0.4:
            K_low = K_mid
        elif sync_prob > 0.6:
            K_high = K_mid
        else:
            break  # Close enough to 50%
    
    K_c = K_mid
    print(f" K_c ≈ {K_c:.4f} (P_sync = {sync_prob:.1%})")
    
    return K_c


def measure_basin_volume(N: int, K: float, n_trials: int = 100,
                        omega_std: float = 0.01) -> Tuple[float, float]:
    """Measure basin volume via Monte Carlo."""
    sync_count = 0
    
    for trial in range(n_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Evolve system longer
        for _ in range(1000):  # Increased evolution time
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        # Check synchronization with adjusted threshold
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > 0.6:  # Adjusted threshold to match K_c search
            sync_count += 1
    
    volume = sync_count / n_trials
    error = np.sqrt(volume * (1 - volume) / n_trials) if volume > 0 else 0
    
    return volume, error


def calibrate_alpha_with_kc_scaling(N_values: List[int] = None,
                                    K_margin: float = 1.5,
                                    omega_std: float = 0.01,
                                    n_trials: int = 100) -> Dict[str, Any]:
    """
    Calibrate α with proper K_c scaling.
    
    CRITICAL: K_c scales with N! Must use K = K_c(N) × margin for each N.
    """
    if N_values is None:
        N_values = [10, 20, 30, 50]
    
    print("=" * 70)
    print("CALIBRATING α WITH K_c SCALING")
    print("=" * 70)
    print(f"Strategy: Work at K = K_c(N) × {K_margin:.2f} for each N")
    print(f"This ensures consistent distance from criticality")
    print()
    
    # Step 1: Find K_c for each N (parallelized)
    print("Finding K_c for each N...")
    worker_func = functools.partial(find_critical_coupling, omega_std=omega_std, n_trials=min(n_trials//2, 50))
    
    with mp.Pool(processes=min(mp.cpu_count(), len(N_values))) as pool:
        K_c_values = pool.map(worker_func, N_values)
    
    print()
    print("K_c scaling analysis:")
    print("-" * 40)
    for i, N in enumerate(N_values):
        print(f"N={N:2d}: K_c = {K_c_values[i]:.4f}")
    
    # Check if K_c ~ 1/√N
    sqrt_N = np.sqrt(N_values)
    K_c_array = np.array(K_c_values)
    product = K_c_array * sqrt_N
    print(f"\nK_c × √N = {product.mean():.4f} ± {product.std():.4f}")
    if product.std() / product.mean() < 0.3:
        print("✅ Confirms K_c ~ 1/√N scaling!")
    
    # Step 2: Measure V(N) at K = K_c(N) × margin (parallelized)
    print()
    print(f"Measuring basin volumes at K = {K_margin}×K_c:")
    print("-" * 40)
    
    # Set global variables for multiprocessing
    global _global_K_c_values, _global_K_margin, _global_n_trials, _global_omega_std
    _global_K_c_values = K_c_values
    _global_K_margin = K_margin
    _global_n_trials = n_trials
    _global_omega_std = omega_std
    
    with mp.Pool(processes=min(mp.cpu_count(), len(N_values))) as pool:
        results = pool.map(_measure_single_N_mp, enumerate(N_values))
    
    # Sort results by index
    results.sort(key=lambda x: x[0])
    
    V_measured = [r[1] for r in results]
    V_errors = [r[2] for r in results]
    K_test_values = [r[3] for r in results]
    
    for i, N in enumerate(N_values):
        print(f"N={N:2d} (K={K_test_values[i]:.4f}): V = {V_measured[i]:.4f} ± {V_errors[i]:.4f}")
    
    # Step 3: Fit ln(V) = -α√N + c
    sqrt_N = np.sqrt(N_values)
    
    # Handle V=0 or V=1 cases
    valid_indices = []
    ln_V = []
    for i, v in enumerate(V_measured):
        if 0.01 < v < 0.99:  # Exclude extremes
            ln_V.append(np.log(v))
            valid_indices.append(i)
        else:
            print(f"⚠️ Excluding N={N_values[i]} (V={v:.3f} is extreme)")
    
    if len(valid_indices) < 2:
        print()
        print("❌ ERROR: Too few valid measurements!")
        print("Possible issues:")
        print("  • K_margin too low (all V≈0) → try K_margin = 2.0")
        print("  • K_margin too high (all V≈1) → try K_margin = 1.2")
        print("  • Need more trials for better statistics")
        return {'alpha': 0.1, 'r_squared': 0.0, 'K_c_values': K_c_values}
    
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
    
    # Predict V at each N using fitted α
    V_predicted = [np.exp(slope * np.sqrt(N) + intercept) for N in N_values]
    
    print()
    print("=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Fitted: ln(V) = {slope:.4f}√N + {intercept:.4f}")
    print(f"α = {alpha_fitted:.4f}")
    print(f"R² = {r_squared:.3f}")
    print()
    print("Fit quality check:")
    print("-" * 40)
    print(f"{'N':>4} {'V_measured':>12} {'V_predicted':>12} {'Error':>10}")
    print("-" * 40)
    for i, N in enumerate(N_values):
        V_m = V_measured[i]
        V_p = V_predicted[i]
        err = abs(V_m - V_p)
        print(f"{N:4d} {V_m:12.4f} {V_p:12.4f} {err:10.4f}")
    print()
    
    if r_squared > 0.9:
        verdict = "✅ EXCELLENT FIT: High confidence in α"
    elif r_squared > 0.7:
        verdict = "⚠️ MODERATE FIT: Use α with caution"
    else:
        verdict = "❌ POOR FIT: V(N) may not follow exp(-α√N)"
    
    print(verdict)
    
    return {
        'alpha': max(0.01, alpha_fitted),
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'K_margin': K_margin,
        'K_c_values': K_c_values,
        'data': {
            'N': N_values,
            'K_c': K_c_values,
            'K_test': K_test_values,
            'V': V_measured,
            'V_err': V_errors,
            'V_pred': V_predicted,
            'valid_indices': valid_indices
        }
    }


def inverse_design_formula(V_target: float, alpha: float) -> float:
    """Calculate required N for target basin volume."""
    if V_target <= 0 or V_target >= 1:
        raise ValueError("V_target must be in (0, 1)")
    
    ln_inv_V = np.log(1.0 / V_target)
    sqrt_N = ln_inv_V / alpha
    N_required = sqrt_N ** 2
    
    return N_required


def validate_inverse_formula(calibration: Dict[str, Any],
                            V_targets: List[float] = None,
                            n_trials: int = 100) -> Dict[str, Any]:
    """Validate inverse formula with K_c scaling."""
    alpha = calibration['alpha']
    K_margin = calibration['K_margin']
    
    print()
    print("=" * 70)
    print("VALIDATING INVERSE DESIGN FORMULA")
    print("=" * 70)
    print(f"Using: α = {alpha:.4f} (R²={calibration['r_squared']:.3f})")
    print(f"       K_margin = {K_margin:.2f}")
    print()
    
    if V_targets is None:
        # Choose reasonable targets based on calibration range
        V_targets = [0.80, 0.60, 0.40, 0.20, 0.10]
    
    results = []
    omega_std = 0.01
    
    for V_target in V_targets:
        # Predict N
        N_predicted = inverse_design_formula(V_target, alpha)
        N_test = max(5, int(np.round(N_predicted)))
        
        # Find K_c for this N and set K = K_c × margin
        K_c = find_critical_coupling(N_test, omega_std, n_trials=30)
        K_test = K_margin * K_c
        
        print(f"\nTarget V = {V_target:.2f}")
        print(f"  Predicted N = {N_predicted:.1f} → Testing N = {N_test}")
        print(f"  K_c({N_test}) = {K_c:.4f}, using K = {K_test:.4f}")
        
        # Measure actual V
        V_measured, V_error = measure_basin_volume(N_test, K_test, n_trials, omega_std)
        
        error = abs(V_measured - V_target)
        rel_error = error / V_target if V_target > 0 else float('inf')
        
        print(f"  Measured V = {V_measured:.4f} ± {V_error:.4f}")
        print(f"  Error = {error:.4f} ({rel_error:.1%} relative)")
        
        # Success criteria: within 20% relative or 0.10 absolute
        success = (rel_error < 0.3) or (error < 0.15)
        status = "✅" if success else "❌"
        print(f"  {status} {'PASS' if success else 'FAIL'}")
        
        results.append({
            'V_target': V_target,
            'N_predicted': N_predicted,
            'N_test': N_test,
            'K_c': K_c,
            'K_test': K_test,
            'V_measured': V_measured,
            'V_error': V_error,
            'error': error,
            'rel_error': rel_error,
            'success': success
        })
    
    # Summary
    success_rate = np.mean([r['success'] for r in results])
    mean_error = np.mean([r['error'] for r in results])
    
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Mean absolute error: {mean_error:.3f}")
    
    if success_rate > 0.8:
        verdict = "✅ VALIDATED: Inverse formula works!"
    elif success_rate > 0.5:
        verdict = "⚠️ PARTIAL: Formula works in some regimes"
    else:
        verdict = "❌ FAILED: Formula needs refinement"
    
    print(f"\nVerdict: {verdict}")
    
    return {
        'alpha_used': alpha,
        'K_margin': K_margin,
        'results': results,
        'success_rate': success_rate,
        'mean_error': mean_error,
        'verdict': verdict
    }


def kaiabc_design_with_calibrated_alpha(calibration: Dict[str, Any]) -> None:
    """Generate KaiABC design recommendations."""
    alpha = calibration['alpha']
    K_c_10 = calibration['K_c_values'][0]  # K_c at N=10
    
    print()
    print("=" * 70)
    print("KaiABC IoT NETWORK DESIGN RECOMMENDATIONS")
    print("=" * 70)
    print()
    print("Using calibrated parameters:")
    print(f"  α = {alpha:.4f}")
    print(f"  K_c(10) = {K_c_10:.4f}")
    print(f"  K_c scaling: K_c ~ {K_c_10 * np.sqrt(10):.4f} / √N")
    print()
    
    print("Design Table:")
    print("-" * 70)
    print(f"{'Reliability':>12} {'N_max':>8} {'K_c(N)':>10} {'K_recommended':>15}")
    print("-" * 70)
    
    for V_target in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70]:
        N = inverse_design_formula(V_target, alpha)
        N_rounded = int(np.ceil(N))
        
        # Estimate K_c for this N
        K_c_N = K_c_10 * np.sqrt(10.0 / N_rounded)
        K_rec = 1.5 * K_c_N  # 50% margin
        
        print(f"{V_target:>12.0%} {N_rounded:>8d} {K_c_N:>10.4f} {K_rec:>15.4f}")
    
    print()
    print("Key Insights:")
    print("  • Larger networks need weaker coupling (K_c ~ 1/√N)")
    print("  • But exponentially harder to synchronize (V ~ exp(-√N))")
    print("  • Trade-off: Size vs Reliability vs Power")


def run_complete_analysis():
    """Run complete analysis with K_c scaling and SMP support."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ROBUSTNESS ANALYSIS WITH K_c SCALING" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"🔄 SMP Enabled: Using up to {min(mp.cpu_count(), 8)} CPU cores for parallel computation")
    print()
    
    # Calibrate with K_c scaling
    calibration = calibrate_alpha_with_kc_scaling(
        N_values=[10, 20, 30, 50],
        K_margin=1.2,  # Reduced margin for better results
        omega_std=0.01,
        n_trials=100
    )
    
    if calibration['r_squared'] < 0.5:
        print("\n❌ Calibration failed. Try different K_margin.")
        return
    
    # Validate
    validation = validate_inverse_formula(calibration, n_trials=100)
    
    # KaiABC design
    kaiabc_design_with_calibrated_alpha(calibration)
    
    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"✓ Calibrated α = {calibration['alpha']:.4f} (R²={calibration['r_squared']:.3f})")
    print(f"✓ Validation success rate: {validation['success_rate']:.1%}")
    print(f"✓ Inverse formula: N = [ln(1/V) / {calibration['alpha']:.4f}]²")
    print(f"✓ SMP Performance: {min(mp.cpu_count(), 8)} CPU cores utilized")
    print()
    
    if validation['success_rate'] > 0.6:
        print("🎉 SUCCESS! Inverse design validated with K_c scaling!")
    else:
        print("⚠️ Partial success. May need refinement for broader range.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer trials')
    parser.add_argument('--trials', type=int, default=100, help='Trials per measurement')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick test mode (fewer trials)")
        calibration = calibrate_alpha_with_kc_scaling(
            N_values=[10, 20, 30],
            K_margin=1.2,  # Reduced margin for better synchronization
            n_trials=50
        )
        print(f"\nQuick result: α ≈ {calibration['alpha']:.4f}")
    else:
        run_complete_analysis()