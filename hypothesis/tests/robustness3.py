#!/usr/bin/env python3
"""
Bootstrap Calibration from Known Working Point
===============================================
Strategy: Start from a known synchronizing configuration,
then scale to other N values using K_c ~ 1/√N prediction.
"""

import numpy as np
from typing import Tuple, Dict, Any, List

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


def test_synchronization_at_K(N: int, K: float, n_trials: int = 50,
                               omega_std: float = 0.01,
                               sync_threshold: float = 0.5) -> float:
    """
    Test what fraction of trials synchronize at given K.
    
    Returns: sync_probability (0 to 1)
    """
    sync_count = 0
    
    for trial in range(n_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Evolve for long time
        for step in range(2000):
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        # Check synchronization
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > sync_threshold:
            sync_count += 1
    
    return sync_count / n_trials


def find_working_K_for_N10(omega_std: float = 0.01) -> Tuple[float, float]:
    """
    Find a K value that works well for N=10.
    This is our bootstrap anchor point.
    """
    print("=" * 70)
    print("STEP 1: FINDING WORKING K FOR N=10 (BOOTSTRAP ANCHOR)")
    print("=" * 70)
    print()
    
    N = 10
    
    # Try a range of K values
    K_test_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    print("Testing K values:")
    print("-" * 40)
    
    results = []
    for K in K_test_values:
        sync_prob = test_synchronization_at_K(N, K, n_trials=50, omega_std=omega_std)
        results.append((K, sync_prob))
        print(f"K = {K:.3f}: P_sync = {sync_prob:.1%}")
    
    # Find K where P_sync ≈ 0.8-0.9 (high but not perfect)
    best_K = None
    best_sync = 0.0
    for K, prob in results:
        if 0.7 <= prob <= 0.95 and prob > best_sync:
            best_K = K
            best_sync = prob
    
    if best_K is None:
        # Fallback: use highest sync probability
        best_K, best_sync = max(results, key=lambda x: x[1])
    
    print()
    print(f"✅ Selected K = {best_K:.3f} (P_sync = {best_sync:.1%})")
    print(f"   This will be our reference point K_ref at N_ref = 10")
    
    return best_K, best_sync


def scale_K_for_N(K_ref: float, N_ref: int, N_target: int) -> float:
    """
    Scale K using K_c ~ 1/√N relationship.
    
    K(N_target) = K_ref × √(N_ref / N_target)
    """
    scaling_factor = np.sqrt(N_ref / N_target)
    K_scaled = K_ref * scaling_factor
    return K_scaled


def measure_basin_volume(N: int, K: float, n_trials: int = 100,
                        omega_std: float = 0.01,
                        sync_threshold: float = 0.5) -> Tuple[float, float]:
    """Measure basin volume fraction."""
    sync_count = 0
    
    for trial in range(n_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Long evolution
        for step in range(2000):
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > sync_threshold:
            sync_count += 1
    
    volume = sync_count / n_trials
    error = np.sqrt(volume * (1 - volume) / n_trials) if volume > 0 else 0
    
    return volume, error


def calibrate_alpha_bootstrap(K_ref: float = None,
                              N_values: List[int] = None,
                              omega_std: float = 0.01,
                              n_trials: int = 100) -> Dict[str, Any]:
    """
    Calibrate α using bootstrap from known working point.
    """
    if N_values is None:
        N_values = [10, 20, 30, 50]
    
    N_ref = 10  # Reference system size
    
    # If K_ref not provided, find it
    if K_ref is None:
        K_ref, _ = find_working_K_for_N10(omega_std)
    else:
        print(f"Using provided K_ref = {K_ref:.3f} at N_ref = {N_ref}")
    
    print()
    print("=" * 70)
    print("STEP 2: MEASURING BASIN VOLUMES WITH SCALED K")
    print("=" * 70)
    print(f"Using K_c scaling: K(N) = {K_ref:.3f} × √(10/N)")
    print()
    
    K_values = []
    V_measured = []
    V_errors = []
    
    for N in N_values:
        # Scale K for this N
        K_test = scale_K_for_N(K_ref, N_ref, N)
        K_values.append(K_test)
        
        print(f"N={N:2d} (K={K_test:.4f}): ", end="", flush=True)
        
        # Measure basin volume
        V, V_err = measure_basin_volume(N, K_test, n_trials, omega_std)
        V_measured.append(V)
        V_errors.append(V_err)
        
        print(f"V = {V:.4f} ± {V_err:.4f}")
    
    print()
    print("=" * 70)
    print("STEP 3: FITTING α FROM V(N) ~ exp(-α√N)")
    print("=" * 70)
    
    # Filter valid measurements
    valid_indices = []
    ln_V = []
    for i, v in enumerate(V_measured):
        if 0.01 < v < 0.99:
            ln_V.append(np.log(v))
            valid_indices.append(i)
        else:
            print(f"⚠️ Excluding N={N_values[i]} (V={v:.4f} is extreme)")
    
    if len(valid_indices) < 2:
        print()
        print("❌ ERROR: Not enough valid measurements to fit!")
        print(f"   Only {len(valid_indices)} valid points")
        print()
        print("Diagnostics:")
        for i, (N, V, K) in enumerate(zip(N_values, V_measured, K_values)):
            status = "✓" if 0.01 < V < 0.99 else "✗"
            print(f"  {status} N={N}: V={V:.4f} at K={K:.4f}")
        
        return {
            'success': False,
            'alpha': 0.1,
            'r_squared': 0.0,
            'K_ref': K_ref,
            'data': {'N': N_values, 'K': K_values, 'V': V_measured}
        }
    
    # Fit ln(V) = -α√N + c
    sqrt_N_valid = np.array([np.sqrt(N_values[i]) for i in valid_indices])
    ln_V_valid = np.array(ln_V)
    
    slope, intercept = np.polyfit(sqrt_N_valid, ln_V_valid, 1)
    alpha_fitted = -slope
    
    # Calculate R²
    ln_V_pred = slope * sqrt_N_valid + intercept
    ss_res = np.sum((ln_V_valid - ln_V_pred)**2)
    ss_tot = np.sum((ln_V_valid - np.mean(ln_V_valid))**2)
    r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    # Predict V for all N
    V_predicted = [np.exp(slope * np.sqrt(N) + intercept) for N in N_values]
    
    print()
    print(f"Fit: ln(V) = {slope:.4f}√N + {intercept:.4f}")
    print(f"α = {alpha_fitted:.4f}")
    print(f"R² = {r_squared:.3f}")
    print()
    
    print("Fit quality:")
    print("-" * 50)
    print(f"{'N':>4} {'K':>8} {'V_meas':>10} {'V_pred':>10} {'Error':>10}")
    print("-" * 50)
    for i, N in enumerate(N_values):
        print(f"{N:4d} {K_values[i]:8.4f} {V_measured[i]:10.4f} "
              f"{V_predicted[i]:10.4f} {abs(V_measured[i]-V_predicted[i]):10.4f}")
    print()
    
    if r_squared > 0.8:
        verdict = "✅ EXCELLENT FIT"
    elif r_squared > 0.6:
        verdict = "⚠️ MODERATE FIT"
    else:
        verdict = "❌ POOR FIT"
    
    print(f"Verdict: {verdict} (R² = {r_squared:.3f})")
    
    return {
        'success': r_squared > 0.5,
        'alpha': max(0.01, alpha_fitted),
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'K_ref': K_ref,
        'N_ref': N_ref,
        'data': {
            'N': N_values,
            'K': K_values,
            'V': V_measured,
            'V_err': V_errors,
            'V_pred': V_predicted,
            'valid_indices': valid_indices
        }
    }


def inverse_design(V_target: float, alpha: float) -> float:
    """N = [ln(1/V) / α]²"""
    return (np.log(1.0 / V_target) / alpha) ** 2


def generate_design_table(calibration: Dict[str, Any]) -> None:
    """Generate practical design recommendations."""
    if not calibration['success']:
        print("\n❌ Cannot generate design table - calibration failed")
        return
    
    alpha = calibration['alpha']
    K_ref = calibration['K_ref']
    N_ref = calibration['N_ref']
    
    print()
    print("=" * 70)
    print("DESIGN RECOMMENDATIONS")
    print("=" * 70)
    print()
    print(f"Calibrated parameters:")
    print(f"  α = {alpha:.4f}")
    print(f"  K_ref = {K_ref:.4f} (at N={N_ref})")
    print(f"  Scaling: K(N) = {K_ref:.4f} × √({N_ref}/N)")
    print()
    
    print("Design Table for KaiABC IoT Networks:")
    print("-" * 70)
    print(f"{'Target':>10} {'Max N':>8} {'Required K':>12} {'Notes':>30}")
    print("-" * 70)
    
    for V_target in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        N = inverse_design(V_target, alpha)
        N_int = int(np.ceil(N))
        K = scale_K_for_N(K_ref, N_ref, N_int)
        
        if N_int <= 5:
            note = "Very small network"
        elif N_int <= 20:
            note = "Practical size"
        elif N_int <= 50:
            note = "Large network"
        else:
            note = "May be challenging"
        
        print(f"{V_target:>10.0%} {N_int:>8d} {K:>12.4f} {note:>30}")
    
    print()
    print("Usage:")
    print("  1. Choose target reliability (e.g., 95%)")
    print("  2. Read maximum N from table")
    print("  3. Use specified K value for coupling strength")
    print("  4. Monitor actual sync rate and adjust if needed")


def quick_validation(calibration: Dict[str, Any], n_trials: int = 50) -> None:
    """Quick validation of inverse formula."""
    if not calibration['success']:
        return
    
    alpha = calibration['alpha']
    K_ref = calibration['K_ref']
    N_ref = calibration['N_ref']
    
    print()
    print("=" * 70)
    print("QUICK VALIDATION")
    print("=" * 70)
    
    # Test one prediction
    V_target = 0.70
    N_pred = inverse_design(V_target, alpha)
    N_test = int(np.round(N_pred))
    K_test = scale_K_for_N(K_ref, N_ref, N_test)
    
    print(f"\nTest: Target V = {V_target:.0%}")
    print(f"  Predicted N = {N_pred:.1f} → Testing N = {N_test}")
    print(f"  Using K = {K_test:.4f}")
    print(f"  Measuring... ", end="", flush=True)
    
    V_meas, V_err = measure_basin_volume(N_test, K_test, n_trials)
    
    print(f"V = {V_meas:.3f} ± {V_err:.3f}")
    
    error = abs(V_meas - V_target)
    if error < 0.15:
        print(f"  ✅ PASS: Within tolerance (error = {error:.3f})")
    else:
        print(f"  ⚠️ Miss: error = {error:.3f}")


def run_bootstrap_analysis(quick: bool = False):
    """Run complete bootstrap analysis."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "BOOTSTRAP CALIBRATION FROM WORKING POINT" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    if quick:
        N_values = [10, 20, 30]
        n_trials = 50
        print("Quick mode: 3 N values, 50 trials each")
    else:
        N_values = [10, 20, 30, 50]
        n_trials = 100
        print("Full mode: 4 N values, 100 trials each")
    
    print()
    
    # Calibrate
    calibration = calibrate_alpha_bootstrap(
        K_ref=None,  # Will find automatically
        N_values=N_values,
        omega_std=0.01,
        n_trials=n_trials
    )
    
    if not calibration['success']:
        print("\n❌ Calibration failed!")
        return
    
    # Generate design table
    generate_design_table(calibration)
    
    # Quick validation
    if not quick:
        quick_validation(calibration, n_trials=50)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Bootstrap succeeded with α = {calibration['alpha']:.4f}")
    print(f"✓ Fit quality: R² = {calibration['r_squared']:.3f}")
    print(f"✓ Inverse formula: N = [ln(1/V) / {calibration['alpha']:.4f}]²")
    print()
    
    if calibration['r_squared'] > 0.8:
        print("🎉 EXCELLENT! High confidence in design predictions.")
    elif calibration['r_squared'] > 0.6:
        print("✅ GOOD! Design predictions should be reliable.")
    else:
        print("⚠️ MODERATE. Use predictions with some caution.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--K', type=float, help='Override K_ref value')
    
    args = parser.parse_args()
    
    if args.K:
        print(f"Using provided K_ref = {args.K:.3f}")
        cal = calibrate_alpha_bootstrap(K_ref=args.K, n_trials=50 if args.quick else 100)
        if cal['success']:
            generate_design_table(cal)
    else:
        run_bootstrap_analysis(quick=args.quick)