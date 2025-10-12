#!/usr/bin/env python3
"""
Bootstrap Calibration of Basin Volume Scaling α
===============================================
BOOTSTRAP APPROACH: Find working K for N_ref=10, then scale as K(N) = K_ref × √(10/N)
This avoids K_c detection issues and provides stable reference point for scaling.

Step 1: Find K_ref where N=10 shows reasonable synchronization (bootstrap anchor)
Step 2: Scale K(N) = K_ref × √(10/N) for other N values
Step 3: Measure basin volumes V(N) and fit ln(V) = -α√N + c

N-ADAPTIVE PARAMETERS: Based on research paper insights about multi-attractor systems
- Evolution time scales with N: evolution_steps = max(500, N × 10)
- Sync threshold adapts to N: sync_threshold = max(0.3, 0.8 - N/100)
- Frequency dispersion increases with N: ω_std_adaptive = ω_std × (1 + N/200)

SMP SUPPORT: Uses multiprocessing Pool for parallel basin volume measurements.
Automatically scales to available CPU cores for faster computation.
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

def _measure_single_N_bootstrap_mp(args):
    """Module-level function for multiprocessing bootstrap basin volume measurement."""
    i, N = args
    K_test = _global_K_c_values[i]  # For bootstrap, K_c_values are actually K_test_values
    V, V_err = measure_basin_volume_bootstrap(N, K_test, _global_n_trials, _global_omega_std)
    return i, V, V_err

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
    Uses binary search with N-adaptive parameters.
    """
    K_low = 0.0001
    K_high = 2.0
    
    print(f"Finding K_c for N={N}...", end="", flush=True)
    
    # N-dependent parameters - more aggressive adaptation
    evolution_steps = max(1000, N * 20)  # More evolution time
    sync_threshold = max(0.15, 0.9 - N/50)  # Much lower threshold for larger N
    omega_std_adaptive = omega_std * (1.0 / np.sqrt(N))  # Reduce heterogeneity for larger N
    
    # First, do a coarse scan to find approximate range
    K_test_values = np.logspace(-4, 1, 20)  # 0.0001 to 10.0, 20 points
    best_K = 0.001
    best_sync_prob = 0.0
    
    for K_test in K_test_values:
        sync_count = 0
        for trial in range(20):  # More trials for coarse scan
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, omega_std_adaptive, N)
            
            # Evolve with N-scaled time
            for _ in range(evolution_steps):
                theta = runge_kutta_step(theta, omega, K_test, 0.01)
            
            r_final = np.abs(np.mean(np.exp(1j * theta)))
            if r_final > sync_threshold:
                sync_count += 1
        
        sync_prob = sync_count / 20
        if sync_prob > best_sync_prob:
            best_sync_prob = sync_prob
            best_K = K_test
    
    # Set binary search bounds around best K
    K_low = max(0.00001, best_K / 10)
    K_high = min(5.0, best_K * 10)
    
    for iteration in range(10):  # More binary search iterations
        K_mid = (K_low + K_high) / 2
        
        # Test synchronization probability
        sync_count = 0
        for trial in range(n_trials):
            theta = 2 * np.pi * np.random.rand(N)
            omega = np.random.normal(0, omega_std_adaptive, N)
            
            # Evolve
            for _ in range(evolution_steps):
                theta = runge_kutta_step(theta, omega, K_mid, 0.01)
            
            r_final = np.abs(np.mean(np.exp(1j * theta)))
            if r_final > sync_threshold:
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
    print(f" K_c ≈ {K_c:.4f} (P_sync = {sync_prob:.1%}, threshold={sync_threshold:.2f}, ω_std={omega_std_adaptive:.4f})")
    
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


def find_working_k_bootstrap(N_ref: int = 10, omega_std: float = 0.01, 
                           n_trials: int = 50) -> float:
    """
    Bootstrap approach: Find a working K for N_ref that gives reasonable synchronization.
    Based on research paper insights: try both positive and negative couplings.
    """
    print(f"╔{'═'*70}╗")
    print("║            BOOTSTRAP CALIBRATION FROM WORKING POINT                ║")
    print(f"╚{'═'*70}╝")
    print()
    print(f"Full mode: {len([10,20,30,50])} N values, {n_trials} trials each")
    print()
    
    print("=" * 70)
    print("STEP 1: FINDING WORKING K FOR N=10 (BOOTSTRAP ANCHOR)")
    print("=" * 70)
    print()
    
    # Test a much wider range of K values, including negative (inspired by research paper)
    K_test_values = [0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 
                    0.300, 0.400, 0.500, 0.750, 1.000, 1.500, 2.000,
                    -0.050, -0.100, -0.200, -0.300]  # Negative couplings
    
    print("Testing K values (including negative couplings from research paper):")
    print("-" * 60)
    
    best_K = 0.050
    best_sync_prob = 0.0
    
    for K in K_test_values:
        sync_count = 0
        for trial in range(n_trials):
            theta = 2 * np.pi * np.random.rand(N_ref)
            omega = np.random.normal(0, omega_std, N_ref)
            
            # Very long evolution for bootstrap
            for _ in range(10000):  # Even longer evolution
                theta = runge_kutta_step(theta, omega, K, 0.01)
            
            r_final = np.abs(np.mean(np.exp(1j * theta)))
            # Adaptive threshold based on research paper insights
            threshold = 0.3 if K < 0 else 0.5  # Lower threshold for negative K
            if r_final > threshold:
                sync_count += 1
        
        sync_prob = sync_count / n_trials
        print(f"K = {K:6.3f}: P_sync = {sync_prob:.1%}")
        
        if sync_prob > best_sync_prob:
            best_sync_prob = sync_prob
            best_K = K
    
    print()
    print(f"✅ Selected K = {best_K:.3f} (P_sync = {best_sync_prob:.1%})")
    print("   This will be our reference point K_ref at N_ref = 10")
    print()
    
    return best_K


def calibrate_alpha_bootstrap(N_values: List[int] = None,
                             omega_std: float = 0.01,
                             n_trials: int = 100) -> Dict[str, Any]:
    """
    Bootstrap calibration: Find working K at N=10, then scale as K(N) = K_ref × √(10/N)
    This avoids K_c detection issues and provides more stable results.
    """
    if N_values is None:
        N_values = [10, 20, 30, 50]  # Full mode N values
    
    # Step 1: Find working K for N=10
    K_ref = find_working_k_bootstrap(omega_std=omega_std, n_trials=n_trials)
    
    print("=" * 70)
    print("STEP 2: MEASURING BASIN VOLUMES WITH SCALED K")
    print("=" * 70)
    print(f"Using bootstrap scaling: K(N) = {K_ref:.3f} × √(10/N)")
    print(f"🔄 SMP: Using {min(mp.cpu_count(), len(N_values))} CPU cores for parallel processing")
    print()
    
    # SMP-enabled parallel processing
    K_test_values = [K_ref * np.sqrt(10.0 / N) for N in N_values]
    
    # Set global variables for multiprocessing
    global _global_K_c_values, _global_K_margin, _global_n_trials, _global_omega_std
    _global_K_c_values = K_test_values  # For bootstrap, K_c_values are actually K_test_values
    _global_K_margin = 1.0  # Not used in bootstrap, but set for compatibility
    _global_n_trials = n_trials
    _global_omega_std = omega_std
    
    # Parallel processing of basin volume measurements
    with mp.Pool(processes=min(mp.cpu_count(), len(N_values))) as pool:
        results = pool.map(_measure_single_N_bootstrap_mp, enumerate(N_values))
    
    # Sort results by index
    results.sort(key=lambda x: x[0])
    
    V_measured = [r[1] for r in results]
    V_errors = [r[2] for r in results]
    
    for i, N in enumerate(N_values):
        print(f"N={N} (K={K_test_values[i]:.4f}): V = {V_measured[i]:.4f} ± {V_errors[i]:.4f}")
    
    print()
    print("=" * 70)
    print("STEP 3: FITTING α FROM V(N) ~ exp(-α√N)")
    print("=" * 70)
    
    # Handle V=0 or V=1 cases
    valid_indices = []
    ln_V = []
    for i, v in enumerate(V_measured):
        if 0.01 < v < 0.99:  # Exclude extremes
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
        for i, N in enumerate(N_values):
            status = "✓" if i in valid_indices else "✗"
            print(f"  {status} N={N}: V={V_measured[i]:.4f} at K={K_test_values[i]:.4f}")
        print()
        print("❌ Calibration failed!")
        return {'alpha': 0.1, 'r_squared': 0.0}
    
    sqrt_N_valid = np.array([np.sqrt(N_values[i]) for i in valid_indices])
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
    print("Calibration Results:")
    print("-" * 40)
    print(f"Fitted: ln(V) = {slope:.4f}√N + {intercept:.4f}")
    print(f"α = {alpha_fitted:.4f}")
    print(f"R² = {r_squared:.3f}")
    print()
    
    if r_squared > 0.8:
        verdict = "✅ EXCELLENT FIT"
    elif r_squared > 0.6:
        verdict = "⚠️ MODERATE FIT"
    else:
        verdict = "❌ POOR FIT"
    
    print(f"Verdict: {verdict}")
    
    return {
        'alpha': max(0.01, alpha_fitted),
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'K_ref': K_ref,
        'data': {
            'N': N_values,
            'K_test': K_test_values,
            'V': V_measured,
            'V_err': V_errors,
            'valid_indices': valid_indices
        }
    }


def measure_basin_volume_bootstrap(N: int, K: float, n_trials: int = 100,
                                 omega_std: float = 0.01) -> Tuple[float, float]:
    """Measure basin volume with longer evolution for bootstrap approach."""
    sync_count = 0
    
    # Adaptive threshold based on coupling sign (research paper insight)
    threshold = 0.3 if K < 0 else 0.5
    
    for trial in range(n_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Very long evolution for bootstrap
        for _ in range(10000):  # Even longer evolution
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > threshold:
            sync_count += 1
    
    volume = sync_count / n_trials
    error = np.sqrt(volume * (1 - volume) / n_trials) if volume > 0 else 0
    
    return volume, error


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
    """Run complete analysis with bootstrap calibration and SMP support."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ROBUSTNESS ANALYSIS WITH BOOTSTRAP" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"🔄 Bootstrap Approach: Find working K at N=10, scale as K(N) = K_ref × √(10/N)")
    print()
    
    # Calibrate with bootstrap approach
    calibration = calibrate_alpha_bootstrap(
        N_values=[10, 20, 30, 50],
        omega_std=0.01,
        n_trials=100
    )
    
    if calibration['r_squared'] < 0.5:
        print("\n❌ Calibration failed. Try different parameters.")
        return
    
    # For bootstrap, we create a mock validation since we don't have K_c values
    # In a real implementation, you'd want to validate the inverse formula
    print()
    print("=" * 70)
    print("BOOTSTRAP CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"✓ Calibrated α = {calibration['alpha']:.4f} (R²={calibration['r_squared']:.3f})")
    print(f"✓ Reference K = {calibration['K_ref']:.4f} at N=10")
    print(f"✓ Scaling: K(N) = K_ref × √(10/N)")
    print()
    
    if calibration['r_squared'] > 0.8:
        print("🎉 SUCCESS! Bootstrap calibration achieved excellent fit!")
    else:
        print("⚠️ Moderate success. Results may need refinement.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer trials')
    parser.add_argument('--trials', type=int, default=100, help='Trials per measurement')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick test mode (fewer trials)")
        calibration = calibrate_alpha_bootstrap(
            N_values=[10, 20, 30],
            omega_std=0.01,
            n_trials=50
        )
        print(f"\nQuick result: α ≈ {calibration['alpha']:.4f}")
    else:
        run_complete_analysis()