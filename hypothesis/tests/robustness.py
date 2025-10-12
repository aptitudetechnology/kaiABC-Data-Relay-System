#!/usr/bin/env python3
"""
FIXED: Robustness Analysis with α Calibration
==============================================
The key insight: α must be measured from data first!
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


def measure_basin_volume(N: int, K: float, n_trials: int = 1000,
                        omega_std: float = 0.01) -> Tuple[float, float]:
    """Measure basin volume via Monte Carlo."""
    sync_count = 0
    
    for trial in range(n_trials):
        theta = 2 * np.pi * np.random.rand(N)
        omega = np.random.normal(0, omega_std, N)
        
        # Evolve system
        for _ in range(500):
            theta = runge_kutta_step(theta, omega, K, 0.01)
        
        # Check synchronization
        r_final = np.abs(np.mean(np.exp(1j * theta)))
        if r_final > 0.8:
            sync_count += 1
    
    volume = sync_count / n_trials
    error = np.sqrt(volume * (1 - volume) / n_trials)
    
    return volume, error


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
        V, V_err = measure_basin_volume(N, K, n_trials, omega_std)
        V_measured.append(V)
        V_errors.append(V_err)
        print(f"V={V:.4f} ± {V_err:.4f}")
    
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
        V_measured, V_error = measure_basin_volume(N_test, K_test, n_trials, omega_std)
        
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
    print()
    
    # Step 1: Calibrate α
    calibration = calibrate_alpha(
        N_values=[10, 20, 30, 50],
        K=0.02,
        omega_std=0.01,
        n_trials=500
    )
    
    if calibration['r_squared'] < 0.5:
        print("\n❌ Calibration failed. Cannot proceed with validation.")
        return
    
    # Step 2: Validate inverse formula
    validation = validate_inverse_formula(
        calibration,
        V_targets=[0.30, 0.20, 0.10, 0.05],
        n_trials=300
    )
    
    # Step 3: Application example
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