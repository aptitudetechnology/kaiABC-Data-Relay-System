#!/usr/bin/env python3
"""
Enhanced KaiABC Basin Volume Test
Focus: Test the CRITICAL REGIME where theory predictions matter most
Runtime: ~10 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================================
# COPY YOUR CORE FUNCTIONS HERE (reuse your code)
# ============================================================================

@dataclass
class SimulationConfig:
    N: int = 10
    K: float = 0.10
    Q10: float = 1.1
    sigma_T: float = 5.0
    tau_ref: float = 24.0
    T_ref: float = 30.0
    t_max: int = 30 * 24
    dt: float = 0.1
    sync_threshold: float = 0.90

def calculate_sigma_omega(Q10, sigma_T, tau_ref):
    return (2*np.pi / tau_ref) * (abs(np.log(Q10)) / 10) * sigma_T

def predict_basin_volume(N, sigma_omega, omega_mean, K, alpha=1.5):
    K_c = 2 * sigma_omega
    if K <= K_c:
        return 0.0
    basin_volume = 1.0 - (K_c / K) ** (2 * N)
    return min(basin_volume, 1.0)

def calculate_order_parameter(phases):
    complex_avg = np.mean(np.exp(1j * phases))
    return abs(complex_avg)

def temperature_frequencies(N, sigma_T, Q10, tau_ref, T_ref):
    temperatures = np.random.normal(T_ref, sigma_T, N)
    periods = tau_ref * Q10 ** ((T_ref - temperatures) / 10)
    omegas = 2 * np.pi / periods
    return omegas

def simulate_kuramoto(config, initial_phases=None, omegas=None):
    N = config.N
    
    if initial_phases is None:
        phases = np.random.uniform(0, 2*np.pi, N)
    else:
        phases = initial_phases.copy()
    
    if omegas is None:
        omegas = temperature_frequencies(
            N, config.sigma_T, config.Q10, config.tau_ref, config.T_ref
        )
    
    R_history = []
    num_steps = int(config.t_max / config.dt)
    
    for step in range(num_steps):
        R = calculate_order_parameter(phases)
        R_history.append(R)
        
        coupling = np.zeros(N)
        for i in range(N):
            coupling[i] = np.sum(np.sin(phases - phases[i])) / N
        
        phases += config.dt * (omegas + config.K * coupling)
        phases = phases % (2*np.pi)
    
    return {
        'phases': phases,
        'R_history': R_history,
        'omegas': omegas,
        'final_R': R_history[-1]
    }

# ============================================================================
# ENHANCED TESTS FOR CRITICAL REGIME
# ============================================================================

def test_critical_regime(base_config, trials_per_K=50, verbose=True):
    """
    Test basin volume in the CRITICAL REGIME where predictions matter
    
    Focus on K/K_c ∈ [0.8, 2.5] where:
    - Below 1.0: Should NOT synchronize
    - 1.0-1.5: Transition regime (interesting!)
    - Above 2.0: Should synchronize (less interesting)
    """
    sigma_omega = calculate_sigma_omega(
        base_config.Q10, base_config.sigma_T, base_config.tau_ref
    )
    K_c = 2 * sigma_omega
    omega_mean = 2*np.pi / base_config.tau_ref
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CRITICAL REGIME TEST")
        print(f"{'='*70}")
        print(f"Configuration: N={base_config.N}, Q10={base_config.Q10}, σ_T={base_config.sigma_T}°C")
        print(f"K_c (critical) = {K_c:.4f} rad/hr")
        print(f"σ_ω/⟨ω⟩ = {sigma_omega/omega_mean:.2%}")
        print(f"\nTesting the INTERESTING regime: K/K_c ∈ [0.8, 2.5]")
        print(f"This is where basin volume formula should earn its keep!\n")
    
    # Test points in critical regime
    K_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]
    
    results = []
    
    print(f"{'K/K_c':<8} {'K':<10} {'Predicted':<12} {'Empirical':<12} {'Error':<10} {'Status'}")
    print("-" * 70)
    
    for K_ratio in K_ratios:
        config = SimulationConfig(
            N=base_config.N,
            K=K_ratio * K_c,
            Q10=base_config.Q10,
            sigma_T=base_config.sigma_T,
            tau_ref=base_config.tau_ref,
            t_max=base_config.t_max,
            dt=base_config.dt
        )
        
        # Predict basin volume
        V_pred = predict_basin_volume(base_config.N, sigma_omega, omega_mean, config.K)
        
        # Run trials
        converged = 0
        for trial in range(trials_per_K):
            result = simulate_kuramoto(config)
            last_day_R = result['R_history'][-int(24/config.dt):]
            if np.mean(last_day_R) > config.sync_threshold:
                converged += 1
        
        V_emp = converged / trials_per_K
        
        # Calculate error (handle zero prediction)
        if V_pred > 0.05:  # Only calculate error if prediction is meaningful
            error = abs(V_emp - V_pred) / V_pred
            error_str = f"{error:.1%}"
        else:
            error = float('inf')
            error_str = "N/A"
        
        # Assess accuracy
        if V_pred < 0.05 and V_emp < 0.15:
            status = "✅ Correct (no sync)"
        elif V_pred > 0.85 and V_emp > 0.75:
            status = "✅ Correct (full sync)"
        elif error < 0.20:
            status = "✅ Accurate"
        elif error < 0.40:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"
        
        print(f"{K_ratio:<8.1f} {config.K:<10.4f} {V_pred:<12.2%} "
              f"{V_emp:<12.2%} {error_str:<10} {status}")
        
        results.append({
            'K_ratio': K_ratio,
            'K': config.K,
            'V_predicted': V_pred,
            'V_empirical': V_emp,
            'error': error,
            'converged': converged,
            'trials': trials_per_K
        })
    
    return results, K_c

def analyze_results(results, K_c):
    """
    Analyze test results and provide actionable insights
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    # Partition results by regime
    below_critical = [r for r in results if r['K'] < K_c]
    transition = [r for r in results if K_c <= r['K'] < 1.5*K_c]
    above_critical = [r for r in results if r['K'] >= 1.5*K_c]
    
    print(f"\n1. BELOW CRITICAL (K < K_c):")
    if below_critical:
        mean_conv = np.mean([r['V_empirical'] for r in below_critical])
        print(f"   Mean convergence: {mean_conv:.1%}")
        print(f"   Expected: <15% (mostly random chance)")
        if mean_conv < 0.15:
            print(f"   ✅ Formula correctly predicts no synchronization")
        else:
            print(f"   ⚠️ Higher than expected - check K_c calculation")
    
    print(f"\n2. TRANSITION REGIME (K_c ≤ K < 1.5×K_c):")
    if transition:
        mean_error = np.mean([r['error'] for r in transition if r['error'] < float('inf')])
        mean_conv = np.mean([r['V_empirical'] for r in transition])
        print(f"   Mean convergence: {mean_conv:.1%}")
        print(f"   Mean formula error: {mean_error:.1%}")
        print(f"   This is the CRITICAL TEST of your theory!")
        if mean_error < 0.25:
            print(f"   ✅ Formula is accurate in transition regime")
        elif mean_error < 0.50:
            print(f"   ⚠️ Formula captures trend but needs refinement")
        else:
            print(f"   ❌ Formula fails in transition regime")
    
    print(f"\n3. ABOVE CRITICAL (K ≥ 1.5×K_c):")
    if above_critical:
        mean_conv = np.mean([r['V_empirical'] for r in above_critical])
        mean_error = np.mean([r['error'] for r in above_critical if r['error'] < float('inf')])
        print(f"   Mean convergence: {mean_conv:.1%}")
        print(f"   Mean formula error: {mean_error:.1%}")
        print(f"   Expected: >75% convergence")
        if mean_conv > 0.75:
            print(f"   ✅ Strong coupling regime behaves as expected")
        else:
            print(f"   ⚠️ Lower than expected - may need longer simulation time")
    
    # Overall assessment
    print(f"\n{'='*70}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'='*70}")
    
    # Count accurate predictions
    accurate = sum(1 for r in results if r['error'] < 0.25 or 
                   (r['V_predicted'] < 0.05 and r['V_empirical'] < 0.15))
    total = len(results)
    
    print(f"\nAccurate predictions: {accurate}/{total} ({accurate/total:.0%})")
    
    if accurate / total > 0.80:
        print(f"\n✅ HYPOTHESIS STRONGLY SUPPORTED")
        print(f"   → Basin volume formula is reliable")
        print(f"   → Kakeya → Kuramoto connection validated")
        print(f"   → Safe to proceed with hardware")
        print(f"\n💡 RECOMMENDED HARDWARE CONFIG:")
        print(f"   → Use K = 1.5-2.0 × K_c for reliable sync")
        print(f"   → Expected success rate: >75%")
    elif accurate / total > 0.60:
        print(f"\n⚠️ HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"   → Formula works in some regimes")
        print(f"   → May need refinement for edge cases")
        print(f"   → Hardware test with caution (use K ≥ 2×K_c)")
    else:
        print(f"\n❌ HYPOTHESIS NOT SUPPORTED")
        print(f"   → Formula needs major revision")
        print(f"   → DO NOT proceed to hardware yet")
        print(f"   → Investigate alternative basin volume formulas")

def test_network_size_scaling(Q10=1.1, sigma_T=5.0, K_ratio=1.5, trials=30):
    """
    Test: Does basin volume formula scale correctly with N?
    
    Theory predicts: V ∝ (1 - K_c/K)^(2N)
    So larger networks should have SMALLER basins (harder to sync)
    """
    print(f"\n{'='*70}")
    print(f"NETWORK SIZE SCALING TEST")
    print(f"{'='*70}")
    print(f"Testing basin volume scaling with N at fixed K/K_c = {K_ratio}")
    print(f"Theory: Larger networks → smaller basin volume\n")
    
    N_values = [3, 5, 10, 15, 20]
    
    print(f"{'N':<6} {'K':<10} {'Predicted':<12} {'Empirical':<12} {'Error':<10}")
    print("-" * 60)
    
    results = []
    
    for N in N_values:
        base_config = SimulationConfig(
            N=N, Q10=Q10, sigma_T=sigma_T, tau_ref=24.0, t_max=30*24, dt=0.1
        )
        
        sigma_omega = calculate_sigma_omega(Q10, sigma_T, 24.0)
        K_c = 2 * sigma_omega
        K = K_ratio * K_c
        omega_mean = 2*np.pi / 24.0
        
        config = SimulationConfig(
            N=N, K=K, Q10=Q10, sigma_T=sigma_T, 
            tau_ref=24.0, t_max=30*24, dt=0.1
        )
        
        V_pred = predict_basin_volume(N, sigma_omega, omega_mean, K)
        
        # Run trials
        converged = 0
        for trial in range(trials):
            result = simulate_kuramoto(config)
            last_day_R = result['R_history'][-int(24/config.dt):]
            if np.mean(last_day_R) > config.sync_threshold:
                converged += 1
        
        V_emp = converged / trials
        error = abs(V_emp - V_pred) / V_pred if V_pred > 0.05 else float('inf')
        
        print(f"{N:<6} {K:<10.4f} {V_pred:<12.2%} {V_emp:<12.2%} "
              f"{error:.1%}" if error < float('inf') else f"{N:<6} {K:<10.4f} {V_pred:<12.2%} {V_emp:<12.2%} N/A")
        
        results.append({
            'N': N,
            'V_predicted': V_pred,
            'V_empirical': V_emp,
            'error': error
        })
    
    # Check if scaling trend is correct
    print(f"\n{'='*70}")
    pred_decreasing = all(results[i]['V_predicted'] >= results[i+1]['V_predicted'] 
                         for i in range(len(results)-1))
    emp_decreasing = all(results[i]['V_empirical'] >= results[i+1]['V_empirical'] 
                        for i in range(len(results)-1))
    
    print(f"Predicted trend: {'✅ Decreasing' if pred_decreasing else '❌ Not monotonic'}")
    print(f"Empirical trend: {'✅ Decreasing' if emp_decreasing else '⚠️ Not monotonic (noisy)'}")
    
    mean_error = np.mean([r['error'] for r in results if r['error'] < float('inf')])
    print(f"\nMean scaling error: {mean_error:.1%}")
    
    if mean_error < 0.30 and pred_decreasing:
        print(f"✅ Scaling formula validated!")
    else:
        print(f"⚠️ Scaling may need refinement")
    
    return results

# ============================================================================
# MAIN ENHANCED TEST
# ============================================================================

def run_enhanced_mvp():
    """
    Run the enhanced test suite focused on critical regime
    """
    print("\n" + "="*70)
    print("ENHANCED KAIABC SOFTWARE TEST")
    print("="*70)
    print("\n🎯 Goal: Test basin volume formula where it matters most")
    print("   (The critical regime, not the trivial K >> K_c case)\n")
    
    # Base configuration
    base_config = SimulationConfig(
        N=10,
        Q10=1.1,
        sigma_T=5.0,
        tau_ref=24.0,
        t_max=30*24,
        dt=0.1
    )
    
    # Test 1: Critical regime sweep
    print("TEST 1: Critical Regime Sweep")
    results, K_c = test_critical_regime(base_config, trials_per_K=50, verbose=True)
    analyze_results(results, K_c)
    
    # Test 2: Network size scaling
    print("\n" + "="*70)
    print("TEST 2: Network Size Scaling")
    scaling_results = test_network_size_scaling(
        Q10=1.1, sigma_T=5.0, K_ratio=1.5, trials=30
    )
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    # Analyze transition regime specifically
    transition_results = [r for r in results if 1.0 <= r['K_ratio'] <= 1.5]
    if transition_results:
        transition_error = np.mean([r['error'] for r in transition_results 
                                   if r['error'] < float('inf')])
        
        print(f"\nCritical metric: Transition regime accuracy")
        print(f"Error in K/K_c ∈ [1.0, 1.5]: {transition_error:.1%}")
        
        if transition_error < 0.25:
            print(f"\n✅ PROCEED TO HARDWARE")
            print(f"   Recommended settings:")
            print(f"   • N = 5-10 devices")
            print(f"   • K = 1.5-2.0 × K_c = {1.5*K_c:.4f}-{2.0*K_c:.4f} rad/hr")
            print(f"   • Expected sync rate: 60-80%")
            print(f"   • Budget: $300-400")
        elif transition_error < 0.50:
            print(f"\n⚠️ PROCEED WITH CAUTION")
            print(f"   Formula works but less accurate than ideal")
            print(f"   Recommendation: Use K = 2.5×K_c for safety")
        else:
            print(f"\n❌ DO NOT PROCEED TO HARDWARE")
            print(f"   Formula needs revision first")
            print(f"   Try: Alternative basin volume formulas")

if __name__ == "__main__":
    run_enhanced_mvp()