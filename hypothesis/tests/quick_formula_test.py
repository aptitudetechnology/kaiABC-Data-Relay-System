#!/usr/bin/env python3
"""
Quick Formula Comparison - Fast version for testing
"""

import numpy as np
from enhanced_test_basin_volume import *

def quick_compare():
    """Fast comparison with fewer trials"""
    print("\n" + "="*70)
    print("QUICK FORMULA COMPARISON (20 trials per point)")
    print("="*70)
    
    base_config = SimulationConfig(N=10, Q10=1.1, sigma_T=5.0, tau_ref=24.0, t_max=15*24, dt=0.2)
    sigma_omega = calculate_sigma_omega(base_config.Q10, base_config.sigma_T, base_config.tau_ref)
    K_c = 2 * sigma_omega
    omega_mean = 2*np.pi / base_config.tau_ref
    
    # Key transition points
    K_ratios = [1.1, 1.2, 1.3, 1.5]
    trials = 20
    
    print(f"\nK_c = {K_c:.4f} rad/hr")
    print(f"\n{'K/K_c':<8} {'Empirical':<12} {'V1 (2N)':<12} {'V2 (N)':<12} {'V3 (tanh)':<12}")
    print("-" * 70)
    
    errors = {1: [], 2: [], 3: []}
    
    for K_ratio in K_ratios:
        config = SimulationConfig(
            N=base_config.N, K=K_ratio * K_c, Q10=base_config.Q10,
            sigma_T=base_config.sigma_T, tau_ref=base_config.tau_ref,
            t_max=base_config.t_max, dt=base_config.dt
        )
        
        # Run simulations
        print(f"{K_ratio:<8.1f} ", end="", flush=True)
        converged = 0
        for trial in range(trials):
            result = simulate_kuramoto(config)
            last_day_R = result['R_history'][-int(24/config.dt):]
            if np.mean(last_day_R) > config.sync_threshold:
                converged += 1
        
        V_emp = converged / trials
        
        # Predictions
        V1 = predict_basin_volume(base_config.N, sigma_omega, omega_mean, config.K, formula_version=1)
        V2 = predict_basin_volume(base_config.N, sigma_omega, omega_mean, config.K, formula_version=2)
        V3 = predict_basin_volume(base_config.N, sigma_omega, omega_mean, config.K, formula_version=3)
        
        print(f"{V_emp:<12.1%} {V1:<12.1%} {V2:<12.1%} {V3:<12.1%}")
        
        # Errors
        if V_emp > 0.1:
            errors[1].append(abs(V1 - V_emp) / V_emp)
            errors[2].append(abs(V2 - V_emp) / V_emp)
            errors[3].append(abs(V3 - V_emp) / V_emp)
    
    # Results
    print("\n" + "="*70)
    print("MEAN ABSOLUTE ERROR:")
    print("-" * 70)
    
    best_error = float('inf')
    best_version = None
    
    for version in [1, 2, 3]:
        if errors[version]:
            mean_error = np.mean(errors[version])
            print(f"Formula V{version}: {mean_error:6.1%}", end="")
            
            if mean_error < best_error:
                best_error = mean_error
                best_version = version
            
            if mean_error < 0.20:
                print(f"  ✅ Excellent")
            elif mean_error < 0.30:
                print(f"  ✅ Good")
            elif mean_error < 0.40:
                print(f"  ⚠️ Acceptable")
            else:
                print(f"  ❌ Poor")
    
    print(f"\n🏆 WINNER: Formula V{best_version} with {best_error:.1%} error")
    
    if best_error < 0.25:
        print(f"\n✅ VALIDATED! Use V{best_version} for hardware predictions")
    elif best_error < 0.35:
        print(f"\n⚠️ Acceptable but could be better")
    else:
        print(f"\n❌ All formulas need improvement")

if __name__ == "__main__":
    quick_compare()
