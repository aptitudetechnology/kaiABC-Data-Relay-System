#!/usr/bin/env python3
"""
KaiABC Basin Volume Test - Minimal Viable Prototype
Tests: Does basin volume formula predict Kuramoto convergence rates?
Runtime: ~5 minutes
"""

import numpy as np
from dataclasses import dataclass

# ============================================================================
# CORE KURAMOTO SIMULATOR
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration for Kuramoto simulation"""
    N: int = 10              # Number of oscillators
    K: float = 0.10          # Coupling strength (rad/hr)
    Q10: float = 1.1         # Temperature compensation coefficient
    sigma_T: float = 5.0     # Temperature variance (°C)
    tau_ref: float = 24.0    # Reference period (hours)
    T_ref: float = 30.0      # Reference temperature (°C)
    t_max: int = 30 * 24     # Simulation duration (30 days in hours)
    dt: float = 0.1          # Time step (hours)
    sync_threshold: float = 0.90  # Order parameter for sync

def calculate_sigma_omega(Q10, sigma_T, tau_ref):
    """
    The Missing Link: σ_T → σ_ω
    
    σ_ω = (2π/τ_ref) · (|ln(Q10)|/10) · σ_T
    
    This is the novel contribution from deep-research-prompt-claude.md
    """
    return (2*np.pi / tau_ref) * (abs(np.log(Q10)) / 10) * sigma_T

def predict_basin_volume(N, sigma_omega, omega_mean, K, alpha=1.5):
    """
    Coupling-dependent basin volume formula
    
    Original formula: V ≈ (1 - α·σ_ω/⟨ω⟩)^N (too pessimistic for strong coupling)
    
    Corrected formula: V ≈ 1 - (K_c/K)^(2N) for K > K_c
    
    This accounts for the fact that strong coupling (K >> K_c) can synchronize
    even with significant frequency heterogeneity.
    
    Args:
        N: Number of oscillators
        sigma_omega: Frequency standard deviation (rad/hr)
        omega_mean: Mean frequency (rad/hr)
        K: Coupling strength (rad/hr)
        alpha: Scaling constant (unused in new formula, kept for backwards compatibility)
    
    Returns:
        Basin volume fraction (0 to 1)
    """
    # Critical coupling for synchronization
    K_c = 2 * sigma_omega  # Conservative estimate
    
    if K <= K_c:
        return 0.0  # Below critical coupling - no synchronization
    
    # For K > K_c, basin volume grows rapidly with coupling strength
    # Theoretical basis: Kuramoto phase transition theory
    basin_volume = 1.0 - (K_c / K) ** (2 * N)
    
    return min(basin_volume, 1.0)  # Cap at 100%

def calculate_order_parameter(phases):
    """
    Kuramoto order parameter: R = |⟨e^(iφ)⟩|
    
    R = 0: Completely desynchronized
    R = 1: Perfectly synchronized
    """
    complex_avg = np.mean(np.exp(1j * phases))
    return abs(complex_avg)

def temperature_frequencies(N, sigma_T, Q10, tau_ref, T_ref):
    """
    Generate heterogeneous frequencies due to temperature variance
    
    Each oscillator experiences slightly different temperature
    → Different periods via Q10 compensation
    → Frequency distribution with spread σ_ω
    """
    # Sample temperatures from Gaussian distribution
    temperatures = np.random.normal(T_ref, sigma_T, N)
    
    # Calculate period for each temperature
    periods = tau_ref * Q10 ** ((T_ref - temperatures) / 10)
    
    # Convert to angular frequencies
    omegas = 2 * np.pi / periods
    
    return omegas

def simulate_kuramoto(config, initial_phases=None, omegas=None):
    """
    Core Kuramoto dynamics with KaiABC temperature compensation
    
    dφᵢ/dt = ωᵢ + (K/N)·Σⱼ sin(φⱼ - φᵢ)
    """
    N = config.N
    
    # Initialize phases
    if initial_phases is None:
        phases = np.random.uniform(0, 2*np.pi, N)
    else:
        phases = initial_phases.copy()
    
    # Initialize frequencies with temperature heterogeneity
    if omegas is None:
        omegas = temperature_frequencies(
            N, config.sigma_T, config.Q10, config.tau_ref, config.T_ref
        )
    
    # Storage for order parameter history
    R_history = []
    
    # Main simulation loop
    num_steps = int(config.t_max / config.dt)
    for step in range(num_steps):
        # Calculate order parameter
        R = calculate_order_parameter(phases)
        R_history.append(R)
        
        # Kuramoto update (vectorized for speed)
        coupling = np.zeros(N)
        for i in range(N):
            coupling[i] = np.sum(np.sin(phases - phases[i])) / N
        
        phases += config.dt * (omegas + config.K * coupling)
        phases = phases % (2*np.pi)  # Wrap to [0, 2π]
    
    return {
        'phases': phases,
        'R_history': R_history,
        'omegas': omegas,
        'final_R': R_history[-1]
    }

# ============================================================================
# BASIN VOLUME HYPOTHESIS TEST
# ============================================================================

def test_basin_volume(config, trials=100, verbose=True):
    """
    H1: Basin Volume Hypothesis Test
    
    Run Monte Carlo trials with random initial conditions
    Count: What fraction converge to synchronization?
    Compare: Empirical rate vs. predicted basin volume
    """
    # Calculate theoretical prediction
    omega_mean = 2*np.pi / config.tau_ref
    sigma_omega = calculate_sigma_omega(config.Q10, config.sigma_T, config.tau_ref)
    K_c = 2 * sigma_omega  # Critical coupling
    V_predicted = predict_basin_volume(config.N, sigma_omega, omega_mean, config.K)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BASIN VOLUME TEST")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  N = {config.N} oscillators")
        print(f"  Q10 = {config.Q10}")
        print(f"  σ_T = {config.sigma_T}°C")
        print(f"  K = {config.K} rad/hr")
        print(f"\nTheoretical Predictions:")
        print(f"  σ_ω = {sigma_omega:.4f} rad/hr")
        print(f"  σ_ω/⟨ω⟩ = {sigma_omega/omega_mean:.2%}")
        print(f"  K_c (critical) = {K_c:.4f} rad/hr")
        print(f"  K/K_c = {config.K/K_c:.2f}× critical")
        print(f"  Basin Volume = {V_predicted:.2%}")
        print(f"\nRunning {trials} Monte Carlo trials...")
    
    # Monte Carlo simulation
    converged = 0
    for trial in range(trials):
        if verbose and (trial + 1) % 20 == 0:
            print(f"  Trial {trial+1}/{trials}...")
        
        result = simulate_kuramoto(config)
        
        # Check if synchronized (R > threshold for last 24 hours)
        last_day_R = result['R_history'][-int(24/config.dt):]
        if np.mean(last_day_R) > config.sync_threshold:
            converged += 1
    
    # Calculate empirical convergence rate
    V_empirical = converged / trials
    error = abs(V_empirical - V_predicted) / V_predicted if V_predicted > 0 else float('inf')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Predicted Basin Volume:  {V_predicted:.2%}")
        print(f"Empirical Convergence:   {V_empirical:.2%}")
        print(f"Error:                   {error:.1%}")
        print(f"Converged Trials:        {converged}/{trials}")
        
        if error < 0.15:
            print(f"\n✅ HYPOTHESIS ACCEPTED (error < 15%)")
            print(f"   → Basin volume formula is accurate!")
            print(f"   → Kakeya → Kuramoto connection validated!")
        elif error < 0.30:
            print(f"\n⚠️ MODERATE AGREEMENT (error < 30%)")
            print(f"   → Formula captures order of magnitude")
            print(f"   → May need refinement for precision")
        else:
            print(f"\n❌ HYPOTHESIS REJECTED (error > 30%)")
            print(f"   → Basin volume formula needs revision")
            print(f"   → Check assumptions or alternative formulas")
    
    return {
        'V_predicted': V_predicted,
        'V_empirical': V_empirical,
        'error': error,
        'converged': converged,
        'trials': trials,
        'sigma_omega': sigma_omega
    }

# ============================================================================
# MAIN MVP TEST
# ============================================================================

def run_mvp():
    """
    Minimal Viable Prototype: Test Q10 = 1.1 case
    
    This is the realistic KaiABC scenario from deep-research-prompt-claude.md
    Expected: ~28% convergence rate
    """
    print("\n" + "="*60)
    print("KAIABC SOFTWARE MVP: BASIN VOLUME TEST")
    print("="*60)
    print("\nTesting the Missing Link: σ_T → σ_ω")
    print("Source: deep-research-prompt-claude.md")
    print("\nThis tests the core Kakeya → Kuramoto hypothesis:")
    print("  Can we predict basin volume from temperature variance?")
    
    # Test Case: Realistic KaiABC (Q10 = 1.1)
    config = SimulationConfig(
        N=10,
        K=0.10,          # 2.4× critical coupling
        Q10=1.1,         # Realistic temperature compensation
        sigma_T=5.0,     # ±5°C temperature variance
        tau_ref=24.0,
        t_max=30*24,     # 30 days
        dt=0.1
    )
    
    # Run test with 100 trials (5 min runtime)
    result = test_basin_volume(config, trials=100, verbose=True)
    
    # Save results
    print(f"\n{'='*60}")
    print(f"NEXT STEPS")
    print(f"{'='*60}")
    
    if result['error'] < 0.15:
        print("✅ Theory validated! Proceed to:")
        print("   1. Test other Q10 values (1.0, 2.2)")
        print("   2. Test different N (3, 5, 20, 50)")
        print("   3. Run full hypothesis suite (all 6 tests)")
        print("   4. ORDER HARDWARE ($104)")
    else:
        print("⚠️ Theory needs refinement:")
        print("   1. Try alternative basin formulas")
        print("   2. Adjust α parameter")
        print("   3. Re-examine σ_T → σ_ω derivation")
        print("   4. Test with longer simulation time")
    
    return result

def test_coupling_sweep():
    """
    Test basin volume formula across different coupling strengths
    
    This validates that the formula correctly predicts the transition
    from no synchronization (K < K_c) to full synchronization (K >> K_c)
    """
    print("\n" + "="*60)
    print("COUPLING STRENGTH SWEEP TEST")
    print("="*60)
    print("\nTesting basin volume predictions across K/K_c range")
    print("Expected: Rapid transition near K_c, ~100% sync for K > 2×K_c\n")
    
    # Fixed parameters
    base_config = SimulationConfig(
        N=10,
        Q10=1.1,
        sigma_T=5.0,
        tau_ref=24.0,
        t_max=30*24,
        dt=0.1
    )
    
    # Calculate critical coupling
    sigma_omega = calculate_sigma_omega(base_config.Q10, base_config.sigma_T, base_config.tau_ref)
    K_c = 2 * sigma_omega
    
    print(f"K_c (critical coupling) = {K_c:.4f} rad/hr")
    print(f"\n{'K/K_c':<8} {'K (rad/hr)':<12} {'Predicted':<12} {'Empirical':<12} {'Error':<10}")
    print("-" * 60)
    
    # Test different coupling strengths
    K_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    results = []
    
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
        
        result = test_basin_volume(config, trials=50, verbose=False)
        results.append(result)
        
        error_str = f"{result['error']:.1%}" if result['error'] < float('inf') else "N/A"
        
        print(f"{K_ratio:<8.1f} {config.K:<12.4f} {result['V_predicted']:<12.2%} "
              f"{result['V_empirical']:<12.2%} {error_str:<10}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Check transition behavior
    below_critical = [r for i, r in enumerate(results) if K_ratios[i] < 1.0]
    above_critical = [r for i, r in enumerate(results) if K_ratios[i] > 1.5]
    
    print(f"\nBelow critical (K < K_c):")
    print(f"  Mean convergence: {np.mean([r['V_empirical'] for r in below_critical]):.1%}")
    print(f"  Expected: <10% (random chance)")
    
    print(f"\nWell above critical (K > 1.5×K_c):")
    print(f"  Mean convergence: {np.mean([r['V_empirical'] for r in above_critical]):.1%}")
    print(f"  Expected: >80% (strong synchronization)")
    
    # Calculate mean absolute error for K > K_c
    sync_regime = [r for i, r in enumerate(results) if K_ratios[i] > 1.0 and r['error'] < float('inf')]
    if sync_regime:
        mean_error = np.mean([r['error'] for r in sync_regime])
        print(f"\nFormula accuracy (K > K_c): {mean_error:.1%} mean error")
        
        if mean_error < 0.20:
            print("✅ Basin volume formula validated!")
        elif mean_error < 0.40:
            print("⚠️ Formula captures trend but needs refinement")
        else:
            print("❌ Formula needs significant revision")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Run coupling sweep test
        results = test_coupling_sweep()
    else:
        # Run standard MVP test
        result = run_mvp()
