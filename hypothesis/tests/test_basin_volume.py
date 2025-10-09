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

def predict_basin_volume(N, sigma_omega, omega_mean, alpha=1.5):
    """
    Kakeya-derived basin volume formula
    
    V_basin/V_total ≈ (1 - α·σ_ω/⟨ω⟩)^N
    
    where α ≈ 1.5 (empirical constant for Kuramoto)
    """
    ratio = alpha * sigma_omega / omega_mean
    if ratio >= 1.0:
        return 0.0  # No basin if frequency spread too large
    return (1.0 - ratio) ** N

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
    V_predicted = predict_basin_volume(config.N, sigma_omega, omega_mean)
    
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

if __name__ == "__main__":
    result = run_mvp()
