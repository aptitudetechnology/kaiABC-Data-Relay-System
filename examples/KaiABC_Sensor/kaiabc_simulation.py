#!/usr/bin/env python3
"""
KaiABC Oscillator Network Simulation

This script simulates a network of KaiABC biological oscillators using the
Kuramoto model to validate the theoretical predictions before hardware deployment.

Usage:
    python kaiabc_simulation.py --nodes 10 --q10 1.1 --coupling 0.1 --days 30

Requirements:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import argparse
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SimulationConfig:
    """Configuration for KaiABC simulation"""
    num_nodes: int = 10
    q10: float = 1.1
    coupling: float = 0.1
    period_base: float = 24.0  # hours
    temp_ref: float = 30.0     # °C
    temp_variance: float = 5.0  # ±°C
    sim_days: int = 30
    dt: float = 0.1            # hours

class KaiABCNetwork:
    """Simulate a network of coupled KaiABC oscillators"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.n = config.num_nodes
        
        # Assign random temperatures to each node
        self.temperatures = np.random.uniform(
            config.temp_ref - config.temp_variance,
            config.temp_ref + config.temp_variance,
            self.n
        )
        
        # Calculate natural frequencies based on temperature
        self.omega = self.calculate_frequencies()
        
        # Initialize phases randomly
        self.phases = np.random.uniform(0, 2*np.pi, self.n)
        
        # Time array
        self.t = np.arange(0, config.sim_days * 24, config.dt)
        
        # Results storage
        self.phase_history = None
        self.order_param_history = None
        
    def calculate_frequencies(self) -> np.ndarray:
        """Calculate natural frequencies based on Q10 and temperature"""
        temp_diff = (self.config.temp_ref - self.temperatures) / 10.0
        periods = self.config.period_base * (self.config.q10 ** temp_diff)
        omega = 2 * np.pi / periods
        return omega
    
    def calculate_sigma_omega(self) -> float:
        """Calculate frequency heterogeneity"""
        ln_q10 = np.log(self.config.q10)
        omega_avg = 2 * np.pi / self.config.period_base
        dw_dt = (omega_avg / self.config.period_base) * (ln_q10 / 10.0)
        sigma_omega = abs(dw_dt) * self.config.temp_variance
        return sigma_omega
    
    def calculate_critical_coupling(self) -> float:
        """Calculate critical coupling K_c"""
        sigma_omega = self.calculate_sigma_omega()
        return 2.0 * sigma_omega
    
    def kuramoto_derivatives(self, phases: np.ndarray, t: float) -> np.ndarray:
        """Calculate phase derivatives using Kuramoto model"""
        # Natural frequency terms
        dphases = self.omega.copy()
        
        # Coupling terms: K/N * Σ sin(φ_j - φ_i)
        for i in range(self.n):
            coupling_sum = 0.0
            for j in range(self.n):
                if i != j:
                    coupling_sum += np.sin(phases[j] - phases[i])
            dphases[i] += (self.config.coupling / self.n) * coupling_sum
        
        return dphases
    
    def calculate_order_parameter(self, phases: np.ndarray) -> float:
        """Calculate order parameter R"""
        sum_complex = np.sum(np.exp(1j * phases))
        R = np.abs(sum_complex) / self.n
        return R
    
    def run_simulation(self) -> None:
        """Run the full simulation"""
        print(f"\n{'='*60}")
        print("KaiABC Network Simulation")
        print(f"{'='*60}")
        print(f"Number of nodes: {self.n}")
        print(f"Q10 coefficient: {self.config.q10}")
        print(f"Coupling strength K: {self.config.coupling}")
        print(f"Temperature range: {self.config.temp_ref}°C ± {self.config.temp_variance}°C")
        print(f"Simulation duration: {self.config.sim_days} days")
        
        # Calculate predictions
        sigma_omega = self.calculate_sigma_omega()
        k_c = self.calculate_critical_coupling()
        heterogeneity = sigma_omega / (2*np.pi/self.config.period_base) * 100
        basin_volume = (1 - 1.5*sigma_omega/(2*np.pi/self.config.period_base))**self.n * 100
        
        print(f"\nTheoretical Predictions:")
        print(f"  σ_ω: {sigma_omega:.6f} rad/hr")
        print(f"  K_c (critical): {k_c:.4f}")
        print(f"  K/K_c ratio: {self.config.coupling/k_c:.2f}")
        print(f"  Heterogeneity: {heterogeneity:.2f}%")
        print(f"  Basin volume: {basin_volume:.2f}%")
        
        if self.config.coupling > k_c:
            print(f"  ✓ Coupling ABOVE critical - synchronization expected")
        else:
            print(f"  ✗ Coupling BELOW critical - synchronization unlikely")
        
        print(f"\nRunning simulation...")
        
        # Integrate the Kuramoto equations
        self.phase_history = odeint(self.kuramoto_derivatives, self.phases, self.t)
        
        # Wrap phases to [0, 2π)
        self.phase_history = np.mod(self.phase_history, 2*np.pi)
        
        # Calculate order parameter over time
        self.order_param_history = np.array([
            self.calculate_order_parameter(phases) 
            for phases in self.phase_history
        ])
        
        # Find synchronization time (R > 0.95)
        sync_idx = np.where(self.order_param_history > 0.95)[0]
        if len(sync_idx) > 0:
            sync_time_hours = self.t[sync_idx[0]]
            sync_time_days = sync_time_hours / 24.0
            print(f"\n✓ Network synchronized!")
            print(f"  Synchronization time: {sync_time_hours:.1f} hours ({sync_time_days:.1f} days)")
        else:
            print(f"\n✗ Network did not synchronize within {self.config.sim_days} days")
        
        # Final statistics
        final_R = self.order_param_history[-1]
        final_phase_std = np.std(self.phase_history[-1])
        
        print(f"\nFinal State:")
        print(f"  Order parameter R: {final_R:.4f}")
        print(f"  Phase std dev: {final_phase_std:.4f} rad ({np.rad2deg(final_phase_std):.2f}°)")
        
        if final_R > 0.95:
            print(f"  Status: ✓ SYNCHRONIZED")
        elif final_R > 0.5:
            print(f"  Status: ○ Partially synchronized")
        else:
            print(f"  Status: ✗ Desynchronized")
        
        print(f"{'='*60}\n")
    
    def plot_results(self) -> None:
        """Create visualization plots"""
        if self.phase_history is None:
            print("Run simulation first!")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Phase trajectories
        ax = axes[0]
        t_days = self.t / 24.0
        for i in range(self.n):
            ax.plot(t_days, self.phase_history[:, i], alpha=0.6, linewidth=0.8)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f'Phase Trajectories (N={self.n}, Q10={self.config.q10}, K={self.config.coupling})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 2*np.pi])
        
        # Plot 2: Order parameter evolution
        ax = axes[1]
        ax.plot(t_days, self.order_param_history, 'b-', linewidth=2)
        ax.axhline(y=0.95, color='g', linestyle='--', label='Sync threshold (R=0.95)')
        ax.axhline(y=self.calculate_critical_coupling(), color='r', linestyle='--', 
                   label=f'Critical coupling (K_c={self.calculate_critical_coupling():.3f})')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Order Parameter R')
        ax.set_title('Synchronization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot 3: Phase space (final state)
        ax = axes[2]
        final_phases = self.phase_history[-1]
        x = np.cos(final_phases)
        y = np.sin(final_phases)
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        # Plot oscillator positions
        scatter = ax.scatter(x, y, c=self.temperatures, cmap='coolwarm', 
                           s=100, alpha=0.7, edgecolors='black')
        
        # Add colorbar for temperature
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)')
        
        # Draw average phase vector
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        ax.arrow(0, 0, avg_x, avg_y, head_width=0.1, head_length=0.1, 
                fc='red', ec='red', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('cos(φ)')
        ax.set_ylabel('sin(φ)')
        ax.set_title(f'Phase Space (Final State, R={self.order_param_history[-1]:.3f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'kaiabc_simulation_N{self.n}_Q{self.config.q10}_K{self.config.coupling}.png', dpi=150)
        print(f"Plot saved as kaiabc_simulation_N{self.n}_Q{self.config.q10}_K{self.config.coupling}.png")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simulate KaiABC oscillator network')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes (default: 10)')
    parser.add_argument('--q10', type=float, default=1.1, help='Q10 coefficient (default: 1.1)')
    parser.add_argument('--coupling', type=float, default=0.1, help='Coupling strength K (default: 0.1)')
    parser.add_argument('--days', type=int, default=30, help='Simulation days (default: 30)')
    parser.add_argument('--temp-var', type=float, default=5.0, help='Temperature variance ±°C (default: 5.0)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        num_nodes=args.nodes,
        q10=args.q10,
        coupling=args.coupling,
        sim_days=args.days,
        temp_variance=args.temp_var
    )
    
    # Create and run simulation
    network = KaiABCNetwork(config)
    network.run_simulation()
    
    if not args.no_plot:
        network.plot_results()

if __name__ == '__main__':
    main()
