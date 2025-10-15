#!/usr/bin/env python3
"""
Effective Degrees of Freedom Analysis for Kuramoto Model
Tests the hypothesis: N_eff ~ N^(1/2)

This code measures the effective dimensionality of phase space dynamics
near the synchronization threshold using multiple independent methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

@dataclass
class EffectiveDOFResult:
    """Results from effective DOF measurement"""
    N: int
    N_eff_pca: float
    N_eff_variance: float
    N_eff_correlation: float
    N_eff_eigenspectrum: float
    order_parameter_std: float
    correlation_length: float
    eigenvalue_gap: float
    trajectories: np.ndarray

class KuramotoEffectiveDOF:
    """Measure effective degrees of freedom in Kuramoto model"""

    def __init__(self, K: float = 2.0, sigma_omega: float = 1.0, dt: float = 0.05):
        """
        Parameters:
        -----------
        K : float
            Coupling strength (should be near K_c ~ 1.5-2.0)
        sigma_omega : float
            Standard deviation of natural frequencies
        dt : float
            Integration timestep
        """
        self.K = K
        self.sigma_omega = sigma_omega
        self.dt = dt

    def kuramoto_ode(self, theta: np.ndarray, t: float, omega: np.ndarray, K: float) -> np.ndarray:
        """Kuramoto model ODE"""
        N = len(theta)
        dtheta = omega.copy()
        for i in range(N):
            dtheta[i] += (K / N) * np.sum(np.sin(theta - theta[i]))
        return dtheta

    def generate_trajectory(self, N: int, T: float = 50.0, transient: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a trajectory from random initial conditions

        Returns:
        --------
        t : array of time points
        theta : array of phases (shape: time x N)
        omega : natural frequencies
        """
        # Random natural frequencies from Lorentzian
        omega = np.random.standard_cauchy(N) * self.sigma_omega

        # Random initial phases
        theta0 = np.random.uniform(0, 2*np.pi, N)

        # Time points
        t_transient = np.arange(0, transient, self.dt)
        t_measure = np.arange(0, T, self.dt)

        # Remove transient
        theta_transient = odeint(self.kuramoto_ode, theta0, t_transient,
                                 args=(omega, self.K))

        # Measure trajectory
        theta_trajectory = odeint(self.kuramoto_ode, theta_transient[-1], t_measure,
                                  args=(omega, self.K))

        # Wrap to [-π, π]
        theta_trajectory = np.mod(theta_trajectory + np.pi, 2*np.pi) - np.pi

        return t_measure, theta_trajectory, omega

    def measure_pca_dimension(self, trajectories: List[np.ndarray], variance_threshold: float = 0.95) -> float:
        """
        Measure effective dimension using PCA

        Parameters:
        -----------
        trajectories : list of arrays
            List of phase trajectories (each shape: time x N)
        variance_threshold : float
            Cumulative variance threshold (default 95%)

        Returns:
        --------
        N_eff : float
            Number of components needed to explain variance_threshold of variance
        """
        # Concatenate all trajectories
        all_data = np.vstack(trajectories)

        # PCA
        pca = PCA()
        pca.fit(all_data)

        # Find number of components for threshold
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        N_eff = np.searchsorted(cumsum, variance_threshold) + 1

        return float(N_eff)

    def measure_variance_dimension(self, trajectories: List[np.ndarray]) -> float:
        """
        Measure effective dimension from variance scaling
        Counts modes with variance above threshold
        """
        all_data = np.vstack(trajectories)

        # Variance in each dimension
        variances = np.var(all_data, axis=0)

        # Normalized participation ratio
        total_var = np.sum(variances)
        if total_var > 0:
            p = variances / total_var
            N_eff = 1.0 / np.sum(p**2)  # Inverse participation ratio
        else:
            N_eff = 1.0

        return N_eff

    def measure_correlation_dimension(self, theta: np.ndarray) -> Tuple[float, float]:
        """
        Measure spatial correlation length and dimension

        Returns:
        --------
        correlation_length : float
            Spatial correlation length ξ
        N_eff : float
            Effective dimension ~ N/ξ
        """
        N = theta.shape[1]

        # Time-averaged correlation matrix
        C = np.zeros((N, N))
        for t in range(len(theta)):
            for i in range(N):
                for j in range(N):
                    C[i, j] += np.cos(theta[t, i] - theta[t, j])
        C /= len(theta)

        # Extract correlation length from decay
        # Fit C[0, j] ~ exp(-j/ξ)
        distances = np.arange(1, min(N//2, 50))
        correlations = np.array([np.mean(C[np.arange(N), (np.arange(N) + d) % N])
                                for d in distances])

        # Fit exponential decay
        log_corr = np.log(np.maximum(correlations, 1e-10))
        valid = log_corr < 0

        if np.sum(valid) > 5:
            slope, _ = np.polyfit(distances[valid], log_corr[valid], 1)
            xi = -1.0 / slope if slope < 0 else N
        else:
            xi = N  # Fully correlated

        N_eff = N / max(xi, 1.0)

        return xi, N_eff

    def measure_eigenspectrum_dimension(self, theta: np.ndarray, omega: np.ndarray) -> Tuple[float, float]:
        """
        Measure effective dimension from eigenvalue spectrum of Jacobian

        Returns:
        --------
        eigenvalue_gap : float
            Gap between largest and next eigenvalue
        N_eff : float
            Number of eigenvalues above threshold
        """
        N = theta.shape[1]

        # Compute Jacobian at mean configuration
        theta_mean = np.mean(theta, axis=0)

        # Jacobian of Kuramoto: J[i,j] = (K/N) * cos(theta_j - theta_i)
        J = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    J[i, j] = (self.K / N) * np.cos(theta_mean[j] - theta_mean[i])
                else:
                    J[i, i] = -(self.K / N) * np.sum(np.cos(theta_mean - theta_mean[i]))

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(J)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        # Gap between first and second eigenvalue
        eigenvalue_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0

        # Count eigenvalues above 1% of maximum
        threshold = 0.01 * np.abs(eigenvalues[0])
        N_eff = np.sum(np.abs(eigenvalues) > threshold)

        return eigenvalue_gap, float(N_eff)

    def measure_order_parameter_std(self, theta: np.ndarray) -> float:
        """Measure standard deviation of order parameter"""
        # Order parameter: R = |<e^{i*theta}>|
        R = np.abs(np.mean(np.exp(1j * theta), axis=1))
        return np.std(R)

    def analyze_single_N(self, N: int, n_trials: int = 50,
                        T: float = 50.0, verbose: bool = True) -> EffectiveDOFResult:
        """
        Complete analysis for a single N value

        Parameters:
        -----------
        N : int
            Number of oscillators
        n_trials : int
            Number of independent trajectories
        T : float
            Trajectory length

        Returns:
        --------
        EffectiveDOFResult with all measurements
        """
        if verbose:
            print(f"\nAnalyzing N={N} with {n_trials} trials...")

        trajectories = []
        omega_list = []
        correlation_lengths = []
        eigenvalue_gaps = []
        order_stds = []

        for trial in range(n_trials):
            if verbose and trial % 10 == 0:
                print(f"  Trial {trial}/{n_trials}...", end='\r')

            # Generate trajectory
            t, theta, omega = self.generate_trajectory(N, T=T)
            trajectories.append(theta)
            omega_list.append(omega)

            # Measure correlation length
            xi, _ = self.measure_correlation_dimension(theta)
            correlation_lengths.append(xi)

            # Measure eigenvalue gap
            gap, _ = self.measure_eigenspectrum_dimension(theta, omega)
            eigenvalue_gaps.append(gap)

            # Order parameter
            order_stds.append(self.measure_order_parameter_std(theta))

        if verbose:
            print(f"  Trial {n_trials}/{n_trials}... Done!")

        # Aggregate measurements
        N_eff_pca = self.measure_pca_dimension(trajectories, variance_threshold=0.95)
        N_eff_variance = self.measure_variance_dimension(trajectories)

        # Average correlation-based N_eff
        xi_mean = np.mean(correlation_lengths)
        N_eff_correlation = N / max(xi_mean, 1.0)

        # Eigenspectrum-based (use first trajectory for speed)
        _, N_eff_eigen = self.measure_eigenspectrum_dimension(trajectories[0], omega_list[0])

        result = EffectiveDOFResult(
            N=N,
            N_eff_pca=N_eff_pca,
            N_eff_variance=N_eff_variance,
            N_eff_correlation=N_eff_correlation,
            N_eff_eigenspectrum=N_eff_eigen,
            order_parameter_std=np.mean(order_stds),
            correlation_length=xi_mean,
            eigenvalue_gap=np.mean(eigenvalue_gaps),
            trajectories=np.array(trajectories[0])  # Save one example
        )

        if verbose:
            print(f"\nResults for N={N}:")
            print(f"  N_eff (PCA):          {N_eff_pca:.2f}")
            print(f"  N_eff (variance):     {N_eff_variance:.2f}")
            print(f"  N_eff (correlation):  {N_eff_correlation:.2f}")
            print(f"  N_eff (eigenspectrum):{N_eff_eigen:.2f}")
            print(f"  σ_R:                  {np.mean(order_stds):.4f}")
            print(f"  ξ (corr. length):     {xi_mean:.2f}")
            print(f"  λ_gap:                {np.mean(eigenvalue_gaps):.4f}")

        return result

def test_effective_dof_scaling(N_values: List[int] = [10, 20, 30, 50, 75, 100],
                               n_trials: int = 50,
                               K: float = 2.0,
                               plot: bool = True) -> Dict:
    """
    Test the N_eff ~ N^(1/2) scaling hypothesis

    Parameters:
    -----------
    N_values : list of int
        System sizes to test
    n_trials : int
        Trials per N value
    K : float
        Coupling strength
    plot : bool
        Whether to generate plots

    Returns:
    --------
    results : dict
        Complete analysis results and fitted parameters
    """
    print("="*70)
    print("EFFECTIVE DEGREES OF FREEDOM SCALING TEST")
    print("="*70)
    print(f"Hypothesis: N_eff ~ N^(1/2)")
    print(f"Testing N ∈ {N_values}")
    print(f"Coupling: K = {K}")
    print(f"Trials per N: {n_trials}")
    print("="*70)

    analyzer = KuramotoEffectiveDOF(K=K)

    results_list = []
    start_time = time.time()

    for N in N_values:
        result = analyzer.analyze_single_N(N, n_trials=n_trials)
        results_list.append(result)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    # Extract data for fitting
    N_array = np.array([r.N for r in results_list])
    N_eff_pca = np.array([r.N_eff_pca for r in results_list])
    N_eff_variance = np.array([r.N_eff_variance for r in results_list])
    N_eff_correlation = np.array([r.N_eff_correlation for r in results_list])
    N_eff_eigenspectrum = np.array([r.N_eff_eigenspectrum for r in results_list])

    order_stds = np.array([r.order_parameter_std for r in results_list])
    corr_lengths = np.array([r.correlation_length for r in results_list])
    eigen_gaps = np.array([r.eigenvalue_gap for r in results_list])

    # Fit power laws: N_eff = A * N^ν
    def fit_power_law(x, y):
        log_x = np.log(x)
        log_y = np.log(np.maximum(y, 1e-10))

        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), log_y)

        nu = model.coef_[0]
        A = np.exp(model.intercept_)
        r2 = model.score(log_x.reshape(-1, 1), log_y)

        return A, nu, r2

    print("\n" + "="*70)
    print("POWER LAW FITS: N_eff = A * N^ν")
    print("="*70)

    A_pca, nu_pca, r2_pca = fit_power_law(N_array, N_eff_pca)
    print(f"PCA:          N_eff = {A_pca:.2f} * N^{nu_pca:.3f}  (R² = {r2_pca:.4f})")

    A_var, nu_var, r2_var = fit_power_law(N_array, N_eff_variance)
    print(f"Variance:     N_eff = {A_var:.2f} * N^{nu_var:.3f}  (R² = {r2_var:.4f})")

    A_corr, nu_corr, r2_corr = fit_power_law(N_array, N_eff_correlation)
    print(f"Correlation:  N_eff = {A_corr:.2f} * N^{nu_corr:.3f}  (R² = {r2_corr:.4f})")

    A_eigen, nu_eigen, r2_eigen = fit_power_law(N_array, N_eff_eigenspectrum)
    print(f"Eigenspectrum:N_eff = {A_eigen:.2f} * N^{nu_eigen:.3f}  (R² = {r2_eigen:.4f})")

    # Test secondary predictions
    print("\n" + "="*70)
    print("SECONDARY PREDICTIONS")
    print("="*70)

    A_order, nu_order, r2_order = fit_power_law(N_array, order_stds)
    print(f"Order param std: σ_R = {A_order:.3f} * N^{nu_order:.3f}  (R² = {r2_order:.4f})")
    print(f"  Expected: ν = -0.5, Observed: ν = {nu_order:.3f}")

    A_xi, nu_xi, r2_xi = fit_power_law(N_array, corr_lengths)
    print(f"Corr. length:    ξ = {A_xi:.2f} * N^{nu_xi:.3f}  (R² = {r2_xi:.4f})")
    print(f"  Expected: ν = 0.5, Observed: ν = {nu_xi:.3f}")

    A_gap, nu_gap, r2_gap = fit_power_law(N_array, eigen_gaps)
    print(f"Eigenvalue gap:  λ = {A_gap:.3f} * N^{nu_gap:.3f}  (R² = {r2_gap:.4f})")
    print(f"  Expected: ν = -0.25, Observed: ν = {nu_gap:.3f}")

    # Verdict
    print("\n" + "="*70)
    print("HYPOTHESIS VALIDATION")
    print("="*70)

    expected_nu = 0.5
    tolerance = 0.15  # ν ∈ [0.35, 0.65]

    # Check if any method gives ν ≈ 0.5
    methods = [
        ("PCA", nu_pca, r2_pca),
        ("Variance", nu_var, r2_var),
        ("Correlation", nu_corr, r2_corr),
        ("Eigenspectrum", nu_eigen, r2_eigen)
    ]

    validated = False
    for name, nu, r2 in methods:
        in_range = abs(nu - expected_nu) < tolerance
        good_fit = r2 > 0.7

        status = "✓ PASS" if (in_range and good_fit) else "✗ FAIL"
        print(f"{name:15s}: ν = {nu:.3f}, R² = {r2:.3f}  {status}")

        if in_range and good_fit:
            validated = True

    print("\n" + "="*70)
    if validated:
        print("✓ HYPOTHESIS VALIDATED: N_eff ~ N^(1/2)")
        print("  At least one method shows ν ∈ [0.35, 0.65] with R² > 0.7")
    else:
        print("✗ HYPOTHESIS REJECTED: N_eff ≁ N^(1/2)")
        print("  No method shows consistent N^(1/2) scaling")
    print("="*70)

    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Effective DOF Scaling Analysis (K={K})', fontsize=14, fontweight='bold')

        # Plot 1: N_eff by different methods
        ax = axes[0, 0]
        ax.loglog(N_array, N_eff_pca, 'o-', label='PCA', markersize=8)
        ax.loglog(N_array, N_eff_variance, 's-', label='Variance', markersize=8)
        ax.loglog(N_array, N_eff_correlation, '^-', label='Correlation', markersize=8)
        ax.loglog(N_array, N_eff_eigenspectrum, 'd-', label='Eigenspectrum', markersize=8)
        ax.loglog(N_array, np.sqrt(N_array), 'k--', linewidth=2, label='√N (theory)')
        ax.set_xlabel('N (number of oscillators)', fontsize=11)
        ax.set_ylabel('$N_{eff}$ (effective DOF)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('Effective Degrees of Freedom')

        # Plot 2: Order parameter fluctuations
        ax = axes[0, 1]
        ax.loglog(N_array, order_stds, 'o-', markersize=8, label='Measured')
        ax.loglog(N_array, order_stds[0] * np.sqrt(N_array[0]/N_array), 'k--',
                 linewidth=2, label='$N^{-1/2}$')
        ax.set_xlabel('N', fontsize=11)
        ax.set_ylabel('$\\sigma_R$', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Order Parameter Std (ν={nu_order:.3f})')

        # Plot 3: Correlation length
        ax = axes[0, 2]
        ax.loglog(N_array, corr_lengths, 'o-', markersize=8, label='Measured')
        ax.loglog(N_array, corr_lengths[0] * np.sqrt(N_array/N_array[0]), 'k--',
                 linewidth=2, label='$N^{1/2}$')
        ax.set_xlabel('N', fontsize=11)
        ax.set_ylabel('$\\xi$ (correlation length)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Correlation Length (ν={nu_xi:.3f})')

        # Plot 4: Eigenvalue gap
        ax = axes[1, 0]
        ax.loglog(N_array, eigen_gaps, 'o-', markersize=8, label='Measured')
        ax.loglog(N_array, eigen_gaps[0] * (N_array[0]/N_array)**0.25, 'k--',
                 linewidth=2, label='$N^{-1/4}$')
        ax.set_xlabel('N', fontsize=11)
        ax.set_ylabel('$\\lambda_{gap}$', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Eigenvalue Gap (ν={nu_gap:.3f})')

        # Plot 5: Scaling exponents comparison
        ax = axes[1, 1]
        methods_names = ['PCA', 'Variance', 'Correlation', 'Eigen']
        exponents = [nu_pca, nu_var, nu_corr, nu_eigen]
        r2_values = [r2_pca, r2_var, r2_corr, r2_eigen]

        colors = ['green' if abs(e - 0.5) < tolerance and r > 0.7 else 'red'
                 for e, r in zip(exponents, r2_values)]

        bars = ax.bar(methods_names, exponents, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='Theory (ν=0.5)')
        ax.axhline(0.35, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.65, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('Scaling exponent ν', fontsize=11)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title('Scaling Exponents (green = validated)')

        # Plot 6: Example trajectory
        ax = axes[1, 2]
        example_theta = results_list[-1].trajectories[:500]  # Last N value
        N_example = results_list[-1].N

        im = ax.imshow(example_theta.T, aspect='auto', cmap='twilight',
                      vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        ax.set_xlabel('Time step', fontsize=11)
        ax.set_ylabel('Oscillator index', fontsize=11)
        ax.set_title(f'Example Trajectory (N={N_example})')
        plt.colorbar(im, ax=ax, label='Phase')

        plt.tight_layout()
        plt.savefig('effective_dof_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: effective_dof_analysis.png")
        plt.show()

    return {
        'results': results_list,
        'N_values': N_array,
        'fits': {
            'pca': (A_pca, nu_pca, r2_pca),
            'variance': (A_var, nu_var, r2_var),
            'correlation': (A_corr, nu_corr, r2_corr),
            'eigenspectrum': (A_eigen, nu_eigen, r2_eigen)
        },
        'validated': validated
    }

if __name__ == "__main__":
    # Run the full analysis
    results = test_effective_dof_scaling(
        N_values=[10, 20, 30, 50, 75, 100],
        n_trials=50,  # Increase for better statistics
        K=2.0,  # Near synchronization threshold
        plot=True
    )

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
