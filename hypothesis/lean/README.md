# Kuramoto Basin Scaling Formalization

This Lean project formalizes the mathematical foundations of the hypothesis that Kuramoto oscillators near the synchronization threshold behave as if they have only √N effective degrees of freedom.

## Files

- `Kuramoto.lean`: Basic definitions of the Kuramoto model, order parameters, and synchronization
- `BasinVolume.lean`: Formalization of basin volumes, scaling laws, and large deviation principles
- `ScalingLaws.lean`: Asymptotic scaling behaviors and hypothesis testing framework
- `EffectiveDOF.lean`: The central hypothesis and mechanistic explanations
- `Examples.lean`: Computational examples and validation tests

## Central Hypothesis

In the Kuramoto model near K ≈ K_c, N coupled oscillators behave as if there are only √N effective independent degrees of freedom, explaining the observed basin volume scaling V ∼ exp(-α√N).

## Key Theorems

- Effective DOF scale as N_eff ∼ √N
- Basin volume follows V ∼ 1 - exp(-α√N)
- Order parameter fluctuations σ_R ∼ N^(-1/2)
- Correlation length ξ ∼ N^(1/2)
- Eigenvalue gap λ_gap ∼ N^(-1/4)

## Mechanistic Explanations

1. **Spatial correlation clusters**: Oscillators form √N-sized clusters
2. **Watanabe-Strogatz reduction**: (N-1)-dimensional manifold with √N transverse directions
3. **Critical slowing down**: Only modes with λ < ξ ∼ √N are relevant

## Validation Protocol

The hypothesis can be tested by measuring N_eff via PCA and checking if it scales as N^(1/2) with exponent ν ∈ [0.4, 0.6].

## Building

```bash
leanpkg build
```

## Running Examples

```bash
lean Examples.lean
```

This formalization provides a foundation for rigorous mathematical proof of the empirical basin scaling observations.