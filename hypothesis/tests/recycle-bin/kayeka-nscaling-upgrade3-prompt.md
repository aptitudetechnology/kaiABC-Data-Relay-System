# GitHub Copilot Prompt: Derive Functional Forms from Geometric Measure Theory

## Context
You are implementing a mathematical derivation that explains WHY the empirically discovered basin volume formula V9.1 has its specific functional form. V9.1 achieves 4.9% error over 2000 trials but was discovered through trial-and-error parameter optimization, not theoretical derivation.

## Current Function to Implement
```python
def derive_functional_forms_from_geometric_measure_theory(
    formula_version: float,
    N: int,
    K: float,
    K_c: float
) -> Dict[str, Any]:
    """
    Derive V9.1 and Other Functional Forms from Geometric Measure Theory
    
    GOAL: Prove why basin volume formulas like V9.1 should have their specific structure
    
    Currently returns placeholder values. Need rigorous GMT derivation.
    """
```

## The V9.1 Formula (Empirically Discovered)

### Complete Formula Structure
```python
V9.1(N, K, K_c) = 1 - (K_c/K)^(α√N) × corrections

Where:
- α ≈ 2.5 (empirically fitted constant)
- K_c ≈ 1.0 (critical coupling, varies slightly with N)
- corrections include:
  * Logarithmic terms: f(log N)
  * Exponential decay: exp(-β/√N)
  * Polynomial adjustments: (1 + γ/N)
```

### Key Structural Elements to Explain
1. **Power law form**: (K_c/K)^exponent
2. **Square root scaling**: exponent ~ √N
3. **Critical coupling dependence**: K_c/K ratio
4. **Correction terms**: Why logarithmic and exponential?
5. **Asymptotic behavior**: V → 0 as K → K_c, V → 1 as K → ∞

## Mathematical Background

### Geometric Measure Theory (GMT) Framework

#### 1. Hausdorff Dimension and Measure
- **Hausdorff dimension**: dim_H(S) = inf{d : H^d(S) = 0}
- **Hausdorff measure**: H^d(S) = lim_{ε→0} inf{Σ_i diam(U_i)^d : S ⊂ ∪U_i, diam(U_i) < ε}
- For basin boundary B: dim_H(B) ~ N - √N (from Kakeya connection)
- Basin volume V ~ (phase space volume) - H^d(B)

#### 2. Minkowski Dimension and Content
- **Minkowski dimension**: dim_M(S) = d - lim_{ε→0} [log M_ε(S) / log(1/ε)]
  where M_ε(S) is ε-neighborhood measure
- **Minkowski content**: C^d(S) = lim_{ε→0} [volume(S_ε) / ε^(D-d)]
- Relation to basin: V(K) ~ C^d(boundary at K)

#### 3. Rectifiable Sets and Density
- **Rectifiable set**: Can be covered by countably many Lipschitz images
- **Density theorem**: For rectifiable S at scale r: μ(B_r(x) ∩ S) ~ r^d
- Basin boundaries may have partial rectifiability

### Phase Transition Theory

#### Critical Phenomena Near K_c
From dynamical systems theory:
```
Order parameter: r(K) = |⟨exp(iθ_j)⟩|
Near K_c: r ~ (K - K_c)^β (power law)
Basin volume: V ~ ∫[states with r > threshold]
```

**Critical Exponents**:
- β ≈ 0.5 (mean-field theory)
- ν ≈ 0.5 (correlation length exponent)
- Connection to √N: ν × √N scaling

#### Scaling Form Near Critical Point
General scaling theory suggests:
```
V(N, K) ~ N^(-γ) × F((K - K_c) × N^ν)

Where:
- γ is finite-size exponent
- ν is correlation exponent  
- F is universal scaling function
```

### Kakeya-Inspired Measure Bounds

From Kakeya theory (established in previous function):
```
Basin boundary has directional structure
⟹ Measure bounds: V(K) ≥ (K/K_c - 1)^(d_phase - d_boundary)
⟹ With d_phase = N, d_boundary ~ N - √N:
   V(K) ≥ (K/K_c - 1)^√N
⟹ Or equivalently: 1 - V(K) ≤ (K_c/K)^√N
```

## Implementation Requirements

### Component 1: Critical Exponent Derivation (`_derive_critical_exponents`)

```python
def _derive_critical_exponents(N: int, K_c: float) -> Dict[str, Any]:
    """
    Derive critical exponents from GMT and phase transition theory.
    
    Uses:
    - Mean-field theory (Landau functional)
    - Renormalization group analysis
    - Finite-size scaling theory
    
    Returns:
        exponents: Dict[str, float]
            - 'alpha': Power law coefficient (~ 2.5)
            - 'beta': Order parameter exponent (~ 0.5)
            - 'nu': Correlation length exponent (~ 0.5)
            - 'gamma': Finite-size exponent (~ 1.0)
        scaling_form: str
            Mathematical form of scaling function
        derivation: str
            Step-by-step derivation from first principles
        confidence: float
            How rigorous is this derivation? (0-1)
    """
```

**Mathematical Approach**:
1. Start with Kuramoto Landau functional: F[r] = -K∫r²/2 + ∫r⁴/4 + ...
2. Minimize to find equilibrium: δF/δr = 0
3. Near K_c: r ~ (K - K_c)^(1/2) (standard mean-field)
4. Basin volume ~ ∫_{r>r_sync} d^N θ ~ (K - K_c)^(N/2)
5. But directional constraints (Kakeya) reduce this: ~ (K - K_c)^(√N/2)
6. Therefore: α = 1/2 from mean-field, modified by √N scaling

### Component 2: Power Law Structure (`_derive_power_law_form`)

```python
def _derive_power_law_form(N: int, K: float, K_c: float) -> Dict[str, Any]:
    """
    Derive why the formula has (K_c/K)^(α√N) structure.
    
    Uses:
    - Hausdorff dimension of basin boundary
    - Minkowski content scaling
    - Dimensional analysis
    
    Returns:
        power_law_base: float
            Why K_c/K specifically (not K - K_c)
        exponent_structure: str
            Why α√N (not N or log N)
        geometric_interpretation: str
            What this means for basin geometry
        predicted_alpha: float
            Theoretical value of α
        bounds: Tuple[float, float]
            (lower, upper) bounds on α
    """
```

**Key Insight - Why (K_c/K)^exponent not (K - K_c)^exponent**:
```
Basin measure as K → ∞:
- (K - K_c)^α → ∞ (wrong - basin can't exceed 1)
- (K_c/K)^α → 0 (correct - boundary vanishes)
- Therefore: V = 1 - (K_c/K)^α form is natural

Also from dimensional analysis:
- K_c/K is dimensionless ratio (natural scaling variable)
- (K - K_c)/K_c is equivalent but less symmetric
```

### Component 3: Square Root Scaling (`_derive_sqrt_n_exponent`)

```python
def _derive_sqrt_n_exponent(N: int) -> Dict[str, Any]:
    """
    Derive why exponent scales as √N specifically.
    
    Uses:
    - Results from derive_sqrt_n_scaling_from_kakeya()
    - Kakeya boundary dimension: d_boundary ~ N - √N
    - Minkowski content formula
    
    Returns:
        theoretical_exponent: float
            Should be 0.5 (for √N)
        derivation_chain: List[str]
            Step-by-step mathematical logic
        alternative_scalings_ruled_out: List[Dict]
            Why not N, log N, N^(2/3), etc.
        confidence_level: float
            Strength of theoretical justification
    """
```

**Derivation Chain**:
```
1. Basin boundary dimension: d_b = N - √N (from Kakeya)
2. Minkowski content: C^d(B) ~ ε^(N - d_b) = ε^√N
3. Basin volume deficit: 1 - V ~ C^d(B)
4. Near critical point: ε ~ K_c/K
5. Therefore: 1 - V ~ (K_c/K)^√N ✓
```

### Component 4: Correction Terms (`_derive_correction_terms`)

```python
def _derive_correction_terms(N: int, K: float, K_c: float) -> Dict[str, Any]:
    """
    Derive logarithmic and exponential correction terms.
    
    Uses:
    - Finite-size corrections from GMT
    - Subleading terms in dimensional analysis
    - Higher-order phase transition effects
    
    Returns:
        corrections: Dict[str, Any]
            - 'logarithmic': f(log N) terms
            - 'exponential': exp(-β/√N) terms
            - 'polynomial': (1 + γ/N) terms
        mathematical_origin: str
            Why each correction appears
        magnitude_estimates: Dict[str, float]
            Predicted coefficients
        necessity: str
            Which corrections are essential vs. optional
    """
```

**Expected Corrections**:
```python
# Logarithmic: From finite-size effects
V_log = (log N / N) × poly(K/K_c)

# Exponential: From subdominant modes
V_exp = exp(-β√N / log N)

# Polynomial: From 1/N expansion
V_poly = (1 + c₁/N + c₂/N² + ...)
```

### Component 5: Universal Scaling Function (`_construct_scaling_function`)

```python
def _construct_scaling_function(N: int) -> Dict[str, Any]:
    """
    Construct the universal scaling function F in V ~ F((K-K_c)N^ν).
    
    Uses:
    - Renormalization group theory
    - Universality class identification
    - Asymptotic matching
    
    Returns:
        scaling_function: callable
            F(x) where x = (K - K_c) × N^ν
        asymptotic_forms: Dict
            Behavior as x → 0 and x → ∞
        universality_class: str
            Which universality class (mean-field, Ising, etc.)
        validation: Dict
            Comparison with numerical data
    """
```

### Component 6: Main Derivation

```python
def derive_functional_forms_from_geometric_measure_theory(
    formula_version: float,
    N: int,
    K: float,
    K_c: float
) -> Dict[str, Any]:
    """
    Complete derivation combining all components.
    """
    
    # 1. Derive critical exponents
    exponents = _derive_critical_exponents(N, K_c)
    
    # 2. Derive power law structure
    power_law = _derive_power_law_form(N, K, K_c)
    
    # 3. Derive √N scaling
    sqrt_n = _derive_sqrt_n_exponent(N)
    
    # 4. Derive corrections
    corrections = _derive_correction_terms(N, K, K_c)
    
    # 5. Construct scaling function
    scaling = _construct_scaling_function(N)
    
    # 6. Synthesize into final formula
    theoretical_formula = _synthesize_theoretical_formula(
        exponents, power_law, sqrt_n, corrections, scaling
    )
    
    # 7. Compare to V9.1
    v91_comparison = _compare_to_empirical_v91(theoretical_formula, N, K, K_c)
    
    return {
        'theoretical_formula': theoretical_formula,
        'formula_components': {
            'exponents': exponents,
            'power_law': power_law,
            'sqrt_n_scaling': sqrt_n,
            'corrections': corrections,
            'scaling_function': scaling
        },
        'v91_comparison': v91_comparison,
        'derivation_status': 'THEORETICAL' | 'HEURISTIC' | 'CONJECTURAL',
        'mathematical_rigor': float,  # 0-1 scale
        'predicted_parameters': {
            'alpha': float,
            'beta': float,
            'gamma': float
        },
        'empirical_v91_parameters': {
            'alpha': 2.5,  # Known from V9.1 fit
        },
        'parameter_matching': Dict[str, bool],  # Which parameters match?
        'proof_sketch': List[str],
        'key_theorems_used': List[str],
        'assumptions': List[str],
        'gaps': List[str]
    }
```

## Expected Theoretical Formula Structure

Based on GMT, the formula should emerge as:

```python
V_theoretical(N, K, K_c) = 1 - (K_c/K)^(α√N) × [
    1 + 
    (c₁ log N) / N +                    # Logarithmic correction
    c₂ exp(-β√N / log N) +              # Exponential correction
    (c₃ / N) × (K/K_c - 1)^γ           # Polynomial correction
]

Where:
- α ≈ 2.5 (from GMT + Kakeya bounds)
- c₁, c₂, c₃ are GMT-derived constants
- β, γ are subleading exponents
```

## Validation Requirements

The theoretical formula must:

1. **Asymptotic Correctness**:
   - V → 1 as K → ∞ (full synchronization)
   - V → 0 as K → K_c⁺ (critical point)
   - V = 0 for K < K_c (below threshold)

2. **Scaling Consistency**:
   - Exponent scales as √N (matches derive_sqrt_n_scaling_from_kakeya)
   - Finite-size corrections ~ 1/√N or log N / N
   - No unphysical divergences

3. **Empirical Agreement**:
   - Within 10% of V9.1 predictions
   - Same qualitative behavior for all N ≥ 10
   - Correctly predicts α ≈ 2.5 ± 0.5

4. **Mathematical Consistency**:
   - All parameters have GMT derivation
   - No free parameters (all predicted from theory)
   - Satisfies dimensional analysis

## Comparison to V9.1 Helper

```python
def _compare_to_empirical_v91(theoretical_formula: Dict, N: int, K: float, K_c: float) -> Dict[str, Any]:
    """
    Detailed comparison between theoretical derivation and empirical V9.1.
    
    Returns:
        parameter_comparison: Dict
            Side-by-side theoretical vs. empirical
        prediction_accuracy: float
            How close are the formulas? (0-1)
        structural_match: bool
            Do they have same functional form?
        discrepancies: List[str]
            Where do they differ and why?
        theoretical_improvements: List[str]
            What does GMT predict that V9.1 missed?
    """
```

## Key Mathematical Theorems to Reference

1. **Minkowski Content Theorem**: Relates dimension to measure scaling
2. **Hausdorff Dimension Theorem**: Bounds on fractal dimension
3. **Finite-Size Scaling Theory**: Corrections near critical points
4. **Mean-Field Critical Exponents**: β = 0.5, ν = 0.5 for Kuramoto
5. **Kakeya Measure Bounds**: From previous derivation

## Success Criteria

**Minimum Acceptable**:
- ✅ Derives (K_c/K)^exponent form from GMT
- ✅ Explains why √N appears in exponent
- ✅ Identifies correction terms
- ✅ Within 20% of V9.1 parameter values
- ✅ Clear proof sketch with assumptions

**Excellent Implementation**:
- ✅ All of above PLUS:
- ✅ Predicts α = 2.5 ± 0.5 from first principles
- ✅ Derives all correction term coefficients
- ✅ Within 10% of V9.1 predictions
- ✅ Rigorous GMT theorems at each step
- ✅ Explains why V9.1 works so well (< 5% error)
- ✅ Suggests theoretical improvements to V9.1

## Common Pitfalls to Avoid

1. **Circular reasoning**: Don't use V9.1 to derive V9.1
2. **Wrong asymptotic**: Ensure V → 1 as K → ∞
3. **Dimension confusion**: d_boundary vs. d_phase_space
4. **Free parameters**: All constants should be derived, not fitted
5. **Missing corrections**: Logarithmic terms are essential for large N

---

**BEGIN IMPLEMENTATION**: Please implement `derive_functional_forms_from_geometric_measure_theory` and all helper functions with maximum mathematical rigor. The goal is to show V9.1's functional form is not arbitrary but emerges naturally from geometric measure theory + Kakeya constraints + phase transition theory.