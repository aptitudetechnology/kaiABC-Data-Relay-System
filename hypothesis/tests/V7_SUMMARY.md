# Formula V7: Asymmetric Boundary Layer

## Motivation

Empirical data suggests the transition region is **asymmetric** around K_c:
- **Below K_c**: Sharp cutoff (hard to sync without enough coupling)
- **Above K_c**: Gradual growth (competing with partial sync states)

## Mathematical Structure

V7 divides the coupling space into **4 regimes** with asymmetric widths:

### 1. Deep Subcritical (K < 0.9×K_c)
```
V = 0.20 × (K/K_lower)²
```
- Quadratic growth from zero
- Metastable clusters form transiently
- Explains 6-16% sync below K_c

### 2. Lower Boundary (0.9 ≤ K < 1.0×K_c)
```
V = base + (0.25 - base) × (1 - exp(-4×margin))
```
- **Narrow** transition region (Δ = 0.10√(10/N))
- Exponential approach to K_c
- Sharp because disorder dominates

### 3. Upper Boundary (1.0 ≤ K < 1.5×K_c)
```
V = 0.25 + 0.70 / (1 + exp(-exponent×margin))
exponent = 2.0 × √N × margin
```
- **Wide** transition region (Δ = 0.50√(10/N))
- Sigmoid with √N scaling (like V4)
- Gradual because partial sync competes with full sync

### 4. Supercritical (K ≥ 1.5×K_c)
```
V = 0.95 + 0.05 × (1 - exp(-2×N×excess))
```
- Exponential approach to 100%
- Full sync dominates

## Key Innovation: Asymmetric Widths

```
δ_lower = 0.10√(10/N) ≈ 0.10 for N=10  (narrow)
δ_upper = 0.50√(10/N) ≈ 0.50 for N=10  (wide)
```

This 5:1 ratio reflects the physical asymmetry:
- Below K_c: System snaps into disorder quickly
- Above K_c: System gradually transitions through partial→full sync

## Expected Performance

### Below Critical (K=0.8-0.9)
- Should predict ~13% at K=0.8, ~20% at K=0.9
- Better than V4's 8%/0% predictions
- Comparable to empirical 6-20%

### Transition Regime (K=1.0-1.5)
- Uses √N scaling like V4 (proven best)
- But centers sigmoid at K=K_c with wider spread
- **Target**: 5-8% error (improve on V4's 10.5%)

### Strong Coupling (K>1.5)
- Should match V4's excellent 1-5% error
- Exponential saturation at 95-100%

## Physical Interpretation

**Why wider above K_c?**

Below K_c:
- Disorder is attractive: Random phases stabilize each other
- Requires significant coupling to overcome
- Sharp threshold behavior

Above K_c:
- Partial synchronization states exist
- N/2 nodes sync, rest wander
- These compete with full sync
- Gradual transition as K increases

## Hypothesis

If V7 achieves **<7% overall error** and **<9% transition error**, it validates:
1. ✅ Asymmetric boundary layer theory
2. ✅ Competing sync states above K_c
3. ✅ Sharp disorder→sync transition below K_c

## Test Plan

- 200 trials per K value (4× better statistics than V5/V6 test)
- Compare V7 vs V4 head-to-head
- If V7 wins: Validates asymmetric physics
- If V4 wins: Symmetric transition is sufficient
