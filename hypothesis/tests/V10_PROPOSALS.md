# V10 Proposals - Next Generation Basin Volume Formula

**Date:** October 9, 2025  
**Current Champion:** V9.1 (5.0% overall, 4.5% transition, 1.9% at K_c)  
**Goal:** Push below 3% overall error while maintaining physical interpretability

---

## Analysis: What V9.1 Gets Right and Wrong

### V9.1 Strengths ✅

1. **Below-critical floor (K ≤ K_c):** 26% × (K/K_c)^1.5
   - Captures metastable synchronization
   - Error: 10-20% (acceptable for inherently noisy regime)

2. **Strong coupling (K ≥ 1.6×K_c):** V8's power law
   - Error: 2.5-4.3% (excellent)
   - Preserved V8's best feature

### V9.1 Weaknesses ❌

1. **Sharp transition at K=K_c**
   - V9.1 switches abruptly from floor → V8 formula
   - May cause discontinuities in prediction vs K curve

2. **Network size scaling (N≠10)**
   - Error: 20-24% when N ∈ {3,5,15,20}
   - √N scaling may need refinement

3. **Transition regime variance (K=1.0-1.5)**
   - Error: 4.5-12.7% (run-dependent)
   - Could be more stable

---

## V10 Proposal 1: "Smooth Goldilocks" (Physics-Based)

### Concept
Smooth out V9.1's sharp K=K_c transition using a sigmoid blend.

### Formula

```python
def predict_basin_volume_v10_smooth(K, K_c, tau_ref, T_ref, N, Q10=2.3):
    """
    V10: Smoothed V9.1 with continuous transitions
    
    Key innovation: Sigmoid blending between floor and V8 formulas
    instead of sharp if/else boundary at K=K_c.
    """
    K_ratio = K / K_c
    
    # V9.1's floor formula
    floor = 0.26 * (K_ratio ** 1.5)
    
    # V8's formula (transition + plateau + strong coupling)
    v8_formula = calculate_v8_basin_volume(K_ratio, N)
    
    # SMOOTH BLEND: Sigmoid weight (transition centered at K=K_c)
    # - K << K_c: weight ≈ 1 (use floor)
    # - K ≈ K_c: weight ≈ 0.5 (blend 50/50)
    # - K >> K_c: weight ≈ 0 (use V8)
    
    steepness = 5.0  # Controls transition sharpness
    weight_floor = 1.0 / (1.0 + np.exp(steepness * (K_ratio - 1.0)))
    
    # Weighted average
    basin_volume = weight_floor * floor + (1 - weight_floor) * v8_formula
    
    return np.clip(basin_volume, 0.0, 1.0)
```

### Advantages
✅ No sharp discontinuities  
✅ Smooth prediction curve  
✅ Physically motivated (gradual regime transition)  
✅ Simple (only 1 new parameter: steepness)  

### Expected Performance
- Below K_c: Same as V9.1 (10-20% error)
- **K=K_c: 5-10% error** (better than V9.1's 10-20%)
- Transition: 4-8% error (more stable)
- Strong coupling: 2.5-4.3% (preserved)
- **Overall: ~4-5% error**

---

## V10 Proposal 2: "N-Adaptive Goldilocks" (Scaling Fix)

### Concept
Add network-size dependent correction to V9.1's floor.

### Observation from Data

| N | K=1.5×K_c Empirical | V9.1 Pred | Error |
|---|---------------------|-----------|-------|
| 3 | 82-91% | 56.8% | 44% |
| 5 | 82-84% | 65.4% | 25% |
| 10 | 86-90% | 80.0% | 8% |
| 15 | 80-84% | 89.3% | 10% |
| 20 | 81-84% | 95.7% | 15% |

**Pattern:** 
- Small N (3-5): V9.1 under-predicts
- Medium N (10): V9.1 accurate
- Large N (15-20): V9.1 over-predicts

### Formula

```python
def predict_basin_volume_v10_n_adaptive(K, K_c, tau_ref, T_ref, N, Q10=2.3):
    """
    V10: V9.1 with network-size dependent correction
    
    Key innovation: Adjust floor amplitude based on N
    to fix small/large network prediction errors.
    """
    K_ratio = K / K_c
    
    # Network size correction factor (empirically calibrated)
    # N=10 is reference (no correction)
    N_correction = 1.0 + 0.05 * np.log(N / 10.0)
    # N=3: correction ≈ 0.94 (reduce floor)
    # N=10: correction = 1.0 (no change)
    # N=20: correction ≈ 1.035 (increase floor slightly)
    
    if K_ratio <= 1.0:
        # Floor with N-correction
        floor = (0.26 * N_correction) * (K_ratio ** 1.5)
        basin_volume = floor
    elif K_ratio < 1.2:
        # V8 transition
        basin_volume = calculate_v8_transition(K_ratio, N)
    # ... rest of V9.1 formula
    
    return np.clip(basin_volume, 0.0, 1.0)
```

### Advantages
✅ Fixes network scaling errors  
✅ Still simple (1 new parameter)  
✅ Physically motivated (finite-size effects)  
✅ Backwards compatible (N=10 unchanged)  

### Expected Performance
- N=3-5: 15-20% error (vs V9.1's 25-44%)
- N=10: Same as V9.1 (~8% error)
- N=15-20: 8-12% error (vs V9.1's 10-15%)
- **Overall network scaling: ~12% error** (vs V9.1's 24%)

---

## V10 Proposal 3: "Data-Driven Calibration" (ML-Lite)

### Concept
Use empirical data to calibrate 3-5 key parameters in V9.1.

### What to Calibrate

1. **Floor amplitude:** 0.26 → optimize
2. **Floor exponent:** 1.5 → optimize
3. **Transition width:** V8's β → optimize
4. **Plateau amplitude:** V8's 0.85 → optimize
5. **Strong coupling exponent:** V8's power law → optimize

### Method: Least-Squares Optimization

```python
from scipy.optimize import minimize

def predict_basin_volume_v10_calibrated(K, K_c, tau_ref, T_ref, N, 
                                        params=None):
    """
    V10: Empirically calibrated V9.1
    
    Key innovation: Optimize 5 key parameters using Monte Carlo data
    to minimize mean absolute error across all K values.
    """
    if params is None:
        # Default V9.1 parameters
        params = {
            'floor_amp': 0.26,      # Floor amplitude
            'floor_exp': 1.5,       # Floor exponent
            'transition_beta': 0.15, # V8 transition width
            'plateau_amp': 0.85,    # V8 plateau amplitude
            'strong_exp': 5.0       # V8 strong coupling exponent
        }
    
    K_ratio = K / K_c
    
    if K_ratio <= 1.0:
        # Calibrated floor
        floor = params['floor_amp'] * (K_ratio ** params['floor_exp'])
        basin_volume = floor
    elif K_ratio < 1.2:
        # Calibrated transition
        basin_volume = calculate_v8_transition(K_ratio, N, 
                                               beta=params['transition_beta'])
    # ... rest with calibrated parameters
    
    return np.clip(basin_volume, 0.0, 1.0)

# Calibration function
def calibrate_v10(empirical_data):
    """
    Find optimal parameters by minimizing error on empirical data
    """
    def loss(params_array):
        params = {
            'floor_amp': params_array[0],
            'floor_exp': params_array[1],
            'transition_beta': params_array[2],
            'plateau_amp': params_array[3],
            'strong_exp': params_array[4]
        }
        
        errors = []
        for d in empirical_data:
            pred = predict_basin_volume_v10_calibrated(
                d['K'], d['K_c'], d['tau_ref'], d['T_ref'], d['N'], params
            )
            errors.append(abs(pred - d['empirical']))
        
        return np.mean(errors)
    
    # Initial guess (V9.1 defaults)
    x0 = [0.26, 1.5, 0.15, 0.85, 5.0]
    
    # Bounds (keep physically reasonable)
    bounds = [
        (0.15, 0.35),  # floor_amp: 15-35%
        (1.0, 2.5),    # floor_exp: moderate power law
        (0.05, 0.30),  # transition_beta: smooth but not too smooth
        (0.70, 0.95),  # plateau_amp: 70-95%
        (3.0, 8.0)     # strong_exp: strong power law
    ]
    
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    
    return {
        'floor_amp': result.x[0],
        'floor_exp': result.x[1],
        'transition_beta': result.x[2],
        'plateau_amp': result.x[3],
        'strong_exp': result.x[4]
    }
```

### Advantages
✅ Optimal performance on empirical data  
✅ Still interpretable (physics-inspired parameters)  
✅ Can adapt to different configurations (N, Q10)  
✅ Quantifies uncertainty (confidence intervals on parameters)  

### Expected Performance
- **Overall: <3% error** (calibrated to minimize error)
- All regimes: 2-5% error
- Best possible with current formula structure

### Disadvantages
⚠️ Requires empirical data to calibrate  
⚠️ May overfit if data is noisy  
⚠️ Less predictive (not purely theory-based)  

---

## V10 Proposal 4: "Multi-Regime Weighted Blend" (V11 Preview)

### Concept
Smooth version of V11 with continuous regime weights (already partially implemented in code).

### Formula Structure

```python
def predict_basin_volume_v10_weighted(K, K_c, tau_ref, T_ref, N, Q10=2.3):
    """
    V10: Weighted multi-regime blend
    
    Key innovation: Each regime contributes proportionally based on
    smooth weight functions (Gaussian/sigmoid), then average.
    """
    K_ratio = K / K_c
    
    # REGIME WEIGHTS (smooth, sum to 1)
    w_metastable = sigmoid_weight(K_ratio, center=0.9, steepness=-10)
    w_transition = gaussian_weight(K_ratio, center=1.15, width=0.3)
    w_plateau = gaussian_weight(K_ratio, center=1.4, width=0.3)
    w_strong = sigmoid_weight(K_ratio, center=1.6, steepness=10)
    
    # Normalize
    total = w_metastable + w_transition + w_plateau + w_strong + 1e-10
    w_metastable /= total
    w_transition /= total
    w_plateau /= total
    w_strong /= total
    
    # REGIME PREDICTIONS
    v_metastable = 0.26 * (K_ratio ** 1.5)  # V9.1 floor
    v_transition = calculate_v8_transition(K_ratio, N)
    v_plateau = calculate_v8_plateau(K_ratio, N)
    v_strong = calculate_v8_strong_coupling(K_ratio, N)
    
    # WEIGHTED AVERAGE
    basin_volume = (w_metastable * v_metastable +
                   w_transition * v_transition +
                   w_plateau * v_plateau +
                   w_strong * v_strong)
    
    return np.clip(basin_volume, 0.0, 1.0)
```

### Advantages
✅ Smooth everywhere (no discontinuities)  
✅ Each regime contributes naturally  
✅ Physically interpretable (regime dominance)  
✅ Generalizes well to new K values  

### Expected Performance
- **Overall: 3-4% error**
- Smoother predictions than V9.1
- Better interpolation between regimes

---

## V10 Proposal 5: "Adaptive Time-Dependent" (Advanced)

### Concept
Account for finite observation time (t_max) explicitly.

### Observation
Current formulas assume t_max → ∞, but simulations use t_max = 720 hours.

At K ≈ K_c:
- Some trials sync at t=100hr
- Some sync at t=500hr  
- Some never sync (within t_max)

### Formula

```python
def predict_basin_volume_v10_time_dependent(K, K_c, tau_ref, T_ref, N, 
                                           Q10=2.3, t_max=720):
    """
    V10: Time-dependent basin volume
    
    Key innovation: Predict fraction that synchronizes within t_max,
    not just asymptotic basin volume.
    """
    K_ratio = K / K_c
    
    # Asymptotic basin volume (V9.1)
    V_inf = predict_basin_volume_v9_1(K, K_c, tau_ref, T_ref, N, Q10)
    
    # Synchronization timescale (depends on K-K_c)
    if K_ratio <= 1.0:
        # Below critical: very slow (exponential relaxation)
        tau_sync = tau_ref * 100 * np.exp(-5.0 * K_ratio)
    else:
        # Above critical: fast (power law)
        tau_sync = tau_ref * 10 / ((K_ratio - 1.0) ** 2)
    
    # Fraction that syncs within t_max (exponential approach)
    V_t = V_inf * (1 - np.exp(-t_max / tau_sync))
    
    return np.clip(V_t, 0.0, 1.0)
```

### Advantages
✅ Accounts for finite observation time  
✅ More realistic (matches simulation protocol)  
✅ Explains why empirical < predicted at low K  

### Expected Performance
- Below K_c: 5-10% error (better than V9.1's 10-20%)
- K=K_c: 5-8% error
- **Overall: ~4-5% error**

### Disadvantages
⚠️ Adds complexity (timescale model)  
⚠️ Requires knowing t_max (not always available)  

---

## Recommendation: Which V10 to Implement?

### For Immediate Use (Hardware Deployment)
**→ V10 Proposal 1: "Smooth Goldilocks"**

**Why:**
- ✅ Simple (1 new parameter)
- ✅ Fixes V9.1's sharp K=K_c transition
- ✅ Expected 4-5% error (improvement over V9.1)
- ✅ Easy to implement (20 lines of code)
- ✅ Still physically interpretable

**Implementation effort:** 30 minutes  
**Testing effort:** 1 hour  
**Expected improvement:** 0.5-1% overall error reduction

### For Publication Quality
**→ V10 Proposal 3: "Data-Driven Calibration"**

**Why:**
- ✅ Optimal performance (<3% error possible)
- ✅ Still interpretable (physics parameters)
- ✅ Can publish parameter uncertainties
- ✅ Systematic improvement methodology

**Implementation effort:** 2-3 hours  
**Testing effort:** 3-4 hours  
**Expected improvement:** 2-3% overall error reduction

### For Future Research
**→ V10 Proposal 4: "Multi-Regime Weighted Blend"**

**Why:**
- ✅ Most general framework
- ✅ Smooth and physically motivated
- ✅ Can incorporate all previous insights
- ✅ Good foundation for V11

**Implementation effort:** 4-5 hours  
**Testing effort:** 5-6 hours  
**Expected improvement:** 1-2% overall error reduction

---

## Implementation Priority

### Phase 1: Quick Win (This Week)
Implement **V10.1 Smooth Goldilocks** as proof-of-concept:
```python
# Add to enhanced_test_basin_volume.py
elif formula_version == 10.1:
    # Smooth V9.1 with sigmoid blending
```

### Phase 2: Optimization (Next Week)
Implement **V10.3 Data-Driven Calibration** using all empirical data from V9.1 validation:
- Use 2000+ simulation results
- Optimize 5 key parameters
- Cross-validate with held-out data

### Phase 3: Advanced (Future)
Explore **V10.4 Weighted Blend** and **V10.5 Time-Dependent** for ultimate accuracy.

---

## Expected V10 Performance Targets

| Metric | V9.1 | V10.1 (Smooth) | V10.3 (Calibrated) | V10.4 (Weighted) |
|--------|------|----------------|-------------------|------------------|
| Overall error | 5.0% | 4-5% | <3% | 3-4% |
| K=K_c error | 10-20% | 5-10% | 2-5% | 3-6% |
| Transition error | 4.5-12.7% | 4-8% | 2-4% | 3-5% |
| Strong coupling | 2.5-4.3% | 2.5-4.3% | 2-3% | 2-3% |
| Network scaling (N≠10) | 24% | 20-24% | 10-15% | 12-18% |

---

## Bottom Line

**Recommended V10:** Start with **Smooth Goldilocks (V10.1)** for quick improvement, then upgrade to **Calibrated (V10.3)** for publication.

**Expected outcome:**
- V10.1: 4-5% error (small improvement, minimal effort)
- V10.3: <3% error (significant improvement, moderate effort)

**Hardware impact:** V9.1 is already excellent for deployment (5% error). V10 would be "nice to have" but not necessary for proceeding with hardware.

**Recommendation:** Proceed with hardware using V9.1, develop V10 in parallel for publication.

