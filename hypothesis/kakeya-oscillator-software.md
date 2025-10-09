# KaiABC Software-Only Hypothesis Tests

## 🎯 Why Test in Software First?

**Advantages:**
- ✅ Zero hardware cost
- ✅ Perfect parameter control
- ✅ 10,000× faster than real-time
- ✅ Exhaustive parameter sweeps
- ✅ Validates theory before spending $400+
- ✅ Identifies which hardware tests are worth doing

**Philosophy:** If it doesn't work in simulation, it won't work in hardware.

---

## 🧪 Software Hypothesis 1: Basin of Attraction (Primary Test)

### **Research Question**
Does the basin volume formula accurately predict convergence rate across parameter space?

### **H₀ (Null):**
The predicted basin volume formula **does NOT** match simulated convergence rates:
```
V_basin/V_total ≈ (1 - 1.5·σ_ω/⟨ω⟩)^N
```

### **H₁ (Alternate):**
Simulated convergence rate matches predicted basin volume within ±15% across:
- N ∈ [3, 5, 10, 20, 50]
- K/K_c ∈ [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
- σ_T ∈ [2, 5, 10, 15]°C

### **Why This Matters**
- Tests the **Kakeya → Kuramoto connection** (your core theoretical claim)
- Predicts hardware success rate before buying devices
- If basin = 5%, don't waste time on hardware
- If basin = 80%, hardware should be easy

### **Computational Experiment**

```python
# Pseudocode
for N in [3, 5, 10, 20, 50]:
    for K_ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for sigma_T in [2, 5, 10, 15]:
            # Calculate theoretical basin
            K_c = calculate_critical_coupling(sigma_T, Q10=1.1)
            K = K_ratio * K_c
            V_theory = basin_volume_formula(N, sigma_T, K)
            
            # Run Monte Carlo
            converged = 0
            for trial in range(1000):
                phases = random_initial_phases(N)
                omegas = temperature_frequencies(sigma_T, Q10=1.1)
                
                result = simulate_kuramoto(
                    phases, omegas, K, 
                    t_max=30_days, 
                    temp_profile=diurnal_cycle
                )
                
                if result.order_parameter[-1] > 0.90:
                    converged += 1
            
            V_empirical = converged / 1000
            error = abs(V_theory - V_empirical) / V_theory
            
            results.append({
                'N': N, 'K_ratio': K_ratio, 'sigma_T': sigma_T,
                'V_theory': V_theory, 'V_empirical': V_empirical,
                'error': error
            })

# Statistical test
mean_error = np.mean([r['error'] for r in results])
accept_H1 = mean_error < 0.15  # Within 15% error
```

### **Success Criteria**
- ✅ **Strong**: Mean error < 10% across all parameters
- ⚠️ **Moderate**: Mean error < 20%, identifies problematic regimes
- ❌ **Failure**: Mean error > 30%, formula needs revision

### **Estimated Runtime**
- 5 × 6 × 4 = 120 parameter combinations
- 1000 trials each = 120,000 simulations
- ~0.1 sec per simulation = **3.3 hours on single CPU**
- Parallelize across 8 cores = **25 minutes**

---

## 🧪 Software Hypothesis 2: Critical Coupling Scaling

### **Research Question**
Does K_c scale correctly with temperature variance and network size?

### **H₀ (Null):**
The formula K_c = 2σ_ω **does NOT** predict the 50% convergence threshold within ±30% error.

### **H₁ (Alternate):**
Binary search in simulation finds K_c,empirical that matches K_c,theory within ±30% for:
- Q₁₀ ∈ [0.8, 1.0, 1.1, 1.2, 1.5]
- σ_T ∈ [2, 5, 10, 15]°C
- N ∈ [3, 10, 50]

### **Why This Matters**
- Validates your power budget calculations (need K = 2.4×K_c)
- If K_c formula is wrong, your entire energy model breaks
- Cheaper to find errors in simulation than after hardware deployment

### **Computational Experiment**

```python
# Binary search for empirical K_c
def find_critical_coupling(N, sigma_T, Q10, trials=100):
    K_theory = 2 * frequency_variance(sigma_T, Q10)
    
    # Binary search between 0.5× and 2× theoretical
    K_low, K_high = 0.5 * K_theory, 2.0 * K_theory
    
    for iteration in range(10):  # 10 iterations = 0.1% precision
        K_mid = (K_low + K_high) / 2
        
        converged = simulate_batch(N, K_mid, sigma_T, Q10, trials)
        convergence_rate = converged / trials
        
        if convergence_rate < 0.50:
            K_low = K_mid  # Need stronger coupling
        else:
            K_high = K_mid
    
    K_empirical = (K_low + K_high) / 2
    error = abs(K_empirical - K_theory) / K_theory
    
    return K_empirical, error

# Test across parameter space
results = []
for N in [3, 10, 50]:
    for sigma_T in [2, 5, 10, 15]:
        for Q10 in [0.8, 1.0, 1.1, 1.2, 1.5]:
            K_emp, err = find_critical_coupling(N, sigma_T, Q10)
            results.append({'N': N, 'sigma_T': sigma_T, 
                          'Q10': Q10, 'error': err})

accept_H1 = np.mean([r['error'] for r in results]) < 0.30
```

### **Success Criteria**
- ✅ **Strong**: <10% error → Formula is highly accurate
- ⚠️ **Moderate**: 10-30% error → Usable with safety margin
- ❌ **Failure**: >50% error → Need different coupling formula

### **Estimated Runtime**
- 3 × 4 × 5 = 60 parameter sets
- 10 iterations × 100 trials = 1000 sims per set
- 60,000 simulations × 0.1 sec = **100 minutes**

---

## 🧪 Software Hypothesis 3: Synchronization Time Prediction

### **Research Question**
Does the synchronization time formula accurately predict convergence speed?

```
τ_sync ≈ (1/(K - K_c)) · ln(N/ε) · τ_ref
```

### **H₀ (Null):**
Predicted sync time has >50% error compared to simulation.

### **H₁ (Alternate):**
For K > 2K_c, predicted τ_sync matches simulation within ±50% for 80% of trials.

### **Why This Matters**
- Your hardware test duration depends on this (16 days vs. 60 days?)
- If prediction is wrong by 3×, you'll give up before sync happens
- Validates whether your theoretical model captures dynamics

### **Computational Experiment**

```python
def measure_sync_time(N, K, sigma_T, Q10, trials=100):
    K_c = 2 * frequency_variance(sigma_T, Q10)
    
    # Theoretical prediction
    tau_ref = 24  # hours
    epsilon = 0.1  # order parameter threshold
    tau_predicted = (1/(K - K_c)) * np.log(N/epsilon) * tau_ref
    
    # Measure empirical sync times
    sync_times = []
    for trial in range(trials):
        phases = random_initial_phases(N)
        omegas = temperature_frequencies(sigma_T, Q10)
        
        result = simulate_kuramoto(phases, omegas, K, t_max=60*24)
        
        # Find first time R > 0.90 for 48 hours
        sync_time = detect_sync_time(result.R, threshold=0.90)
        
        if sync_time is not None:
            sync_times.append(sync_time)
    
    tau_empirical = np.median(sync_times)
    error = abs(tau_empirical - tau_predicted) / tau_predicted
    success_rate = len(sync_times) / trials
    
    return tau_empirical, tau_predicted, error, success_rate

# Test for different coupling strengths
results = []
for K_ratio in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    K_c = 0.042  # Fixed for Q10=1.1, sigma_T=5
    K = K_ratio * K_c
    
    tau_emp, tau_pred, err, success = measure_sync_time(
        N=10, K=K, sigma_T=5, Q10=1.1
    )
    
    results.append({
        'K_ratio': K_ratio,
        'tau_empirical': tau_emp,
        'tau_predicted': tau_pred,
        'error': err,
        'success_rate': success
    })

# Accept H1 if most predictions are within 2×
accurate_predictions = sum(r['error'] < 0.50 for r in results)
accept_H1 = accurate_predictions >= 0.80 * len(results)
```

### **Success Criteria**
- ✅ **Strong**: <30% error → Reliable timing prediction
- ⚠️ **Moderate**: 30-70% error → Order-of-magnitude correct
- ❌ **Failure**: >100% error → Formula missing key physics

### **Estimated Runtime**
- 6 K_ratio values × 100 trials = 600 simulations
- Each sim runs 60 days = longer runtime
- ~1 sec per simulation = **10 minutes**

---

## 🧪 Software Hypothesis 4: Diurnal Temperature Entrainment

### **Research Question**
Can Kuramoto coupling overcome diurnal temperature cycles to achieve synchronization?

### **H₀ (Null):**
The 24-hour temperature cycle acts as a **zeitgeber** (entrainment signal) that dominates Kuramoto coupling, preventing independent phase synchronization.

### **H₁ (Alternate):**
For K > 2K_c, the network achieves phase synchronization (R > 0.90) that is **independent** of the temperature phase, with individual oscillator phases **not** locked to the temperature cycle.

### **Why This Matters**
- Your hardware test uses **real outdoor temperature** with diurnal cycles
- Temperature varies as: T(t) = 20°C + 8°C·cos(2π(t-6hr)/24hr)
- This creates a potential **entrainment signal** that could dominate coupling
- Need to verify that phase synchronization is due to **Kuramoto dynamics**, not just "all nodes tracking temperature"

### **Computational Experiment**

```python
def diurnal_temperature_profile(t_hours, T_mean=20, T_amplitude=8, phase_shift=6):
    """
    Natural outdoor temperature cycle
    - Coldest at 6 AM (sunrise): T_mean - T_amplitude = 12°C
    - Hottest at 6 PM (sunset): T_mean + T_amplitude = 28°C
    - Smooth sinusoidal transition
    """
    T = T_mean + T_amplitude * np.cos(2*np.pi * (t_hours - phase_shift) / 24)
    return T

def test_temperature_entrainment(N, K, sigma_T, Q10, trials=100):
    """Test if sync is independent of temperature phase"""
    
    independent_sync = 0  # Phases NOT locked to temperature
    temperature_locked = 0  # Phases locked to temperature cycle
    
    for trial in range(trials):
        phases = random_initial_phases(N)
        omegas_base = temperature_frequencies(sigma_T, Q10)
        
        # Simulate with time-varying temperature
        t_max = 30 * 24  # 30 days in hours
        R_history = []
        phase_temp_correlation = []
        
        for t in range(t_max):
            T_current = diurnal_temperature_profile(t)
            
            # Adjust frequencies based on current temperature
            omegas = omegas_base * Q10**((25 - T_current)/10)
            
            # Kuramoto update
            for i in range(N):
                coupling = sum(np.sin(phases[j] - phases[i]) 
                              for j in range(N)) / N
                phases[i] += (omegas[i] + K * coupling) * dt
            
            R = calculate_order_parameter(phases)
            R_history.append(R)
            
            # Check correlation with temperature phase
            temp_phase = 2*np.pi * (t % 24) / 24
            corr = np.abs(np.corrcoef(phases, 
                         [temp_phase]*N)[0,1])
            phase_temp_correlation.append(corr)
        
        # Classification criteria
        final_R = np.mean(R_history[-24:])  # Last 24 hours
        final_corr = np.mean(phase_temp_correlation[-24:])
        
        if final_R > 0.90 and final_corr < 0.3:
            independent_sync += 1  # Synchronized but NOT to temperature
        elif final_R > 0.90 and final_corr > 0.7:
            temperature_locked += 1  # Locked to temperature cycle
    
    return {
        'independent_sync_rate': independent_sync / trials,
        'temperature_locked_rate': temperature_locked / trials,
        'total_sync_rate': (independent_sync + temperature_locked) / trials
    }

# Test across coupling strengths
results = []
for K_ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    K_c = 0.042
    K = K_ratio * K_c
    
    result = test_temperature_entrainment(
        N=10, K=K, sigma_T=5, Q10=1.1
    )
    
    results.append({
        'K_ratio': K_ratio,
        **result
    })

# Accept H1 if >70% achieve independent sync for K > 2*K_c
strong_coupling = [r for r in results if r['K_ratio'] >= 2.0]
accept_H1 = np.mean([r['independent_sync_rate'] 
                     for r in strong_coupling]) > 0.70
```

### **Success Criteria**
- ✅ **Strong**: >80% independent sync, <10% temperature-locked
- ⚠️ **Moderate**: >60% independent sync, <30% temperature-locked  
- ❌ **Failure**: <40% independent sync, >50% temperature-locked

### **Estimated Runtime**
- 6 K_ratio values × 100 trials × 30 days = 600 long simulations
- ~2 sec per simulation = **20 minutes**

---

## 🧪 Software Hypothesis 5: Temperature Compensation Robustness

### **Research Question**
How sensitive is synchronization to Q₁₀ accuracy?

### **H₀ (Null):**
System fails to synchronize if Q₁₀ varies by ±20% across nodes.

### **H₁ (Alternate):**
Network still achieves R > 0.80 even with heterogeneous Q₁₀ ∈ [0.9, 1.3].

### **Why This Matters**
- Real hardware will have manufacturing variance
- Your software implementation might drift over time
- Tests robustness to imperfect temperature compensation

### **Computational Experiment**

```python
def test_heterogeneous_Q10(N, K, sigma_T, Q10_range, trials=100):
    """Test sync with different Q10 per node"""
    
    converged = 0
    for trial in range(trials):
        # Each node gets random Q10 from range
        Q10_values = np.random.uniform(Q10_range[0], Q10_range[1], N)
        
        # Generate frequency distributions
        omegas_list = []
        for i in range(N):
            omega_i = temperature_frequencies(sigma_T, Q10_values[i])
            omegas_list.append(omega_i)
        
        # Simulate with heterogeneous Q10
        phases = random_initial_phases(N)
        result = simulate_kuramoto_heterogeneous(
            phases, omegas_list, K, t_max=30*24
        )
        
        if result.R[-1] > 0.80:
            converged += 1
    
    return converged / trials

# Test increasing heterogeneity
results = []
for Q10_spread in [0.0, 0.1, 0.2, 0.3, 0.4]:
    Q10_range = [1.1 - Q10_spread, 1.1 + Q10_spread]
    
    conv_rate = test_heterogeneous_Q10(
        N=10, K=0.10, sigma_T=5, Q10_range=Q10_range
    )
    
    results.append({
        'Q10_spread': Q10_spread,
        'convergence_rate': conv_rate
    })

# Accept H1 if >70% converge with ±20% spread
accept_H1 = results[2]['convergence_rate'] > 0.70  # ±20% = index 2
```

### **Success Criteria**
- ✅ **Robust**: >70% sync with ±30% Q₁₀ variance
- ⚠️ **Moderate**: >50% sync with ±20% variance
- ❌ **Fragile**: <30% sync with ±10% variance

### **Estimated Runtime**
- 5 spread levels × 100 trials = 500 simulations
- ~0.1 sec per sim = **1 minute**

---

## 🧪 Software Hypothesis 6: Communication Packet Loss

### **Research Question**
Can the network synchronize despite realistic message failures?

### **H₀ (Null):**
Packet loss >5% prevents synchronization (R < 0.90).

### **H₁ (Alternate):**
Network achieves R > 0.85 with up to 10% packet loss when K = 2.5×K_c.

### **Why This Matters**
- Your hardware uses ESP-NOW wireless mesh (not perfect reliability)
- Outdoor deployment has obstacles, interference, distance effects
- Typical wireless networks have 1-10% packet loss
- Need to know if coupling calculations are robust to stale phase data

### **Computational Experiment**

```python
def simulate_with_packet_loss(N, K, sigma_T, Q10, 
                               packet_loss_rate, trials=100):
    """
    Simulate realistic communication failures
    - Each coupling update has X% chance to fail
    - Node uses old phase estimate from previous timestep
    """
    
    converged = 0
    
    for trial in range(trials):
        phases = random_initial_phases(N)
        phases_stale = np.copy(phases)  # Last known phases
        omegas = temperature_frequencies(sigma_T, Q10)
        
        t_max = 30 * 24  # 30 days
        
        for t in range(t_max):
            T_current = diurnal_temperature_profile(t)
            
            # Update with packet loss
            for i in range(N):
                coupling_force = 0
                
                for j in range(N):
                    # Each message independently fails with probability
                    if np.random.random() > packet_loss_rate:
                        # Fresh data received
                        coupling_force += np.sin(phases[j] - phases[i])
                        phases_stale[j] = phases[j]  # Update cache
                    else:
                        # Use stale data from last successful reception
                        coupling_force += np.sin(phases_stale[j] - phases[i])
                
                # Kuramoto update with temperature compensation
                omega_temp = omegas[i] * Q10**((25 - T_current)/10)
                phases[i] += (omega_temp + K/N * coupling_force) * dt
            
            phases = phases % (2*np.pi)  # Wrap to [0, 2π]
        
        # Check final synchronization
        R = calculate_order_parameter(phases)
        if R > 0.85:
            converged += 1
    
    return converged / trials

# Test across packet loss rates
results = []
for loss_rate in [0.00, 0.01, 0.05, 0.10, 0.15, 0.20]:
    conv_rate = simulate_with_packet_loss(
        N=10, K=0.105, sigma_T=5, Q10=1.1,  # K = 2.5×K_c
        packet_loss_rate=loss_rate
    )
    
    results.append({
        'packet_loss': loss_rate,
        'convergence_rate': conv_rate
    })

# Accept H1 if >70% converge with 10% loss
accept_H1 = results[3]['convergence_rate'] > 0.70  # 10% = index 3
```

### **Success Criteria**
- ✅ **Robust**: >80% sync with 15% packet loss
- ⚠️ **Moderate**: >70% sync with 10% packet loss
- ❌ **Fragile**: <50% sync with 5% packet loss

### **Estimated Runtime**
- 6 loss rates × 100 trials = 600 simulations
- ~0.1 sec per sim = **1 minute**

### **Mitigation Strategies if H₁ Fails:**
```python
# If packet loss breaks synchronization, try:

# Option 1: Increase coupling strength
K = 3.0 * K_c  # More aggressive coupling

# Option 2: Slower update rate with redundancy
update_interval = 2 hours  # Less frequent, more stable

# Option 3: Exponential moving average
phases_ema[j] = 0.8*phases_ema[j] + 0.2*phases_new[j]  # Smooth noise

# Option 4: Add explicit retransmission
if packet_loss_detected:
    retry_send(phase_update)
```

---

## 📊 Comprehensive Test Suite

### **Full Parameter Sweep**

```python
# Master experiment: Test all 6 hypotheses
def run_comprehensive_test():
    results = {
        'basin_volume': test_basin_volume_hypothesis(),
        'critical_coupling': test_critical_coupling_hypothesis(),
        'sync_time': test_sync_time_hypothesis(),
        'temperature_entrainment': test_temperature_entrainment_hypothesis(),
        'Q10_robustness': test_Q10_robustness_hypothesis(),
        'packet_loss': test_packet_loss_hypothesis()
    }
    
    # Generate report
    print("=" * 60)
    print("SOFTWARE VALIDATION RESULTS")
    print("=" * 60)
    
    for test_name, test_result in results.items():
        status = "✅ PASS" if test_result['accept_H1'] else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        print(f"  Error: {test_result['mean_error']:.1%}")
        print(f"  Confidence: {test_result['confidence']:.1%}")
    
    overall_pass = sum(r['accept_H1'] for r in results.values())
    
    print("\n" + "=" * 60)
    print(f"OVERALL: {overall_pass}/6 hypotheses validated")
    
    if overall_pass >= 5:
        print("✅ PROCEED TO HARDWARE TESTING")
        print(f"   Predicted success rate: {estimate_hardware_success()}%")
    elif overall_pass >= 4:
        print("⚠️ CONDITIONAL GO - Address failed hypothesis")
        print(f"   Predicted success rate: {estimate_hardware_success()}%")
    else:
        print("❌ FIX THEORY BEFORE HARDWARE")
        print("   Failed hypotheses:")
        for name, res in results.items():
            if not res['accept_H1']:
                print(f"   - {name}: {res['failure_reason']}")
    
    return results

def detect_synchronization(R_timeseries, threshold=0.90, duration_hours=24):
    """
    Detect when synchronization is achieved and sustained
    
    Sync achieved when R > threshold for continuous duration.
    This prevents counting transient spikes as synchronization.
    
    Returns: First time index where sync is achieved, or None
    """
    for t in range(len(R_timeseries) - duration_hours):
        window = R_timeseries[t:t+duration_hours]
        if np.all(window > threshold):
            return t  # Return sync time in hours
    return None  # Never synchronized
```

---

## 🔄 Backup Plans if Core Hypotheses Fail

### **If Basin Volume << 28% (Hypothesis 1 Fails):**

**Problem:** Random initial phases rarely converge to synchronization.

**Alternative Approaches:**
```python
# Option 1: Adaptive coupling - increase K when R is low
def adaptive_coupling(R, K_base, K_c):
    """Boost coupling when network is desynchronized"""
    if R < 0.5:
        return 5.0 * K_c  # Strong coupling to escape disorder
    elif R < 0.8:
        return 2.5 * K_c  # Moderate coupling
    else:
        return K_base  # Nominal coupling when synchronized

# Option 2: Frequency adaptation - nodes adjust toward mean
def frequency_adaptation(omega_i, omega_mean, adaptation_rate=0.01):
    """Slowly drift frequency toward network average"""
    omega_i += adaptation_rate * (omega_mean - omega_i)
    return omega_i

# Option 3: Leader-follower topology
def leader_follower_coupling(phases, leader_id=0):
    """One node has GPS/RTC, others follow"""
    for i in range(1, N):
        coupling = sin(phases[leader_id] - phases[i])
        phases[i] += K * coupling * dt
```

### **If K_c Formula Wrong (Hypothesis 2 Fails):**

**Problem:** Theoretical K_c = 2σ_ω doesn't match empirical threshold.

**Alternative Formulas:**
```python
# Test alternative critical coupling formulas

# Mean-field theory prediction
K_c_meanfield = (4/np.pi) * sigma_omega

# Finite-size scaling with N dependence
K_c_finite = 2 * sigma_omega * (1 + 1/np.sqrt(N))

# Empirical power-law fit
K_c_empirical = A * sigma_omega**alpha  # Find A, α from data

# Test which formula best predicts empirical threshold
formulas = {
    'conservative': lambda s, N: 2*s,
    'mean_field': lambda s, N: (4/np.pi)*s,
    'finite_size': lambda s, N: 2*s*(1 + 1/np.sqrt(N)),
    'power_law': lambda s, N: 1.8*s**1.1
}

for name, formula in formulas.items():
    test_formula_accuracy(formula)
```

### **If Sync Time Too Slow (Hypothesis 3 Fails):**

**Problem:** Predicted 16 days, but simulation takes 60+ days.

**Acceleration Strategies:**
```python
# Option 1: Initial phase seeding (not random)
def smart_initial_phases(N):
    """Start phases close together instead of random"""
    base_phase = np.random.uniform(0, 2*np.pi)
    phases = base_phase + np.random.normal(0, 0.5, N)
    return phases % (2*np.pi)

# Option 2: Faster update rate
update_interval = 15  # minutes instead of 1 hour

# Option 3: Stronger coupling
K = 5.0 * K_c  # Aggressive coupling for fast sync

# Option 4: Two-stage coupling
if R < 0.7:
    K = 5.0 * K_c  # Strong coupling initially
else:
    K = 2.0 * K_c  # Reduce to save power once near sync
```

### **If Temperature Entrainment Dominates (Hypothesis 4 Fails):**

**Problem:** Oscillators lock to temperature cycle instead of each other.

**Mitigation:**
```python
# Option 1: Detrend temperature signal
T_detrended = T_measured - moving_average(T_measured, window=24hr)

# Option 2: Add high-pass filter to frequency response
if abs(T_current - T_24hr_ago) < threshold:
    omega_adjusted = omega_base  # Ignore slow temperature drift

# Option 3: Explicitly subtract diurnal component
diurnal_phase = 2*np.pi * (hour % 24) / 24
phase_corrected = phase_measured - diurnal_phase
```

### **If Q₁₀ Heterogeneity Breaks Sync (Hypothesis 5 Fails):**

**Problem:** Manufacturing variance in Q₁₀ prevents synchronization.

**Solutions:**
```python
# Option 1: Calibration procedure
for node in network:
    measured_Q10 = calibrate_temperature_response(node)
    node.set_Q10(measured_Q10)  # Use measured value

# Option 2: Frequency voting algorithm
omega_median = median([node.omega for node in network])
for node in network:
    node.omega = 0.9*node.omega + 0.1*omega_median  # Pull toward median

# Option 3: Looser sync criterion
target_R = 0.80  # Accept slightly lower order parameter
```

### **If Packet Loss Breaks Sync (Hypothesis 6 Fails):**

**Problem:** Wireless communication failures prevent convergence.

**Solutions:**
```python
# Option 1: Redundant transmissions
for retry in range(3):
    send_phase_update()
    if ack_received:
        break

# Option 2: Lower update rate with longer averaging
update_interval = 2 hours  # Less frequent but more stable
phase_buffer = []  # Average over multiple successful receptions

# Option 3: Predictive phase estimation
if packet_lost:
    phase_estimated = phase_last + omega * dt  # Dead reckoning

# Option 4: Increase coupling to overwhelm noise
K = 4.0 * K_c  # Stronger coupling compensates for stale data
```

---

## 🎯 Decision Tree

```
Run Software Tests (Total: ~2.5 hours runtime)
│
├─ 6/6 PASS (Perfect Score)
│  └─→ Proceed directly to hardware MVP
│     Expected success rate: >90%
│     Confidence: Very High
│
├─ 5/6 PASS (Strong Evidence)
│  ├─ Identify which hypothesis failed
│  ├─ Apply backup plan from section above
│  └─→ Proceed to hardware with mitigation
│     Expected success rate: >75%
│
├─ 4/6 PASS (Moderate Evidence)
│  ├─ Identify failure patterns
│  ├─ Revise model parameters
│  ├─ Test backup approaches
│  └─→ Re-run software tests (1 hour)
│     Decision: Conditional go/no-go
│
└─ ≤3/6 PASS (Weak Evidence)
   └─→ Major theory revision needed
      Options:
      1. Re-examine Kakeya → Kuramoto connection
      2. Test alternative coupling functions (adaptive K)
      3. Consider frequency adaptation mechanism
      4. Test leader-follower topology
      5. Publish negative results (still valuable!)
      6. Pivot to different synchronization approach
```

---

## 💡 Key Advantages

### **Software Testing First:**
1. **Falsification is cheap** - Find errors in hours, not months
2. **Parameter optimization** - Tune K, N, σ_T before hardware
3. **Risk assessment** - Know expected failure rate
4. **Timeline prediction** - How long should hardware run?

### **What Hardware Adds:**
- Real temperature dynamics (not sinusoidal model)
- Packet loss and communication delays
- Hardware clock drift
- Battery voltage effects
- Environmental noise

### **Hybrid Approach:**
```
Phase 0: Software validation (2 hours) ← START HERE
│
├─ PASS → Phase 1: Lab hardware (1 week, $100)
│  │
│  ├─ PASS → Phase 2: Outdoor deployment (4 weeks, $300)
│  └─ FAIL → Debug hardware issues
│
└─ FAIL → Revise theory (cheap, fast)
```

---

## 🚀 Immediate Next Steps

### **Phase 0: Core Implementation (Week 1 - Priority 1)**

1. **Write Kuramoto Simulator** (Day 1-2)
   ```python
   # Core functions needed:
   - kuramoto_step(phases, omegas, K, dt)
   - calculate_order_parameter(phases)
   - temperature_frequencies(sigma_T, Q10)
   - random_initial_phases(N)
   ```

2. **Implement KaiABC Extensions** (Day 3)
   ```python
   - diurnal_temperature_profile(t)
   - Q10_temperature_compensation(omega_base, T_current, Q10)
   - detect_synchronization(R_timeseries, threshold, duration)
   ```

3. **Run Core Tests** (Day 4)
   - Hypothesis 1: Basin volume (25 min parallelized)
   - Hypothesis 2: Critical coupling (100 min)
   - Hypothesis 3: Sync time (10 min)
   
   **Deliverable:** Know if basic theory is correct

---

### **Phase 1: Realism Extensions (Week 2 - Priority 2)**

4. **Add Environmental Realism** (Day 5)
   ```python
   - Test with diurnal temperature (Hypothesis 4)
   - Test Q10 heterogeneity (Hypothesis 5)
   - Compare with/without realistic effects
   ```

5. **Add Communication Realism** (Day 6)
   ```python
   - Implement packet loss simulation (Hypothesis 6)
   - Test message delay effects
   - Model ESP-NOW range limits
   ```

6. **Parameter Optimization** (Day 7)
   ```python
   - Find optimal N (network size)
   - Find optimal K (coupling strength)
   - Find optimal update_interval
   - Minimize power while ensuring sync
   ```

---

### **Phase 2: Decision Report (Week 3)**

7. **Generate Comprehensive Report** (Day 8)
   ```markdown
   ## Simulation Results Summary
   
   | Hypothesis | Result | Error | Confidence |
   |------------|--------|-------|------------|
   | Basin Volume | ✅ PASS | 8.2% | High |
   | Critical Coupling | ✅ PASS | 12.5% | High |
   | Sync Time | ⚠️ MODERATE | 35% | Medium |
   | Temperature Entrainment | ✅ PASS | 15% | High |
   | Q10 Robustness | ✅ PASS | 9% | High |
   | Packet Loss | ⚠️ MODERATE | 22% | Medium |
   
   **Overall: 5/6 PASS (Strong Evidence)**
   
   **Hardware Recommendation:** GO
   - Predicted success rate: 78%
   - Recommended parameters: N=10, K=0.10, update=1hr
   - Mitigations needed: Increase K to 0.12 for packet loss
   ```

8. **Hardware Go/No-Go Decision** (Day 9)
   - If 5-6/6 pass → Order ESP32 + BME280 hardware
   - If 4/6 pass → Revise parameters, re-run critical tests
   - If ≤3/6 pass → Test backup approaches (adaptive K, etc.)

9. **Prepare Hardware Test Plan** (Day 10)
   - Order bill of materials ($104)
   - Write ESP32 firmware based on optimal parameters
   - Set up MQTT data logging infrastructure
   - Define success criteria matching software predictions

---

### **Resource Requirements**

**Time:**
- Week 1 (Core): 4 days coding + 2.5 hours compute
- Week 2 (Realism): 3 days coding + 1 hour compute  
- Week 3 (Decision): 2 days analysis + reporting
- **Total: 2-3 weeks part-time effort**

**Cost:**
- Software phase: **$0**
- Hardware phase (if approved): **$104**

**Risk Reduction:**
- Catches theory errors before hardware: **Priceless** 🎯
- Optimizes parameters for hardware success
- Provides quantitative success prediction
- Identifies necessary mitigations

---

## 📝 Success Metrics Summary

| Hypothesis | Test | Success Threshold | Runtime |
|------------|------|-------------------|---------|
| **H1: Basin Volume** | Monte Carlo convergence | Mean error <15% | 25 min |
| **H2: Critical Coupling** | Binary search K_c | Error <30% | 100 min |
| **H3: Sync Time** | Time to R>0.90 | Error <50% | 10 min |
| **H4: Temperature Entrainment** | Independence from diurnal | >70% independent | 20 min |
| **H5: Q10 Robustness** | Heterogeneous Q10 | >70% converge ±20% | 1 min |
| **H6: Packet Loss** | Wireless failures | >70% converge at 10% | 1 min |
| | | **TOTAL** | **~2.5 hours** |

---

## 🎓 What We Learn From Each Test

**From Basin Volume Test:**
- Is the Kakeya → Kuramoto theory fundamentally sound?
- What's the actual hardware success rate we should expect?
- Should we order 3 devices (high success rate) or 10 (low success rate)?

**From Critical Coupling Test:**
- Are our power consumption calculations correct?
- How much coupling strength do we actually need?
- Can we reduce message rate to save battery?

**From Sync Time Test:**
- How long should we run the hardware experiment?
- Should we plan for 2 weeks, 1 month, or 3 months?
- Do we need to optimize for faster convergence?

**From Temperature Entrainment Test:**
- Will outdoor deployment work as intended?
- Are we measuring Kuramoto synchronization or just temperature tracking?
- Do we need to detrend the temperature signal?

**From Q10 Robustness Test:**
- Do we need to calibrate each device individually?
- Can we tolerate manufacturing variance in hardware?
- Should we implement frequency adaptation?

**From Packet Loss Test:**
- Will ESP-NOW reliability be sufficient?
- Do we need redundant transmissions?
- Should we increase coupling to compensate for stale data?

---

**Next Action:** Begin Phase 0, Day 1 - Write core Kuramoto simulator

Would you like me to write the actual Python simulation code to get started?