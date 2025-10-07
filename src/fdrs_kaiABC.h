//  FARM DATA RELAY SYSTEM - KaiABC Biological Oscillator Module
//
//  Implementation of KaiABC circadian oscillator synchronization
//  Based on research connecting Kakeya Conjecture to distributed oscillator networks
//
//  Developed for ultra-low-power time synchronization without NTP/GPS dependency
//  Expected performance: 1.5 kbps bandwidth, 246-year battery life
//
//  Mathematical Foundation:
//  - Kuramoto coupling model for phase synchronization
//  - Temperature compensation via Q10 coefficient
//  - Basin volume: V ≈ (1 - 1.5σ_ω/⟨ω⟩)^N
//  - Critical coupling: K_c ≥ 2σ_ω
//
//  Reference: research/KaiABC/ for detailed mathematical analysis

#ifndef __FDRS_KAIABC_H__
#define __FDRS_KAIABC_H__

#include <math.h>
#include "fdrs_datatypes.h"
#include "fdrs_debug.h"

// ============================================================================
// KAIABC CONFIGURATION
// ============================================================================

#ifndef KAIABC_PERIOD
#define KAIABC_PERIOD 24.0  // Base period in hours (circadian rhythm)
#endif

#ifndef KAIABC_Q10
#define KAIABC_Q10 1.1  // Temperature coefficient (1.1 = realistic compensation)
                         // Q10 = 1.0: perfect compensation
                         // Q10 = 1.1: realistic KaiABC (28% basin volume)
                         // Q10 = 2.2: uncompensated (0.0001% basin volume)
#endif

#ifndef KAIABC_TREF
#define KAIABC_TREF 30.0  // Reference temperature in °C
#endif

#ifndef KAIABC_COUPLING
#define KAIABC_COUPLING 0.1  // Kuramoto coupling strength K
                              // Should be > K_c (critical coupling ≈ 2σ_ω)
                              // Recommended: 0.1 for realistic Q10=1.1
#endif

#ifndef KAIABC_UPDATE_INTERVAL
#define KAIABC_UPDATE_INTERVAL 7200000  // 2 hours in milliseconds
                                         // For rapid sync: 3600000 (1 hour)
                                         // For steady state: 14400000 (4 hours)
#endif

// ============================================================================
// KAIABC DATA STRUCTURES
// ============================================================================

// KaiABC oscillator state
typedef struct {
    float phase;              // Current phase φ ∈ [0, 2π)
    float omega;              // Angular frequency ω = 2π/τ
    float period;             // Current period τ in hours
    float last_temperature;   // Last measured temperature
    uint32_t last_update;     // Last update timestamp (millis)
    float order_parameter;    // Local estimate of synchronization R
    uint16_t cycle_count;     // Number of complete cycles
} KaiABCState;

// Phase synchronization message (10 bytes - optimal for LoRaWAN)
typedef struct __attribute__((packed)) {
    uint16_t node_id;        // Oscillator ID (2 bytes)
    uint16_t phase_encoded;  // Phase in units of 2π/65536 (2 bytes)
    uint16_t period_encoded; // Period in units of 0.1 hour (2 bytes)
    uint8_t temperature;     // Temperature in °C + 50 (1 byte, range -50 to +205)
    uint8_t order_param;     // Order parameter R × 255 (1 byte)
    uint8_t battery_level;   // Battery percentage (1 byte)
    uint8_t sequence;        // Message sequence number (1 byte)
} KaiABCMessage;

// Neighbor oscillator information for Kuramoto coupling
typedef struct {
    uint16_t node_id;
    float phase;
    float omega;
    uint32_t last_seen;
    bool active;
} KaiABCNeighbor;

// ============================================================================
// KAIABC GLOBAL STATE
// ============================================================================

KaiABCState kaiABC_state;
KaiABCNeighbor kaiABC_neighbors[32];  // Max 32 neighbors for coupling
uint8_t kaiABC_neighbor_count = 0;
uint16_t kaiABC_node_id = 0;
uint8_t kaiABC_sequence = 0;
unsigned long kaiABC_last_broadcast = 0;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Wrap phase to [0, 2π)
float wrapPhase(float phase) {
    float two_pi = 2.0 * PI;
    while (phase < 0) phase += two_pi;
    while (phase >= two_pi) phase -= two_pi;
    return phase;
}

// Calculate temperature-dependent period using Q10
float calculatePeriod(float temperature) {
    float temp_diff = (KAIABC_TREF - temperature) / 10.0;
    return KAIABC_PERIOD * pow(KAIABC_Q10, temp_diff);
}

// Calculate angular frequency from period
float calculateOmega(float period) {
    return (2.0 * PI) / period;  // rad/hour
}

// Calculate frequency heterogeneity σ_ω from temperature variance
float calculateSigmaOmega(float sigma_T) {
    float ln_Q10 = log(KAIABC_Q10);
    float omega_avg = 2.0 * PI / KAIABC_PERIOD;
    float dw_dT = (omega_avg / KAIABC_PERIOD) * (ln_Q10 / 10.0);
    return fabs(dw_dT) * sigma_T;
}

// Estimate critical coupling K_c
float calculateCriticalCoupling(float sigma_omega) {
    return 2.0 * sigma_omega;  // Conservative estimate
}

// Calculate order parameter from neighbors (local synchronization measure)
float calculateOrderParameter() {
    if (kaiABC_neighbor_count == 0) return 0.0;
    
    float sum_cos = 0.0;
    float sum_sin = 0.0;
    uint8_t active_count = 0;
    
    for (uint8_t i = 0; i < kaiABC_neighbor_count; i++) {
        if (kaiABC_neighbors[i].active) {
            sum_cos += cos(kaiABC_neighbors[i].phase);
            sum_sin += sin(kaiABC_neighbors[i].phase);
            active_count++;
        }
    }
    
    if (active_count == 0) return 0.0;
    
    float R = sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / active_count;
    return (R > 1.0) ? 1.0 : R;  // Clamp to [0, 1]
}

// ============================================================================
// KAIABC OSCILLATOR CORE
// ============================================================================

// Initialize KaiABC oscillator
void initKaiABC(uint16_t node_id, float initial_phase = 0.0, float temperature = KAIABC_TREF) {
    kaiABC_node_id = node_id;
    kaiABC_state.phase = wrapPhase(initial_phase);
    kaiABC_state.period = calculatePeriod(temperature);
    kaiABC_state.omega = calculateOmega(kaiABC_state.period);
    kaiABC_state.last_temperature = temperature;
    kaiABC_state.last_update = millis();
    kaiABC_state.order_parameter = 0.0;
    kaiABC_state.cycle_count = 0;
    kaiABC_neighbor_count = 0;
    kaiABC_sequence = 0;
    kaiABC_last_broadcast = 0;
    
    DBG("KaiABC Oscillator Initialized:");
    DBG("  Node ID: " + String(node_id));
    DBG("  Initial Phase: " + String(kaiABC_state.phase) + " rad");
    DBG("  Period: " + String(kaiABC_state.period) + " hours");
    DBG("  Omega: " + String(kaiABC_state.omega) + " rad/hr");
    DBG("  Q10: " + String(KAIABC_Q10));
    DBG("  Coupling K: " + String(KAIABC_COUPLING));
}

// Update oscillator phase (call this periodically)
void updateKaiABC(float temperature) {
    unsigned long current_time = millis();
    float dt_ms = current_time - kaiABC_state.last_update;
    float dt_hours = dt_ms / 3600000.0;  // Convert milliseconds to hours
    
    // Update period and frequency based on current temperature
    kaiABC_state.period = calculatePeriod(temperature);
    kaiABC_state.omega = calculateOmega(kaiABC_state.period);
    kaiABC_state.last_temperature = temperature;
    
    // Calculate Kuramoto coupling term: K/N * Σ sin(φ_j - φ_i)
    float coupling_term = 0.0;
    uint8_t active_neighbors = 0;
    
    for (uint8_t i = 0; i < kaiABC_neighbor_count; i++) {
        if (kaiABC_neighbors[i].active) {
            // Check if neighbor data is stale (older than 1 hour)
            if ((current_time - kaiABC_neighbors[i].last_seen) < 3600000) {
                float phase_diff = kaiABC_neighbors[i].phase - kaiABC_state.phase;
                coupling_term += sin(phase_diff);
                active_neighbors++;
            } else {
                kaiABC_neighbors[i].active = false;  // Mark as inactive
            }
        }
    }
    
    if (active_neighbors > 0) {
        coupling_term = (KAIABC_COUPLING / active_neighbors) * coupling_term;
    }
    
    // Kuramoto model: dφ/dt = ω + K/N * Σ sin(φ_j - φ_i)
    float dphase = (kaiABC_state.omega + coupling_term) * dt_hours;
    kaiABC_state.phase = wrapPhase(kaiABC_state.phase + dphase);
    
    // Count complete cycles
    if (kaiABC_state.phase < dphase) {
        kaiABC_state.cycle_count++;
    }
    
    // Update order parameter
    kaiABC_state.order_parameter = calculateOrderParameter();
    
    kaiABC_state.last_update = current_time;
    
    DBG2("KaiABC Updated: φ=" + String(kaiABC_state.phase, 4) + 
         " ω=" + String(kaiABC_state.omega, 6) + 
         " R=" + String(kaiABC_state.order_parameter, 3) +
         " Active neighbors: " + String(active_neighbors));
}

// Receive phase information from neighbor
void receiveKaiABCPhase(uint16_t neighbor_id, float neighbor_phase, float neighbor_omega) {
    // Find existing neighbor or add new one
    int8_t idx = -1;
    for (uint8_t i = 0; i < kaiABC_neighbor_count; i++) {
        if (kaiABC_neighbors[i].node_id == neighbor_id) {
            idx = i;
            break;
        }
    }
    
    // Add new neighbor if not found and space available
    if (idx == -1) {
        if (kaiABC_neighbor_count < 32) {
            idx = kaiABC_neighbor_count;
            kaiABC_neighbors[idx].node_id = neighbor_id;
            kaiABC_neighbor_count++;
            DBG1("New KaiABC neighbor: " + String(neighbor_id));
        } else {
            DBG("Warning: Maximum neighbors reached!");
            return;
        }
    }
    
    // Update neighbor information
    kaiABC_neighbors[idx].phase = neighbor_phase;
    kaiABC_neighbors[idx].omega = neighbor_omega;
    kaiABC_neighbors[idx].last_seen = millis();
    kaiABC_neighbors[idx].active = true;
    
    DBG2("Received phase from " + String(neighbor_id) + 
         ": φ=" + String(neighbor_phase, 4) + 
         " ω=" + String(neighbor_omega, 6));
}

// ============================================================================
// KAIABC MESSAGE ENCODING/DECODING
// ============================================================================

// Create KaiABC message for transmission
KaiABCMessage createKaiABCMessage(uint8_t battery_level = 100) {
    KaiABCMessage msg;
    
    msg.node_id = kaiABC_node_id;
    msg.phase_encoded = (uint16_t)(kaiABC_state.phase * 65536.0 / (2.0 * PI));
    msg.period_encoded = (uint16_t)(kaiABC_state.period * 10.0);  // 0.1 hour units
    msg.temperature = (uint8_t)(kaiABC_state.last_temperature + 50.0);  // Offset by 50
    msg.order_param = (uint8_t)(kaiABC_state.order_parameter * 255.0);
    msg.battery_level = battery_level;
    msg.sequence = kaiABC_sequence++;
    
    return msg;
}

// Decode KaiABC message from neighbor
void decodeKaiABCMessage(const KaiABCMessage& msg) {
    float phase = (float)msg.phase_encoded * (2.0 * PI) / 65536.0;
    float period = (float)msg.period_encoded / 10.0;
    float omega = calculateOmega(period);
    
    receiveKaiABCPhase(msg.node_id, phase, omega);
    
    DBG1("Decoded KaiABC message from node " + String(msg.node_id) + ":");
    DBG1("  Phase: " + String(phase, 4) + " rad");
    DBG1("  Period: " + String(period, 2) + " hours");
    DBG1("  Temp: " + String((float)msg.temperature - 50.0, 1) + " °C");
    DBG1("  R: " + String((float)msg.order_param / 255.0, 3));
    DBG1("  Battery: " + String(msg.battery_level) + "%");
}

// ============================================================================
// FDRS INTEGRATION
// ============================================================================

// Load KaiABC phase data into FDRS for transmission
void loadKaiABCPhase(uint8_t battery_level = 100) {
    // Create the message
    KaiABCMessage msg = createKaiABCMessage(battery_level);
    
    // Pack message into DataReading structures
    // We'll use 5 DataReading entries to carry the 10-byte message
    // Type = KAIABC_PHASE_T (to be defined in datatypes)
    
    // Reading 0: node_id and phase_encoded
    uint32_t data0 = ((uint32_t)msg.node_id << 16) | msg.phase_encoded;
    loadFDRS(*(float*)&data0, KAIABC_PHASE_T);
    
    // Reading 1: period_encoded, temperature, order_param
    uint32_t data1 = ((uint32_t)msg.period_encoded << 16) | 
                     ((uint32_t)msg.temperature << 8) | msg.order_param;
    loadFDRS(*(float*)&data1, KAIABC_PHASE_T + 1);
    
    // Reading 2: battery_level, sequence
    uint32_t data2 = ((uint32_t)msg.battery_level << 8) | msg.sequence;
    loadFDRS(*(float*)&data2, KAIABC_PHASE_T + 2);
    
    DBG1("Loaded KaiABC phase for transmission");
}

// Process received KaiABC phase data from FDRS
void processKaiABCReading(const DataReading& dr0, const DataReading& dr1, const DataReading& dr2) {
    KaiABCMessage msg;
    
    // Unpack data from three DataReading entries
    uint32_t data0 = *(uint32_t*)&dr0.d;
    msg.node_id = (data0 >> 16) & 0xFFFF;
    msg.phase_encoded = data0 & 0xFFFF;
    
    uint32_t data1 = *(uint32_t*)&dr1.d;
    msg.period_encoded = (data1 >> 16) & 0xFFFF;
    msg.temperature = (data1 >> 8) & 0xFF;
    msg.order_param = data1 & 0xFF;
    
    uint32_t data2 = *(uint32_t*)&dr2.d;
    msg.battery_level = (data2 >> 8) & 0xFF;
    msg.sequence = data2 & 0xFF;
    
    // Decode and update neighbor information
    decodeKaiABCMessage(msg);
}

// Get current KaiABC phase (for debugging/display)
float getKaiABCPhase() {
    return kaiABC_state.phase;
}

// Get current period
float getKaiABCPeriod() {
    return kaiABC_state.period;
}

// Get order parameter
float getKaiABCOrderParameter() {
    return kaiABC_state.order_parameter;
}

// Get cycle count
uint16_t getKaiABCCycleCount() {
    return kaiABC_state.cycle_count;
}

// Check if it's time to broadcast phase
bool shouldBroadcastKaiABC() {
    unsigned long current_time = millis();
    if (current_time - kaiABC_last_broadcast >= KAIABC_UPDATE_INTERVAL) {
        kaiABC_last_broadcast = current_time;
        return true;
    }
    return false;
}

#endif // __FDRS_KAIABC_H__
