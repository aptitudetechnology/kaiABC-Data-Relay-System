//  FARM DATA RELAY SYSTEM - KaiABC BIOLOGICAL OSCILLATOR NODE
//
//  KaiABC Distributed Synchronization Example
//
//  This example demonstrates biological oscillator synchronization using the
//  KaiABC circadian clock model combined with Kuramoto phase coupling.
//
//  Key Features:
//  - Ultra-low bandwidth: ~1.5 kbps (6 messages/day)
//  - Ultra-low power: ~0.3 J/day (246-year battery life theoretical)
//  - No GPS/NTP dependency: Distributed phase synchronization
//  - Temperature compensation: Q10 ≈ 1.1 for realistic operation
//
//  Hardware Requirements:
//  - ESP32 or ESP8266
//  - BME280 temperature sensor (or any I2C temp sensor)
//  - Optional: Battery voltage monitor
//
//  Expected Performance (10 nodes, ±5°C variance):
//  - Synchronization time: ~16 days
//  - Basin volume: 28% (Q10=1.1)
//  - Critical coupling: K_c ≈ 0.042
//  - Order parameter: R > 0.95 at steady state
//
//  Reference: research/KaiABC/ for mathematical foundation

#include "fdrs_node_config.h"
#include <fdrs_node.h>
#include <fdrs_kaiABC.h>

// Optional: BME280 sensor for temperature measurement
#ifdef USE_BME280
  #include <Adafruit_BME280.h>
  Adafruit_BME280 bme;
  bool bme_available = false;
#endif

// KaiABC Configuration
#define KAIABC_NODE_ID 1          // Unique node ID (1-65535)
#define INITIAL_PHASE 0.0         // Initial phase in radians (0 to 2π)
#define USE_TEMPERATURE_SENSOR    // Comment out to use fixed temperature

// Performance monitoring
unsigned long last_print = 0;
uint32_t total_messages_sent = 0;
uint32_t total_messages_received = 0;

void setup() {
  // Initialize FDRS communication
  beginFDRS();
  
  Serial.println("\n========================================");
  Serial.println("KaiABC Biological Oscillator Node");
  Serial.println("Distributed Time Synchronization");
  Serial.println("========================================\n");
  
  // Initialize temperature sensor
  float initial_temp = KAIABC_TREF;  // Default to reference temperature
  
#ifdef USE_BME280
  if (bme.begin(0x76)) {
    bme_available = true;
    initial_temp = bme.readTemperature();
    Serial.println("✓ BME280 sensor detected");
    Serial.print("  Initial temperature: ");
    Serial.print(initial_temp);
    Serial.println(" °C");
  } else {
    Serial.println("✗ BME280 not found, using reference temperature");
  }
#else
  Serial.println("Temperature sensor disabled, using reference temp");
#endif
  
  // Initialize KaiABC oscillator
  initKaiABC(KAIABC_NODE_ID, INITIAL_PHASE, initial_temp);
  
  // Print configuration
  Serial.println("\nConfiguration:");
  Serial.print("  Node ID: ");
  Serial.println(KAIABC_NODE_ID);
  Serial.print("  Q10 coefficient: ");
  Serial.println(KAIABC_Q10);
  Serial.print("  Coupling strength K: ");
  Serial.println(KAIABC_COUPLING);
  Serial.print("  Update interval: ");
  Serial.print(KAIABC_UPDATE_INTERVAL / 1000);
  Serial.println(" seconds");
  Serial.print("  Base period: ");
  Serial.print(KAIABC_PERIOD);
  Serial.println(" hours");
  
  // Calculate and display theoretical predictions
  float sigma_omega = calculateSigmaOmega(5.0);  // Assume ±5°C variance
  float K_c = calculateCriticalCoupling(sigma_omega);
  
  Serial.println("\nTheoretical Predictions (±5°C variance):");
  Serial.print("  σ_ω: ");
  Serial.print(sigma_omega, 6);
  Serial.println(" rad/hr");
  Serial.print("  Critical coupling K_c: ");
  Serial.println(K_c, 4);
  Serial.print("  K/K_c ratio: ");
  Serial.println(KAIABC_COUPLING / K_c, 2);
  
  if (KAIABC_COUPLING > K_c) {
    Serial.println("  ✓ Coupling is ABOVE critical threshold - sync expected");
  } else {
    Serial.println("  ✗ WARNING: Coupling BELOW critical - sync unlikely!");
  }
  
  Serial.println("\n========================================");
  Serial.println("Oscillator running... Broadcasting phase every " + 
                 String(KAIABC_UPDATE_INTERVAL/1000) + " seconds");
  Serial.println("========================================\n");
}

void loop() {
  // Read current temperature
  float temperature = KAIABC_TREF;  // Default
  
#ifdef USE_BME280
  if (bme_available) {
    temperature = bme.readTemperature();
  }
#endif
  
  // Update oscillator phase (this includes Kuramoto coupling)
  updateKaiABC(temperature);
  
  // Check if it's time to broadcast our phase
  if (shouldBroadcastKaiABC()) {
    // Read battery level (optional - implement based on your hardware)
    uint8_t battery = getBatteryLevel();
    
    // Load KaiABC phase data for transmission
    loadKaiABCPhase(battery);
    
    // Send via FDRS
    if (sendFDRS()) {
      total_messages_sent++;
      Serial.println("✓ Phase broadcast successful");
    } else {
      Serial.println("✗ Phase broadcast failed");
    }
  }
  
  // Print status every 60 seconds
  if (millis() - last_print > 60000) {
    printStatus();
    last_print = millis();
  }
  
  // Small delay to prevent CPU hogging
  delay(100);
}

// Print current oscillator status
void printStatus() {
  Serial.println("\n--- KaiABC Status ---");
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");
  
  Serial.print("Phase (φ): ");
  Serial.print(getKaiABCPhase(), 4);
  Serial.print(" rad  (");
  Serial.print(getKaiABCPhase() * 360.0 / (2 * PI), 1);
  Serial.println("°)");
  
  Serial.print("Period (τ): ");
  Serial.print(getKaiABCPeriod(), 2);
  Serial.println(" hours");
  
  Serial.print("Cycle count: ");
  Serial.println(getKaiABCCycleCount());
  
  Serial.print("Order parameter (R): ");
  Serial.print(getKaiABCOrderParameter(), 3);
  
  float R = getKaiABCOrderParameter();
  if (R > 0.95) {
    Serial.println("  ✓ SYNCHRONIZED");
  } else if (R > 0.5) {
    Serial.println("  ○ Partially synchronized");
  } else {
    Serial.println("  ✗ Desynchronized");
  }
  
  Serial.print("Messages sent: ");
  Serial.println(total_messages_sent);
  Serial.print("Messages received: ");
  Serial.println(total_messages_received);
  
  Serial.print("Active neighbors: ");
  Serial.println(kaiABC_neighbor_count);
  
  Serial.println("--------------------\n");
}

// Get battery level (implement based on your hardware)
uint8_t getBatteryLevel() {
  // TODO: Implement actual battery voltage reading
  // For now, return 100%
  
  #ifdef BATTERY_PIN
    // Example for voltage divider on analog pin
    float voltage = analogRead(BATTERY_PIN) * (3.3 / 4095.0) * 2.0;  // Adjust for your circuit
    // Convert to percentage (3.0V = 0%, 4.2V = 100% for LiPo)
    uint8_t percent = (uint8_t)((voltage - 3.0) / 1.2 * 100.0);
    return constrain(percent, 0, 100);
  #else
    return 100;  // Default if no battery monitoring
  #endif
}

// Callback function for received data (optional)
void receivedCallback(DataReading theData) {
  // Check if this is KaiABC phase data
  if (theData.t == KAIABC_PHASE_T) {
    total_messages_received++;
    // Phase data will be automatically processed by processKaiABCReading
    // This callback is for additional custom handling if needed
    Serial.print("← Received phase from node ");
    Serial.println(theData.id);
  }
}
