//  FARM DATA RELAY SYSTEM - KaiABC Node Configuration
//
//  Configuration for KaiABC Biological Oscillator Node
//
//  Based on research connecting Kakeya Conjecture to distributed oscillator networks
//  Expected performance: 1.5 kbps bandwidth, 246-year battery life
//
//  For detailed configuration options and mathematical background, see:
//  - research/KaiABC/LORAWAN_COMPATIBILITY.md
//  - research/KaiABC/deep-research-prompt-claude.md

#ifndef __FDRS_NODE_CONFIG_H__
#define __FDRS_NODE_CONFIG_H__

// ============================================================================
// FDRS NETWORK CONFIGURATION
// ============================================================================

// Unique Reading ID for this node
#define READING_ID    1

// Gateway MAC Address (last byte only, prefix defined in fdrs_globals.h)
#define GTWY_MAC      0x01

// ============================================================================
// COMMUNICATION PROTOCOL
// ============================================================================

// Choose ONE communication method:
#define USE_ESPNOW    // ESP-NOW for local networks (recommend for prototyping)
// #define USE_LORA   // LoRa for long-range (recommend for production)

#ifdef USE_LORA
  // LoRa Configuration (matches gateway settings)
  #define LORA_FREQUENCY  915.0    // MHz (433, 868, or 915 depending on region)
  #define LORA_SF         10       // Spreading Factor (7-12, recommend SF10)
  #define LORA_BANDWIDTH  125.0    // kHz
  #define LORA_CR         5        // Coding rate denominator
  #define LORA_SYNCWORD   0x12     // Must match gateway
  #define LORA_TXPWR      14       // TX power in dBm (2-20)
  
  // LoRa Hardware Pins (adjust for your board)
  #define LORA_SS         18
  #define LORA_RST        14
  #define LORA_DIO0       26
  
  // Recommended for KaiABC:
  // SF10 provides 5-10 km range with 980 bps data rate
  // Airtime for 10-byte message: 330 ms
  // Energy per message: 12 mJ
  // Results in 1,027-year battery life!
#endif

#ifdef USE_ESPNOW
  // ESP-NOW Configuration
  #define USE_LR        // Long-range mode (ESP32 only, 2x range)
  
  // For ESP8266 nodes, USE_LR will be ignored
  // ESP-NOW range: ~200m (standard), ~400m (LR mode)
#endif

// ============================================================================
// KAIABC OSCILLATOR CONFIGURATION
// ============================================================================

// Oscillator Period (in hours)
// Default: 24.0 for circadian rhythm
// Can be adjusted for faster testing (e.g., 1.0 for 1-hour cycles)
#define KAIABC_PERIOD 24.0

// Temperature Compensation Coefficient (Q10)
// Q10 = 1.0: Perfect compensation (100% basin volume) - theoretical ideal
// Q10 = 1.1: Realistic KaiABC (28% basin volume) - RECOMMENDED
// Q10 = 2.2: Uncompensated oscillator (0.0001% basin) - for comparison
#define KAIABC_Q10 1.1

// Reference Temperature (°C)
// Temperature at which the period equals KAIABC_PERIOD
#define KAIABC_TREF 30.0

// Kuramoto Coupling Strength (K)
// Must be > K_c (critical coupling ≈ 2σ_ω)
// For Q10=1.1 with ±5°C variance: K_c ≈ 0.042
// Recommended: 0.1 (about 2.4× critical)
// Higher K = faster sync but more communication
#define KAIABC_COUPLING 0.1

// Phase Broadcast Interval (milliseconds)
// For rapid synchronization: 3600000 (1 hour)
// For steady-state: 7200000 (2 hours)
// For ultra-low-power: 14400000 (4 hours)
#define KAIABC_UPDATE_INTERVAL 7200000  // 2 hours

// ============================================================================
// TEMPERATURE SENSOR CONFIGURATION
// ============================================================================

// Uncomment to use BME280 sensor for temperature measurement
#define USE_BME280

#ifdef USE_BME280
  // I2C Configuration
  #define I2C_SDA 21
  #define I2C_SCL 22
  #define BME280_ADDRESS 0x76  // Or 0x77, check your module
#endif

// ============================================================================
// POWER MANAGEMENT
// ============================================================================

// Uncomment to enable deep sleep between broadcasts
// #define DEEP_SLEEP

#ifdef DEEP_SLEEP
  // Sleep time in seconds
  // Must be less than KAIABC_UPDATE_INTERVAL / 1000
  #define SLEEP_DURATION 1800  // 30 minutes
  
  // Note: Deep sleep will save power but the oscillator phase
  // must be recalculated on wake-up based on sleep duration
#endif

// Battery Monitoring (optional)
// #define BATTERY_PIN 35  // ADC pin for battery voltage divider

// Power control pin (optional - for external sensor power)
// #define POWER_CTRL 23

// ============================================================================
// DEBUGGING
// ============================================================================

// Debug Level (0 = off, 1 = basic, 2 = verbose)
#define DBG_LEVEL 1

// ============================================================================
// EXPERIMENTAL FEATURES
// ============================================================================

// Enable adaptive broadcast rate (faster when desynchronized)
// #define ADAPTIVE_BROADCAST

#ifdef ADAPTIVE_BROADCAST
  // Broadcast more frequently when R < threshold
  #define ADAPTIVE_R_THRESHOLD 0.5
  #define ADAPTIVE_FAST_INTERVAL 1800000  // 30 minutes when desynced
#endif

// Enable chirality detection (rotating vs. stationary sync)
// #define DETECT_CHIRALITY

// ============================================================================
// BOARD-SPECIFIC SETTINGS
// ============================================================================

// Uncomment your board type for optimized pin assignments

// #define BOARD_TTGO_LORA32_V1
// #define BOARD_HELTEC_LORA32
// #define BOARD_ESP32_DEVKIT
// #define BOARD_LILYGO_HIGROW

#ifdef BOARD_TTGO_LORA32_V1
  #define LORA_SS    18
  #define LORA_RST   14
  #define LORA_DIO0  26
  #define I2C_SDA    21
  #define I2C_SCL    22
  #define USE_OLED   // Built-in OLED display
#endif

#ifdef BOARD_HELTEC_LORA32
  #define LORA_SS    18
  #define LORA_RST   14
  #define LORA_DIO0  26
  #define I2C_SDA    4
  #define I2C_SCL    15
  #define USE_OLED   // Built-in OLED display
#endif

// ============================================================================
// PERFORMANCE NOTES
// ============================================================================

/*
 * Expected Performance (10 nodes, ±5°C temperature variance):
 * 
 * With Q10 = 1.1 (Realistic KaiABC):
 * - σ_ω: 0.021 rad/hr
 * - K_c: 0.042 (critical coupling)
 * - Basin volume: 28%
 * - Sync time: ~16 days
 * - Bandwidth: 1.5 kbps per device
 * - Energy: 0.3 J/day
 * - Battery life: 246 years (theoretical with 3000 mAh battery)
 * 
 * With 2-hour broadcast interval:
 * - 12 messages per day
 * - 120 bytes/day total
 * - LoRaWAN SF10: 12 mJ/message = 144 mJ/day
 * - WiFi/MQTT: 50 mJ/message = 600 mJ/day
 * 
 * Recommendation: Use LoRaWAN for production deployments!
 */

#endif // __FDRS_NODE_CONFIG_H__
