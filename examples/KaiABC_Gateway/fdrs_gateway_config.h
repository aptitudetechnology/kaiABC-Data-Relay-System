//  FARM DATA RELAY SYSTEM - KaiABC Gateway Configuration
//
//  Configuration for KaiABC Biological Oscillator Gateway
//
//  This gateway relays phase synchronization data between:
//  - ESP-NOW nodes (local, short-range)
//  - LoRa nodes (long-range, up to 15 km)
//  - MQTT broker (for monitoring and data logging)
//  - Serial/UART (for debugging and local display)

#ifndef __FDRS_GATEWAY_CONFIG_H__
#define __FDRS_GATEWAY_CONFIG_H__

// ============================================================================
// GATEWAY IDENTIFICATION
// ============================================================================

// Unique Gateway MAC Address (last byte only)
#define UNIT_MAC      0x01

// Gateway description (for debugging)
#define GATEWAY_NAME  "KaiABC-Gateway-01"

// ============================================================================
// COMMUNICATION PROTOCOLS
// ============================================================================

// ESP-NOW Configuration (for local nodes)
#define USE_ESPNOW
#ifdef USE_ESPNOW
  #define ESPNOWG_ACT   // Enable ESP-NOW gateway function
  #define USE_LR        // Long-range mode (ESP32 only)
#endif

// LoRa Configuration (for long-range nodes)
#define USE_LORA
#ifdef USE_LORA
  #define LORAG_ACT     // Enable LoRa gateway function
  
  // LoRa Radio Settings (must match node settings)
  #define LORA_FREQUENCY  915.0    // MHz (433, 868, or 915)
  #define LORA_SF         10       // Spreading Factor (7-12)
  #define LORA_BANDWIDTH  125.0    // kHz
  #define LORA_CR         5        // Coding rate
  #define LORA_SYNCWORD   0x12     // Sync word
  #define LORA_TXPWR      17       // TX power in dBm
  
  // LoRa Hardware Pins
  #define LORA_SS         18
  #define LORA_RST        14
  #define LORA_DIO0       26
  #define LORA_DIO1       33  // Optional
#endif

// Serial/UART Configuration
#define USE_SERIAL
#ifdef USE_SERIAL
  #define SERIAL_ACT    // Enable serial gateway function
  #define RXD2          16
  #define TXD2          17
#endif

// WiFi Configuration
#define USE_WIFI
#ifdef USE_WIFI
  #define WIFI_SSID     "Your-SSID"
  #define WIFI_PASS     "Your-Password"
  
  // DNS Servers
  #define DNS1_IPADDRESS  "8.8.8.8"
  #define DNS2_IPADDRESS  "8.8.4.4"
#endif

// MQTT Configuration
#define USE_MQTT
#ifdef USE_MQTT
  #define MQTT_ACT      // Enable MQTT publishing
  
  #define MQTT_ADDR     "192.168.1.100"  // MQTT broker IP
  #define MQTT_PORT     1883
  
  // MQTT Authentication (optional)
  // #define MQTT_AUTH
  #ifdef MQTT_AUTH
    #define MQTT_USER   "your_username"
    #define MQTT_PASS   "your_password"
  #endif
  
  // MQTT Topics
  #define MQTT_TOPIC          "kaiabc/data"
  #define MQTT_STATUS_TOPIC   "kaiabc/status"
  #define MQTT_COMMAND_TOPIC  "kaiabc/command"
#endif

// ============================================================================
// NETWORK TOPOLOGY
// ============================================================================

// Gateway topology mode
// TOPOLOGY_STAR:   All nodes communicate through this gateway
// TOPOLOGY_MESH:   Gateway facilitates but nodes can also peer-to-peer
// TOPOLOGY_HYBRID: Both centralized and distributed (recommended)
#define TOPOLOGY_HYBRID

// Neighbor gateways (for multi-gateway networks)
// #define USE_ESPNOW_PEERS
#ifdef USE_ESPNOW_PEERS
  // Add neighbor gateway MAC addresses
  uint8_t peer_gateway_1[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x02};
  uint8_t peer_gateway_2[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x03};
#endif

// ============================================================================
// DATA ROUTING
// ============================================================================

// Automatically relay KaiABC phase data between protocols
#define AUTO_RELAY_KAIABC

// Forward all ESP-NOW data to LoRa
#define ESPNOW_TO_LORA

// Forward all LoRa data to ESP-NOW
#define LORA_TO_ESPNOW

// Forward all data to MQTT
#define ALL_TO_MQTT

// Forward MQTT commands to nodes
#define MQTT_TO_NODES

// ============================================================================
// POWER & PERFORMANCE
// ============================================================================

// Power control pin (optional)
// #define POWER_CTRL 23

// LED indicator (optional)
// #define STATUS_LED 2

// Watchdog timer (seconds)
#define WATCHDOG_TIMEOUT 300  // 5 minutes

// ============================================================================
// DEBUGGING & MONITORING
// ============================================================================

// Debug level (0 = off, 1 = basic, 2 = verbose)
#define DBG_LEVEL 1

// Enable OLED display (if available)
// #define USE_OLED
#ifdef USE_OLED
  #define I2C_SDA 21
  #define I2C_SCL 22
#endif

// Enable detailed statistics logging
#define ENABLE_STATS_LOGGING

// Statistics reporting interval (milliseconds)
#define STATS_INTERVAL 60000  // 1 minute

// MQTT publish interval (milliseconds)
#define MQTT_PUBLISH_INTERVAL 300000  // 5 minutes

// ============================================================================
// KAIABC-SPECIFIC SETTINGS
// ============================================================================

// Maximum nodes to track
#define MAX_NODES 64

// Node timeout (milliseconds)
// Nodes not seen within this time are considered inactive
#define NODE_TIMEOUT 3600000  // 1 hour

// Order parameter threshold for synchronization
#define SYNC_THRESHOLD 0.95

// Enable bandwidth monitoring
#define MONITOR_BANDWIDTH

// Enable synchronization time measurement
#define MEASURE_SYNC_TIME

// ============================================================================
// ADVANCED FEATURES
// ============================================================================

// Enable OTA (Over-The-Air) firmware updates
// #define USE_OTA
#ifdef USE_OTA
  #define OTA_HOSTNAME "kaiabc-gateway-01"
  #define OTA_PASSWORD "your-ota-password"
#endif

// Enable web server for monitoring
// #define USE_WEBSERVER
#ifdef USE_WEBSERVER
  #define WEBSERVER_PORT 80
#endif

// Enable NTP time synchronization (for timestamping)
// #define USE_NTP
#ifdef USE_NTP
  #define NTP_SERVER "pool.ntp.org"
  #define NTP_OFFSET -21600  // UTC offset in seconds (e.g., -6h for CST)
#endif

// ============================================================================
// BOARD-SPECIFIC SETTINGS
// ============================================================================

// Uncomment your board type

// #define BOARD_TTGO_LORA32_V1
// #define BOARD_HELTEC_LORA32
// #define BOARD_ESP32_DEVKIT

#ifdef BOARD_TTGO_LORA32_V1
  #define LORA_SS    18
  #define LORA_RST   14
  #define LORA_DIO0  26
  #define I2C_SDA    21
  #define I2C_SCL    22
  #define USE_OLED
#endif

#ifdef BOARD_HELTEC_LORA32
  #define LORA_SS    18
  #define LORA_RST   14
  #define LORA_DIO0  26
  #define I2C_SDA    4
  #define I2C_SCL    15
  #define USE_OLED
#endif

// ============================================================================
// PERFORMANCE EXPECTATIONS
// ============================================================================

/*
 * Expected Gateway Performance:
 * 
 * Network Capacity:
 * - ESP-NOW: 20-30 nodes per gateway (250-byte packets)
 * - LoRa: 50-100 nodes per gateway (duty cycle limited)
 * - Total: 64 nodes tracked simultaneously
 * 
 * Bandwidth Usage (10 nodes, 2-hour intervals):
 * - Receive: ~1.5 kbps per node × 10 = 15 kbps
 * - Transmit: ~1.5 kbps to MQTT
 * - Total: ~16.5 kbps sustained
 * 
 * Power Consumption:
 * - Always-on WiFi + ESP-NOW: ~500 mW
 * - With LoRa RX: ~700 mW
 * - Gateway should be mains-powered or solar
 * 
 * Latency:
 * - ESP-NOW: <10 ms
 * - LoRa: 100-1000 ms (depends on SF)
 * - MQTT: 50-200 ms (depends on network)
 * 
 * Recommended Deployment:
 * - 1 gateway per 10-20 nodes
 * - Gateways spaced 5-10 km apart (LoRa)
 * - Centralized MQTT broker for monitoring
 */

#endif // __FDRS_GATEWAY_CONFIG_H__
