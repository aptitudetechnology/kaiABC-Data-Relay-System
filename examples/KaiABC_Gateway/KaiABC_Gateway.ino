//  FARM DATA RELAY SYSTEM - KaiABC GATEWAY
//
//  Gateway for KaiABC Biological Oscillator Network
//
//  This gateway:
//  - Receives phase synchronization messages from KaiABC nodes
//  - Relays phase data between different communication protocols
//  - Publishes synchronization metrics to MQTT
//  - Provides centralized monitoring of network synchronization state
//
//  Supported topologies:
//  - Star: All nodes communicate through this gateway
//  - Mesh: Gateway facilitates but nodes can also communicate peer-to-peer
//  - Hybrid: Both centralized and distributed synchronization
//
//  Hardware Requirements:
//  - ESP32 (required for WiFi + ESP-NOW/LoRa simultaneous operation)
//  - LoRa module (optional, for long-range nodes)
//  - Ethernet (optional, for network server connection)
//
//  Performance monitoring:
//  - Tracks order parameter R for entire network
//  - Measures synchronization time
//  - Reports bandwidth usage
//
//  Reference: research/KaiABC/NETWORK_TOPOLOGY_IMPROVEMENTS.md

#include "fdrs_gateway_config.h"
#include <fdrs_gateway.h>
#include <fdrs_kaiABC.h>

// Network Statistics
struct NetworkStats {
  uint32_t total_nodes_seen = 0;
  uint32_t active_nodes = 0;
  uint32_t total_messages = 0;
  uint32_t messages_per_hour = 0;
  float network_order_parameter = 0.0;
  float avg_period = 0.0;
  float period_std_dev = 0.0;
  unsigned long sync_start_time = 0;
  bool is_synchronized = false;
} netStats;

// Node tracking for network-wide order parameter
struct NodeInfo {
  uint16_t node_id;
  float phase;
  float period;
  float temperature;
  uint32_t last_seen;
  bool active;
} nodeList[64];  // Support up to 64 nodes

uint8_t node_count = 0;
unsigned long last_stats_print = 0;
unsigned long last_mqtt_publish = 0;

void setup() {
  beginFDRS();
  
  Serial.println("\n========================================");
  Serial.println("KaiABC Gateway");
  Serial.println("Biological Oscillator Network Hub");
  Serial.println("========================================\n");
  
  Serial.println("Gateway Configuration:");
  Serial.print("  Gateway MAC: 0x");
  Serial.println(UNIT_MAC, HEX);
  
#ifdef USE_ESPNOW
  Serial.println("  ESP-NOW: Enabled");
#endif
  
#ifdef USE_LORA
  Serial.println("  LoRa: Enabled");
  Serial.print("    Frequency: ");
  Serial.print(LORA_FREQUENCY);
  Serial.println(" MHz");
  Serial.print("    SF: ");
  Serial.println(LORA_SF);
#endif
  
#ifdef USE_WIFI
  Serial.println("  WiFi: Enabled");
  Serial.print("    SSID: ");
  Serial.println(WIFI_SSID);
#endif
  
#ifdef USE_MQTT
  Serial.println("  MQTT: Enabled");
  Serial.print("    Broker: ");
  Serial.println(MQTT_ADDR);
  Serial.print("    Topic: ");
  Serial.println(MQTT_TOPIC);
#endif
  
  Serial.println("\n========================================");
  Serial.println("Gateway active - relaying KaiABC data");
  Serial.println("========================================\n");
  
  netStats.sync_start_time = millis();
}

void loop() {
  // Main FDRS gateway loop - handles data routing
  loopFDRS();
  
  // Process received KaiABC phase data
  processKaiABCData();
  
  // Calculate and publish network statistics
  if (millis() - last_stats_print > 60000) {  // Every minute
    calculateNetworkStats();
    printNetworkStats();
    last_stats_print = millis();
  }
  
  // Publish to MQTT less frequently
  if (millis() - last_mqtt_publish > 300000) {  // Every 5 minutes
    publishKaiABCMetrics();
    last_mqtt_publish = millis();
  }
}

// Process received KaiABC phase data
void processKaiABCData() {
  // Check if we have new data
  if (data_count > 0) {
    for (uint8_t i = 0; i < data_count; i++) {
      DataReading dr = fdrsData[i];
      
      // Check if this is KaiABC phase data
      if (dr.t == KAIABC_PHASE_T) {
        // Extract node information
        uint32_t data0 = *(uint32_t*)&dr.d;
        uint16_t node_id = (data0 >> 16) & 0xFFFF;
        uint16_t phase_encoded = data0 & 0xFFFF;
        float phase = (float)phase_encoded * (2.0 * PI) / 65536.0;
        
        // Look ahead for the other parts of the message
        if (i + 2 < data_count && 
            fdrsData[i+1].t == KAIABC_PHASE_T + 1 &&
            fdrsData[i+2].t == KAIABC_PHASE_T + 2) {
          
          // Decode period and temperature
          uint32_t data1 = *(uint32_t*)&fdrsData[i+1].d;
          uint16_t period_encoded = (data1 >> 16) & 0xFFFF;
          uint8_t temperature = (data1 >> 8) & 0xFF;
          
          float period = (float)period_encoded / 10.0;
          float temp = (float)temperature - 50.0;
          
          // Update node information
          updateNodeInfo(node_id, phase, period, temp);
          
          netStats.total_messages++;
          
          // Skip the processed entries
          i += 2;
        }
      }
    }
    
    // Clear the buffer after processing
    data_count = 0;
  }
}

// Update node information
void updateNodeInfo(uint16_t node_id, float phase, float period, float temperature) {
  // Find existing node or add new one
  int8_t idx = -1;
  for (uint8_t i = 0; i < node_count; i++) {
    if (nodeList[i].node_id == node_id) {
      idx = i;
      break;
    }
  }
  
  // Add new node if not found
  if (idx == -1) {
    if (node_count < 64) {
      idx = node_count;
      nodeList[idx].node_id = node_id;
      node_count++;
      netStats.total_nodes_seen++;
      Serial.print("New node discovered: ");
      Serial.println(node_id);
    } else {
      Serial.println("Warning: Maximum nodes reached!");
      return;
    }
  }
  
  // Update node information
  nodeList[idx].phase = phase;
  nodeList[idx].period = period;
  nodeList[idx].temperature = temperature;
  nodeList[idx].last_seen = millis();
  nodeList[idx].active = true;
}

// Calculate network-wide statistics
void calculateNetworkStats() {
  // Count active nodes (seen in last 1 hour)
  uint32_t active = 0;
  float sum_cos = 0.0;
  float sum_sin = 0.0;
  float sum_period = 0.0;
  float sum_period_sq = 0.0;
  
  uint32_t current_time = millis();
  uint32_t timeout = 3600000;  // 1 hour timeout
  
  for (uint8_t i = 0; i < node_count; i++) {
    if (current_time - nodeList[i].last_seen < timeout) {
      nodeList[i].active = true;
      active++;
      
      // Accumulate for order parameter
      sum_cos += cos(nodeList[i].phase);
      sum_sin += sin(nodeList[i].phase);
      
      // Accumulate for period statistics
      sum_period += nodeList[i].period;
      sum_period_sq += nodeList[i].period * nodeList[i].period;
    } else {
      nodeList[i].active = false;
    }
  }
  
  netStats.active_nodes = active;
  
  // Calculate network order parameter
  if (active > 0) {
    float R = sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / active;
    netStats.network_order_parameter = (R > 1.0) ? 1.0 : R;
    
    // Calculate average period and standard deviation
    netStats.avg_period = sum_period / active;
    float variance = (sum_period_sq / active) - (netStats.avg_period * netStats.avg_period);
    netStats.period_std_dev = sqrt(variance);
    
    // Check if synchronized (R > 0.95)
    if (netStats.network_order_parameter > 0.95 && !netStats.is_synchronized) {
      netStats.is_synchronized = true;
      unsigned long sync_time = (millis() - netStats.sync_start_time) / 1000;
      Serial.println("\n🎉 NETWORK SYNCHRONIZED! 🎉");
      Serial.print("Time to synchronization: ");
      Serial.print(sync_time);
      Serial.println(" seconds");
      Serial.print("  (");
      Serial.print(sync_time / 3600.0);
      Serial.println(" hours)");
    }
  } else {
    netStats.network_order_parameter = 0.0;
    netStats.avg_period = 0.0;
    netStats.period_std_dev = 0.0;
  }
  
  // Calculate messages per hour
  unsigned long uptime_hours = (millis() - netStats.sync_start_time) / 3600000;
  if (uptime_hours > 0) {
    netStats.messages_per_hour = netStats.total_messages / uptime_hours;
  }
}

// Print network statistics
void printNetworkStats() {
  Serial.println("\n--- Network Statistics ---");
  Serial.print("Total nodes seen: ");
  Serial.println(netStats.total_nodes_seen);
  Serial.print("Active nodes: ");
  Serial.println(netStats.active_nodes);
  Serial.print("Total messages: ");
  Serial.println(netStats.total_messages);
  Serial.print("Messages/hour: ");
  Serial.println(netStats.messages_per_hour);
  
  Serial.print("\nOrder Parameter (R): ");
  Serial.print(netStats.network_order_parameter, 4);
  
  if (netStats.network_order_parameter > 0.95) {
    Serial.println("  ✓ SYNCHRONIZED");
  } else if (netStats.network_order_parameter > 0.5) {
    Serial.println("  ○ Partially synchronized");
  } else {
    Serial.println("  ✗ Desynchronized");
  }
  
  Serial.print("Average period: ");
  Serial.print(netStats.avg_period, 2);
  Serial.println(" hours");
  
  Serial.print("Period std dev: ");
  Serial.print(netStats.period_std_dev, 4);
  Serial.println(" hours");
  
  float heterogeneity = (netStats.avg_period > 0) ? 
                        (netStats.period_std_dev / netStats.avg_period * 100.0) : 0.0;
  Serial.print("Heterogeneity: ");
  Serial.print(heterogeneity, 2);
  Serial.println(" %");
  
  Serial.print("\nUptime: ");
  Serial.print((millis() - netStats.sync_start_time) / 1000);
  Serial.println(" seconds");
  
  // Print individual node status
  if (netStats.active_nodes > 0) {
    Serial.println("\nActive Nodes:");
    for (uint8_t i = 0; i < node_count; i++) {
      if (nodeList[i].active) {
        Serial.print("  Node ");
        Serial.print(nodeList[i].node_id);
        Serial.print(": φ=");
        Serial.print(nodeList[i].phase, 3);
        Serial.print(" rad, τ=");
        Serial.print(nodeList[i].period, 2);
        Serial.print(" hr, T=");
        Serial.print(nodeList[i].temperature, 1);
        Serial.println(" °C");
      }
    }
  }
  
  Serial.println("-------------------------\n");
}

// Publish KaiABC metrics to MQTT
void publishKaiABCMetrics() {
#ifdef USE_MQTT
  // Publish network order parameter
  loadFDRS(netStats.network_order_parameter, STATUS_T);
  sendFDRS();
  
  // Publish active node count
  loadFDRS((float)netStats.active_nodes, IT_T);
  sendFDRS();
  
  // Publish average period
  loadFDRS(netStats.avg_period, TEMP_T);  // Reuse TEMP_T for period
  sendFDRS();
  
  Serial.println("📡 Published KaiABC metrics to MQTT");
#endif
}
