# KaiABC Implementation Plan for ELM11 Lua Microcontroller

**Date:** October 8, 2025  
**Status:** Planning Phase - Ready for Implementation  
**Target Platform:** ELM11 (Lua-based microcontroller)

---

## 🎯 Executive Summary

The ELM11 microcontroller with Lua runtime is an excellent platform for prototyping KaiABC biological oscillator synchronization. While it lacks native ESP-NOW/LoRa support (unlike ESP32), the core mathematical logic can be fully implemented in Lua, with MQTT serving as the communication layer. This creates a "soft" port that focuses on synchronization algorithms while adapting FDRS communication patterns.

**Key Feasibility:**
- ✅ **Core oscillator logic** - Pure mathematics, perfect for Lua
- ✅ **MQTT communication** - Native support in ELM11
- ✅ **Sensor integration** - I2C/SPI sensors work well
- ⚠️ **ESP-NOW/LoRa** - Not native, use MQTT as alternative
- ✅ **Rapid prototyping** - Lua's interpreted nature speeds development

---

## 🔬 Technical Analysis

### What Can Be Implemented in Lua

#### 1. **KaiABC Oscillator Core (100% Feasible)**
- **Kuramoto phase coupling model** - Pure floating-point math
- **Q10 temperature compensation** - Mathematical formula
- **Order parameter calculation** - Statistical computation
- **Phase evolution** - Euler integration
- **Neighbor management** - Table-based data structures

#### 2. **Communication Layer (Feasible with Adaptation)**
- **MQTT transport** - Native ELM11 support
- **Message encoding/decoding** - JSON via sjson library
- **Topic-based routing** - MQTT pub/sub patterns
- **Network discovery** - Broadcast/listen mechanisms

#### 3. **Sensor Integration (Feasible)**
- **Temperature sensors** - I2C (BME280, DS18B20)
- **Data aggregation** - Periodic sampling
- **Calibration** - Offset/compensation math

#### 4. **Data Processing (Highly Feasible)**
- **Statistics calculation** - Mean, variance, correlation
- **Logging and debugging** - File/console output
- **Configuration management** - Lua table-based config
- **Status reporting** - JSON serialization

### Limitations and Workarounds

#### 1. **No Native ESP-NOW/LoRa**
- **Limitation:** ELM11 lacks ESP32-style radio protocols
- **Workaround:** Use MQTT over WiFi as transport layer
- **Impact:** Higher latency but reliable for prototyping
- **Compatibility:** Works with existing FDRS MQTT infrastructure

#### 2. **Performance Considerations**
- **Limitation:** Lua is interpreted (slower than C++)
- **Workaround:** Use longer update intervals (5-15 min vs 1-2 min)
- **Impact:** Minimal for biological timescales (24-hour cycles)
- **Optimization:** Pre-compile Lua bytecode if supported

#### 3. **Hardware Dependencies**
- **Limitation:** No direct LoRa module support
- **Workaround:** External LoRa via UART bridge (ESP32 + ELM11)
- **Impact:** Increased complexity for long-range testing
- **Alternative:** WiFi-only testing initially

---

## 📁 Implementation Structure

### Core Files to Create

```
examples/KaiABC_ELM11/
├── kaiabc.lua              # Core oscillator library
├── node.lua                # Main node script
├── gateway.lua             # Simple gateway/collector
├── config.lua              # Configuration file
└── README.md               # ELM11-specific documentation
```

### 1. Core KaiABC Library (`kaiabc.lua`)

```lua
-- KaiABC Oscillator Library for ELM11 (Lua-based)

local kaiabc = {}

-- Configuration defaults (mirroring fdrs_globals.h)
kaiabc.config = {
    period_hours = 24.0,           -- Base circadian period
    q10 = 1.1,                     -- Temperature compensation
    coupling_strength = 0.1,       -- Kuramoto K parameter
    update_interval_ms = 7200000,  -- 2 hours in milliseconds
    node_id = 1,                   -- Unique node identifier
    mqtt_topic = "fdrs/kaiabc",    -- MQTT topic for phase exchange
    mqtt_broker = "192.168.1.100", -- MQTT broker IP
    mqtt_port = 1883               -- MQTT broker port
}

-- Internal state variables
kaiabc.phase = 0.0                -- Current phase (0-2π radians)
kaiabc.neighbors = {}             -- Table of neighbor phases {id: phase}
kaiabc.temperature = 25.0         -- Current temperature (°C)
kaiabc.last_update = 0            -- Timestamp of last update
kaiabc.cycle_count = 0            -- Number of completed cycles

-- Calculate temperature-compensated period using Q10 model
function kaiabc.calculatePeriod(temp)
    local base_temp = 25.0  -- Reference temperature
    local q10_factor = kaiabc.config.q10 ^ ((temp - base_temp) / 10.0)
    return kaiabc.config.period_hours * q10_factor
end

-- Update phase using Kuramoto model with Euler integration
function kaiabc.updatePhase(dt_hours)
    -- Calculate natural frequency from temperature-compensated period
    local omega = (2 * math.pi) / kaiabc.calculatePeriod(kaiabc.temperature)
    
    -- Calculate coupling term from neighbors
    local coupling_sum = 0.0
    local neighbor_count = 0
    
    for id, neighbor_phase in pairs(kaiabc.neighbors) do
        coupling_sum = coupling_sum + math.sin(neighbor_phase - kaiabc.phase)
        neighbor_count = neighbor_count + 1
    end
    
    -- Apply Kuramoto equation: dφ/dt = ω + (K/N) * Σsin(φⱼ - φᵢ)
    if neighbor_count > 0 then
        local dphi_dt = omega + (kaiabc.config.coupling_strength / neighbor_count) * coupling_sum
        kaiabc.phase = kaiabc.phase + dphi_dt * dt_hours
        
        -- Keep phase in [0, 2π) range
        kaiabc.phase = kaiabc.phase % (2 * math.pi)
        
        -- Count completed cycles
        if kaiabc.phase < math.pi and (kaiabc.phase + dphi_dt * dt_hours) >= math.pi then
            kaiabc.cycle_count = kaiabc.cycle_count + 1
        end
    end
end

-- Calculate order parameter R (synchronization metric)
function kaiabc.calculateOrderParameter()
    local sum_cos = 0.0
    local sum_sin = 0.0
    local total_nodes = 0
    
    -- Sum over all neighbors
    for id, phase in pairs(kaiabc.neighbors) do
        sum_cos = sum_cos + math.cos(phase)
        sum_sin = sum_sin + math.sin(phase)
        total_nodes = total_nodes + 1
    end
    
    -- Include self
    sum_cos = sum_cos + math.cos(kaiabc.phase)
    sum_sin = sum_sin + math.sin(kaiabc.phase)
    total_nodes = total_nodes + 1
    
    -- Calculate resultant vector magnitude
    if total_nodes > 0 then
        return math.sqrt(sum_cos^2 + sum_sin^2) / total_nodes
    end
    return 0.0
end

-- Process incoming phase message from neighbor
function kaiabc.processMessage(id, phase, temp)
    kaiabc.neighbors[id] = phase
    -- Could also store temperature for network analysis
    kaiabc.neighbors[id .. "_temp"] = temp
end

-- Clean up stale neighbor data (simple timeout mechanism)
function kaiabc.cleanupNeighbors(max_age_ms)
    local current_time = tmr.now()
    for id, phase in pairs(kaiabc.neighbors) do
        if type(id) == "number" and kaiabc.neighbors[id .. "_time"] then
            if current_time - kaiabc.neighbors[id .. "_time"] > max_age_ms then
                kaiabc.neighbors[id] = nil
                kaiabc.neighbors[id .. "_time"] = nil
                kaiabc.neighbors[id .. "_temp"] = nil
            end
        end
    end
end

-- Get comprehensive status information
function kaiabc.getStatus()
    return {
        phase = kaiabc.phase,
        period = kaiabc.calculatePeriod(kaiabc.temperature),
        order_param = kaiabc.calculateOrderParameter(),
        neighbor_count = (function()
            local count = 0
            for id, _ in pairs(kaiabc.neighbors) do
                if type(id) == "number" then count = count + 1 end
            end
            return count
        end)(),
        temperature = kaiabc.temperature,
        cycle_count = kaiabc.cycle_count,
        config = kaiabc.config
    }
end

-- Initialize oscillator with random phase
function kaiabc.init()
    math.randomseed(tmr.now())
    kaiabc.phase = math.random() * 2 * math.pi
    kaiabc.last_update = tmr.now()
    print("KaiABC initialized with random phase: " .. kaiabc.phase)
end

return kaiabc
```

### 2. Node Implementation (`node.lua`)

```lua
-- KaiABC Node for ELM11 - MQTT-based synchronization

local kaiabc = require("kaiabc")
local mqtt = require("mqtt")
local sjson = require("sjson")

-- Initialize KaiABC oscillator
kaiabc.init()

-- Configure MQTT client
local client = mqtt.Client("kaiabc_node_" .. kaiabc.config.node_id, 120)
client:connect(kaiabc.config.mqtt_broker, kaiabc.config.mqtt_port)

-- MQTT message handler
client:on("message", function(client, topic, data)
    if topic == kaiabc.config.mqtt_topic then
        local success, msg = pcall(sjson.decode, data)
        if success and msg.id and msg.phase then
            kaiabc.processMessage(msg.id, msg.phase, msg.temp or 25.0)
            -- Update timestamp for cleanup
            kaiabc.neighbors[msg.id .. "_time"] = tmr.now()
        end
    end
end)

-- Subscribe to synchronization topic
client:subscribe(kaiabc.config.mqtt_topic, 0)

-- Temperature sensor setup (example: BME280 via I2C)
local bme280 = require("bme280")
bme280.init(5, 4)  -- SDA=GPIO5, SCL=GPIO4
bme280.read()  -- Initial read

-- Main update timer (every 5 minutes for Lua performance)
local update_timer = tmr.create()
update_timer:register(300000, tmr.ALARM_AUTO, function()
    -- Read temperature sensor
    local temp = bme280.temp()
    if temp then
        kaiabc.temperature = temp / 100  -- BME280 returns temp*100
    end
    
    -- Calculate time delta in hours
    local current_time = tmr.now()
    local dt_ms = current_time - kaiabc.last_update
    local dt_hours = dt_ms / 3600000.0  -- Convert ms to hours
    kaiabc.last_update = current_time
    
    -- Update oscillator phase
    kaiabc.updatePhase(dt_hours)
    
    -- Clean up old neighbor data (1 hour timeout)
    kaiabc.cleanupNeighbors(3600000)
    
    -- Publish current phase and status
    local status = kaiabc.getStatus()
    local msg = sjson.encode({
        id = kaiabc.config.node_id,
        phase = status.phase,
        temp = status.temperature,
        order_param = status.order_param,
        timestamp = current_time
    })
    client:publish(kaiabc.config.mqtt_topic, msg, 0, 0)
    
    -- Debug output
    print(string.format("[Node %d] Phase: %.2f, Order: %.2f, Neighbors: %d, Temp: %.1f°C",
          kaiabc.config.node_id, status.phase, status.order_param, 
          status.neighbor_count, status.temperature))
end)

-- Start the oscillator
update_timer:start()
print("KaiABC Node started - Node ID: " .. kaiabc.config.node_id)

-- Keep the script running
-- Note: In ELM11, this would be handled by the runtime
```

### 3. Gateway/Collector (`gateway.lua`)

```lua
-- KaiABC Gateway for ELM11 - Network monitoring and data collection

local mqtt = require("mqtt")
local sjson = require("sjson")

-- Network state tracking
local nodes = {}          -- Current node states
local network_history = {} -- Historical data for analysis
local max_history = 100   -- Keep last 100 readings

-- MQTT client setup
local client = mqtt.Client("kaiabc_gateway", 120)
client:connect("192.168.1.100", 1883)  -- Configure your broker

-- Message processing
client:on("message", function(client, topic, data)
    if topic == "fdrs/kaiabc" then
        local success, msg = pcall(sjson.decode, data)
        if success and msg.id then
            -- Update node state
            nodes[msg.id] = {
                phase = msg.phase,
                temp = msg.temp or 25.0,
                order_param = msg.order_param or 0.0,
                timestamp = msg.timestamp or tmr.now(),
                last_seen = tmr.now()
            }
            
            -- Add to history
            table.insert(network_history, {
                timestamp = tmr.now(),
                node_count = (function()
                    local count = 0
                    for _ in pairs(nodes) do count = count + 1 end
                    return count
                end)(),
                network_order = calculateNetworkOrder(),
                active_nodes = nodes
            })
            
            -- Trim history
            if #network_history > max_history then
                table.remove(network_history, 1)
            end
            
            -- Publish network statistics
            local stats = {
                node_count = #network_history[#network_history].active_nodes,
                network_order = network_history[#network_history].network_order,
                timestamp = tmr.now()
            }
            client:publish("fdrs/kaiabc/network", sjson.encode(stats), 0, 0)
            
            -- Console output
            print(string.format("[Gateway] Network: %d nodes, Order: %.3f", 
                  stats.node_count, stats.network_order))
        end
    end
end)

-- Calculate network-wide order parameter
function calculateNetworkOrder()
    local sum_cos, sum_sin, count = 0, 0, 0
    for id, node in pairs(nodes) do
        sum_cos = sum_cos + math.cos(node.phase)
        sum_sin = sum_sin + math.sin(node.phase)
        count = count + 1
    end
    if count > 0 then
        return math.sqrt(sum_cos^2 + sum_sin^2) / count
    end
    return 0.0
end

-- Clean up inactive nodes
local cleanup_timer = tmr.create()
cleanup_timer:register(600000, tmr.ALARM_AUTO, function()  -- Every 10 minutes
    local current_time = tmr.now()
    for id, node in pairs(nodes) do
        if current_time - node.last_seen > 3600000 then  -- 1 hour timeout
            nodes[id] = nil
            print("Removed inactive node: " .. id)
        end
    end
end)

-- Subscribe and start
client:subscribe("fdrs/kaiabc", 0)
cleanup_timer:start()
print("KaiABC Gateway started")
```

### 4. Configuration File (`config.lua`)

```lua
-- Configuration for KaiABC ELM11 Implementation

return {
    -- MQTT Settings
    mqtt = {
        broker = "192.168.1.100",  -- IP address of MQTT broker
        port = 1883,               -- MQTT port (default 1883)
        topic = "fdrs/kaiabc",     -- Base topic for synchronization
        client_id = "kaiabc_node_1" -- Unique client ID
    },
    
    -- Oscillator Parameters
    oscillator = {
        node_id = 1,              -- Unique node identifier (1-255)
        period_hours = 24.0,      -- Base circadian period
        q10 = 1.1,                -- Temperature compensation factor
        coupling_strength = 0.1,  -- Kuramoto coupling parameter K
        update_interval_min = 5   -- Update interval in minutes
    },
    
    -- Hardware Settings
    hardware = {
        temp_sensor = "bme280",   -- "bme280", "ds18b20", or "dht22"
        i2c = {
            sda = 5,              -- GPIO for I2C SDA
            scl = 4               -- GPIO for I2C SCL
        },
        led_pin = 2               -- GPIO for status LED
    },
    
    -- Debug and Logging
    debug = {
        enabled = true,           -- Enable debug output
        log_level = "info",       -- "debug", "info", "warn", "error"
        serial_baud = 115200      -- Serial baud rate
    }
}
```

---

## 🔄 Integration with FDRS

### Communication Bridge

Since ELM11 uses MQTT instead of ESP-NOW/LoRa, create a bridge script that runs on an ESP32 to translate between protocols:

```lua
-- FDRS Bridge: MQTT ↔ ESP-NOW (runs on ESP32 with FDRS)

-- This would be implemented in Arduino/C++ on ESP32
-- Receives MQTT messages from ELM11 nodes
-- Translates to FDRS DataReading format
-- Sends via ESP-NOW to FDRS network
```

### Data Format Compatibility

- **ELM11 MQTT Message:**
```json
{
  "id": 1,
  "phase": 1.57,
  "temp": 25.3,
  "order_param": 0.85
}
```

- **FDRS DataReading Equivalent:**
```cpp
DataReading reading;
reading.id = KAIABC_PHASE_T;
reading.t = 1;  // Node ID
reading.d = encodeKaiABCMessage(phase, temp, order_param);
```

---

## 🧪 Testing and Validation Plan

### Phase 1: Single Node Testing
1. **Setup:** One ELM11 with temperature sensor
2. **Test:** Phase evolution without coupling
3. **Validate:** Period changes with temperature (Q10 effect)
4. **Duration:** 1-2 days

### Phase 2: Multi-Node Testing (WiFi)
1. **Setup:** 2-3 ELM11 nodes on same WiFi network
2. **Test:** Synchronization via MQTT
3. **Validate:** Order parameter convergence
4. **Duration:** 1 week

### Phase 3: FDRS Integration Testing
1. **Setup:** ELM11 nodes + ESP32 FDRS gateway
2. **Test:** MQTT ↔ ESP-NOW bridge
3. **Validate:** Data flows through FDRS network
4. **Duration:** 1 week

### Phase 4: Performance Validation
1. **Setup:** Compare Lua vs C++ implementations
2. **Test:** Synchronization time, power consumption
3. **Validate:** Theoretical predictions
4. **Duration:** 2-4 weeks

---

## 📊 Performance Expectations

### Lua vs C++ Comparison

| Metric | Lua (ELM11) | C++ (ESP32) | Notes |
|--------|-------------|-------------|-------|
| Update Frequency | 5-15 min | 1-2 min | Lua interpretation overhead |
| Memory Usage | ~50 KB | ~10 KB | Lua runtime overhead |
| Power Consumption | Similar | Similar | Hardware dependent |
| Development Speed | Fast | Slower | Rapid prototyping |
| Synchronization Time | ~20-30 days | ~16 days | Based on theory |
| Code Size | ~500 lines | ~2,900 lines | Excluding FDRS |

### Advantages of Lua Implementation
- **Rapid Prototyping:** Change code without recompiling
- **Easy Configuration:** Lua tables vs C++ defines
- **Rich Ecosystem:** Leverage existing Lua libraries
- **Cross-Platform:** Same code works on different Lua hardware
- **Educational:** Easier to understand and modify

### Disadvantages
- **Performance:** Slower execution
- **Memory:** Higher overhead
- **Hardware Limits:** No direct radio access
- **Ecosystem:** Smaller IoT library ecosystem

---

## 🛠️ Hardware Requirements

### Minimum Setup (WiFi Only)
- **3× ELM11 boards** (~$20 each)
- **3× BME280 sensors** (~$5 each)
- **MQTT broker** (Raspberry Pi or PC with Mosquitto)
- **WiFi network** for communication

### Extended Setup (FDRS Compatible)
- **Additional ESP32** for MQTT ↔ ESP-NOW bridge
- **LoRa modules** (optional, via UART bridge)
- **Power supplies** and USB cables

**Estimated Cost:** $100-200 for basic testing setup

---

## 📈 Development Roadmap

### Week 1-2: Core Implementation
- [ ] Create `kaiabc.lua` library
- [ ] Implement basic node script
- [ ] Test single node operation
- [ ] Validate Q10 temperature compensation

### Week 3-4: Multi-Node Testing
- [ ] Create gateway script
- [ ] Test MQTT communication
- [ ] Implement 2-3 node synchronization
- [ ] Measure order parameter convergence

### Week 5-6: FDRS Integration
- [ ] Create MQTT ↔ ESP-NOW bridge
- [ ] Test with FDRS gateway
- [ ] Validate data flow
- [ ] Compare with C++ implementation

### Week 7-8: Optimization and Documentation
- [ ] Performance optimization
- [ ] Error handling and recovery
- [ ] Complete documentation
- [ ] Create examples and tutorials

---

## 🔍 Research Opportunities

### Unique Advantages of Lua Implementation
1. **Rapid Parameter Tuning:** Change Q10, K, period instantly
2. **Advanced Analytics:** Easy to add statistical analysis
3. **Visualization:** Direct integration with graphing libraries
4. **Educational Tool:** Perfect for teaching synchronization concepts

### Potential Extensions
1. **Adaptive Coupling:** Dynamically adjust K based on network conditions
2. **Multiple Oscillators:** Different periods for different applications
3. **Hierarchical Networks:** Gateway-based synchronization layers
4. **Machine Learning:** Use data to optimize parameters

---

## ⚠️ Important Considerations

### Technical Limitations
- **No Deep Sleep:** ELM11 may not support ultra-low power modes
- **Memory Constraints:** Large networks may exceed RAM limits
- **Timing Precision:** Lua's `tmr.now()` may be less precise than C++
- **Network Latency:** MQTT adds communication delays

### Compatibility Notes
- **FDRS Version:** Ensure MQTT gateway example works with your setup
- **Lua Version:** Confirm ELM11 uses compatible Lua (5.1/5.2/5.3)
- **Libraries:** Verify availability of MQTT, sjson, sensor libraries
- **WiFi Range:** MQTT requires reliable WiFi vs ESP-NOW's mesh capability

---

## 📚 Resources and References

### ELM11 Documentation
- [ELM11 Official Site](https://elm-chan.org/)
- Lua MQTT libraries for embedded systems
- Sensor integration examples

### FDRS Integration
- [`examples/KaiABC_Sensor/`](../examples/KaiABC_Sensor/) - C++ reference
- [`src/fdrs_kaiABC.h`](../src/fdrs_kaiABC.h) - Core algorithm reference
- [`research/KaiABC/`](../research/KaiABC/) - Theoretical foundation

### Lua for Embedded Systems
- NodeMCU documentation (similar architecture)
- ESP8266 Lua examples
- Embedded Lua best practices

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- [ ] Single ELM11 node maintains stable 24-hour oscillation
- [ ] Temperature compensation works (period changes with temp)
- [ ] MQTT communication functional
- [ ] Basic order parameter calculation

### Full Implementation
- [ ] Multi-node synchronization via MQTT
- [ ] Order parameter > 0.8 achieved
- [ ] Integration with FDRS via bridge
- [ ] Performance comparable to C++ version
- [ ] Complete documentation and examples

---

## 🤝 Contributing

This ELM11 implementation provides an excellent opportunity for:
- **Educational Use:** Teaching synchronization concepts
- **Rapid Prototyping:** Testing new oscillator parameters
- **Cross-Platform Development:** Lua runs on many embedded platforms
- **Research Extensions:** Easy to add new features

**Get Involved:** Test the code, report issues, suggest improvements!

---

**Status:** Ready for implementation  
**Next Step:** Create `kaiabc.lua` and test on ELM11 hardware  
**Contact:** Open GitHub issues for questions or contributions

---

*This plan transforms the theoretical KaiABC research into practical Lua code, making biological synchronization accessible to a broader audience through the ELM11 platform.*
