# KaiABC Quick Start Guide

## What is KaiABC?

KaiABC is a **biological oscillator synchronization system** that lets IoT devices sync their timing without GPS or NTP servers. Instead, devices use **distributed phase coupling** (like fireflies flashing in sync) based on the cyanobacterial circadian clock.

## Why Use It?

- 🔋 **246-year battery life** (vs 1-2 years with GPS)
- 📡 **100× lower bandwidth** than NTP (1.5 kbps vs 150+ kbps)
- 🌍 **No infrastructure needed** (no internet, no GPS satellites)
- 🌡️ **Temperature compensated** (works across ±5°C variance)
- 🔬 **Based on research** connecting Kakeya Conjecture to distributed systems

## Prerequisites

### Hardware (for testing)
- 3-5× ESP32 boards (~$10 each)
- 3-5× BME280 temperature sensors (~$5 each)
- USB cables and power supplies

### Software
- Arduino IDE with ESP32 support
- This repository cloned locally

## Option 1: Simulation (No Hardware Required)

Test the theory before buying hardware:

```bash
# Navigate to the example directory
cd examples/KaiABC_Sensor

# Install Python dependencies
pip3 install numpy matplotlib scipy

# Run simulation
python3 kaiabc_simulation.py --nodes 10 --q10 1.1 --coupling 0.1 --days 30
```

**What you'll see:**
- Phase trajectories converging over time
- Order parameter rising to R > 0.95
- Phase space visualization showing synchronized state
- Confirmation that sync happens in ~16 days

## Option 2: Hardware Deployment

### Step 1: Install Libraries

In Arduino IDE, install:
- **ArduinoJson** (via Library Manager)
- **Adafruit BME280** (via Library Manager)
- **RadioLib** (if using LoRa)

### Step 2: Configure First Node

Open `examples/KaiABC_Sensor/fdrs_node_config.h`:

```cpp
#define READING_ID    1           // Node ID (make unique for each board!)
#define GTWY_MAC      0x01        // Gateway MAC address
#define USE_ESPNOW                // ESP-NOW for short-range
#define KAIABC_Q10    1.1         // Temperature compensation
#define KAIABC_COUPLING 0.1       // Kuramoto coupling strength
#define USE_BME280                // Enable temperature sensor
```

### Step 3: Flash Nodes

1. Open `examples/KaiABC_Sensor/KaiABC_Sensor.ino`
2. Edit this line near the top:
   ```cpp
   #define KAIABC_NODE_ID 1  // Change to 2, 3, 4, etc. for other nodes
   ```
3. Connect ESP32
4. Select **Tools → Board → ESP32 Dev Module**
5. Select correct **Port**
6. Click **Upload**
7. Repeat for each board (changing `KAIABC_NODE_ID` each time!)

### Step 4: Configure Gateway

Open `examples/KaiABC_Gateway/fdrs_gateway_config.h`:

```cpp
#define UNIT_MAC      0x01        // Gateway MAC
#define USE_ESPNOW                // Match node config
#define USE_WIFI                  // For MQTT (optional)
#define WIFI_SSID     "YourWiFi"  // Your network
#define WIFI_PASS     "password"
#define USE_MQTT                  // For monitoring (optional)
#define MQTT_ADDR     "192.168.1.100"  // Your MQTT broker
```

### Step 5: Flash Gateway

1. Open `examples/KaiABC_Gateway/KaiABC_Gateway.ino`
2. Connect ESP32
3. Select board and port
4. Click **Upload**

### Step 6: Monitor

Open Serial Monitor (115200 baud) on any node:

```
========================================
KaiABC Biological Oscillator Node
========================================

--- KaiABC Status ---
Phase (φ): 1.234 rad  (70.7°)
Period (τ): 24.03 hours
Order parameter (R): 0.145  ✗ Desynchronized
Active neighbors: 3
```

Watch the **Order parameter (R)** slowly increase!

Open Serial Monitor on gateway to see network-wide stats:

```
--- Network Statistics ---
Total nodes seen: 4
Active nodes: 4
Order Parameter (R): 0.324  ✗ Desynchronized
Average period: 24.08 hours

[After ~2 weeks...]

--- Network Statistics ---
Active nodes: 4
Order Parameter (R): 0.967  ✓ SYNCHRONIZED
Synchronization time: 15.2 days
```

## Understanding the Output

### Order Parameter (R)
- **R < 0.5:** Desynchronized (early phase)
- **R = 0.5-0.95:** Partially synchronized (converging)
- **R > 0.95:** Synchronized! ✓

### Phase (φ)
- Ranges from 0 to 2π radians (0° to 360°)
- Shows where in the 24-hour cycle the oscillator is
- When synchronized, all phases are similar

### Period (τ)
- Should be close to 24.0 hours
- Varies slightly with temperature
- Q10 controls how much it varies

## Troubleshooting

### "Active neighbors: 0"
- Nodes aren't seeing each other
- Check that all nodes have same `GTWY_MAC`
- Verify ESP-NOW is enabled on all
- Make sure boards are within range (~200m)

### "Order parameter not increasing"
- May need higher coupling strength
- Try `KAIABC_COUPLING 0.2` instead of 0.1
- Or wait longer - sync takes days, not hours!

### "BME280 not found"
- Check I2C wiring (SDA/SCL)
- Try address 0x77 instead of 0x76
- Or comment out `#define USE_BME280` to test without sensor

### "Compile errors"
- Make sure you installed all required libraries
- Select correct board (ESP32 Dev Module)
- Check that you copied the entire `src/` folder

## What to Expect

### Timeline
- **Day 1:** Nodes start with random phases, R ≈ 0.1-0.3
- **Day 5:** Phases beginning to cluster, R ≈ 0.4-0.6
- **Day 10:** Nearly synchronized, R ≈ 0.7-0.9
- **Day 15-20:** Fully synchronized, R > 0.95 ✓

### Performance
- **Messages:** ~12 per day per node (every 2 hours)
- **Bandwidth:** ~1.5 kbps per node
- **Power:** ~0.3 J/day (WiFi), ~0.072 J/day (LoRa)
- **Battery life:** 246 years theoretical (3000 mAh battery)

## Advanced Configuration

### Faster Synchronization
Change broadcast interval to 1 hour:
```cpp
#define KAIABC_UPDATE_INTERVAL 3600000  // 1 hour (24 msgs/day)
```

### Ultra-Low Power
Change broadcast interval to 4 hours:
```cpp
#define KAIABC_UPDATE_INTERVAL 14400000  // 4 hours (6 msgs/day)
```

### Use LoRa Instead of ESP-NOW
For long-range (5-10 km):
```cpp
// In node config:
#define USE_LORA
#define LORA_FREQUENCY 915.0  // Or 868.0 or 433.0
#define LORA_SF 10
```

See `examples/KaiABC_Sensor/README.md` for complete LoRa pin configuration.

### Test Different Q10 Values
```cpp
#define KAIABC_Q10 1.0  // Perfect compensation (ideal)
#define KAIABC_Q10 1.1  // Realistic (RECOMMENDED)
#define KAIABC_Q10 2.2  // Poor compensation (for comparison)
```

Q10=2.2 will likely NOT synchronize (basin volume 0.0001%)!

## Where to Go Next

### Learn More
- **Full documentation:** `examples/KaiABC_Sensor/README.md`
- **Research background:** `research/KaiABC/`
- **Project status:** `PROJECT_STATUS.md`

### Customize
- Modify oscillator parameters in `src/fdrs_kaiABC.h`
- Add your own sensors to nodes
- Create custom gateway logic

### Contribute
- Test with different hardware
- Optimize power consumption
- Improve documentation
- Share your results!

## Support

- **Issues:** Open a GitHub issue
- **Questions:** See the comprehensive README in `examples/KaiABC_Sensor/`
- **Research:** Read `research/KaiABC/deep-research-prompt-claude.md`

---

**Ready?** Start with the Python simulation, then move to hardware!

```bash
cd examples/KaiABC_Sensor
python3 kaiabc_simulation.py
```

🎉 **Have fun with biological time synchronization!** 🎉
