#include &lt;Arduino.h&gt;
#include &lt;HardwareSerial.h&gt;

// UART Configuration
#define UART_BAUD 115200
#define UART_TIMEOUT 5000  // 5 second timeout
#define BUFFER_SIZE 1024

// KaiABC Constants (from research/KaiABC/)
#define KAIABC_NUM_OSCILLATORS 10
#define KAIABC_COUPLING_STRENGTH 0.1
#define KAIABC_PHASE_DIFF_THRESHOLD 0.01
#define KAIABC_SYNC_TIME_WINDOW 3600000  // 1 hour in milliseconds

// Global variables
HardwareSerial SerialUART(2);  // Use UART2 for ELM11 communication
char uartBuffer[BUFFER_SIZE];
unsigned long lastCommandTime = 0;

// KaiABC Data Structures
struct OscillatorState {
    float phase;
    float frequency;
    float amplitude;
    unsigned long lastUpdate;
};

struct NetworkState {
    OscillatorState oscillators[KAIABC_NUM_OSCILLATORS];
    float couplingMatrix[KAIABC_NUM_OSCILLATORS][KAIABC_NUM_OSCILLATORS];
    float globalSyncPhase;
    unsigned long networkStartTime;
};

// Global network state
NetworkState kaiabcNetwork;

// Forward declarations
void initializeKaiABC();
void updateOscillator(int index, float dt);
float calculateCouplingForce(int oscillatorIndex);
void synchronizeNetwork();
bool parseUARTCommand(String command);
void sendUARTResponse(String response);
String createJSONResponse(String status, String data = "");
float calculatePhaseDifference(float phase1, float phase2);

void setup() {
    // Initialize serial for debugging
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 KaiABC UART Coprocessor Starting...");

    // Initialize UART for ELM11 communication
    SerialUART.begin(UART_BAUD, SERIAL_8N1, 16, 17);  // RX=16, TX=17
    Serial.println("UART initialized at 115200 baud");

    // Initialize KaiABC network
    initializeKaiABC();
    Serial.println("KaiABC network initialized");

    Serial.println("ESP32 Coprocessor ready for commands");
}

void loop() {
    // Check for UART commands
    if (SerialUART.available()) {
        String command = SerialUART.readStringUntil('\n');
        command.trim();

        if (command.length() > 0) {
            lastCommandTime = millis();
            Serial.print("Received command: ");
            Serial.println(command);

            if (parseUARTCommand(command)) {
                Serial.println("Command processed successfully");
            } else {
                Serial.println("Command processing failed");
                sendUARTResponse(createJSONResponse("error", "Invalid command"));
            }
        }
    }

    // Update KaiABC network periodically (every 100ms)
    static unsigned long lastUpdate = 0;
    if (millis() - lastUpdate > 100) {
        synchronizeNetwork();
        lastUpdate = millis();
    }

    // Check for timeout
    if (millis() - lastCommandTime > UART_TIMEOUT && lastCommandTime > 0) {
        Serial.println("UART timeout - resetting connection");
        lastCommandTime = 0;
    }
}

void initializeKaiABC() {
    kaiabcNetwork.networkStartTime = millis();
    kaiabcNetwork.globalSyncPhase = 0.0;

    // Initialize oscillators with random phases
    for (int i = 0; i < KAIABC_NUM_OSCILLATORS; i++) {
        kaiabcNetwork.oscillators[i].phase = random(0, 628) / 100.0;  // 0-2π
        kaiabcNetwork.oscillators[i].frequency = 1.0 + (random(-10, 10) / 100.0);  // ~1.0 Hz ±10%
        kaiabcNetwork.oscillators[i].amplitude = 1.0;
        kaiabcNetwork.oscillators[i].lastUpdate = millis();

        // Initialize coupling matrix (all-to-all coupling)
        for (int j = 0; j < KAIABC_NUM_OSCILLATORS; j++) {
            kaiabcNetwork.couplingMatrix[i][j] = (i != j) ? KAIABC_COUPLING_STRENGTH : 0.0;
        }
    }
}

void updateOscillator(int index, float dt) {
    if (index < 0 || index >= KAIABC_NUM_OSCILLATORS) return;

    OscillatorState& osc = kaiabcNetwork.oscillators[index];

    // Calculate coupling force
    float couplingForce = calculateCouplingForce(index);

    // Update phase using Kuramoto model
    osc.phase += (osc.frequency + couplingForce) * dt;

    // Keep phase in [0, 2π]
    while (osc.phase >= 2 * PI) osc.phase -= 2 * PI;
    while (osc.phase < 0) osc.phase += 2 * PI;

    osc.lastUpdate = millis();
}

float calculateCouplingForce(int oscillatorIndex) {
    float force = 0.0;

    for (int j = 0; j < KAIABC_NUM_OSCILLATORS; j++) {
        if (j == oscillatorIndex) continue;

        float phaseDiff = calculatePhaseDifference(
            kaiabcNetwork.oscillators[oscillatorIndex].phase,
            kaiabcNetwork.oscillators[j].phase
        );

        force += kaiabcNetwork.couplingMatrix[oscillatorIndex][j] *
                sin(phaseDiff) * kaiabcNetwork.oscillators[j].amplitude;
    }

    return force;
}

void synchronizeNetwork() {
    float dt = 0.1;  // 100ms time step

    // Update all oscillators
    for (int i = 0; i < KAIABC_NUM_OSCILLATORS; i++) {
        updateOscillator(i, dt);
    }

    // Calculate global synchronization
    float avgPhase = 0.0;
    float orderParameter = 0.0;

    for (int i = 0; i < KAIABC_NUM_OSCILLATORS; i++) {
        avgPhase += kaiabcNetwork.oscillators[i].phase;
        orderParameter += cos(kaiabcNetwork.oscillators[i].phase);
    }

    avgPhase /= KAIABC_NUM_OSCILLATORS;
    orderParameter /= KAIABC_NUM_OSCILLATORS;

    kaiabcNetwork.globalSyncPhase = avgPhase;

    // Check for synchronization
    static float lastOrderParameter = 0.0;
    if (abs(orderParameter - lastOrderParameter) < KAIABC_PHASE_DIFF_THRESHOLD) {
        // Network is synchronized
        Serial.print("Network synchronized! Order parameter: ");
        Serial.println(orderParameter);
    }
    lastOrderParameter = orderParameter;
}

bool parseUARTCommand(String command) {
    // Expected JSON format: {"command": "type", "data": {...}}
    if (!command.startsWith("{") || !command.endsWith("}")) {
        return false;
    }

    // Simple JSON parsing (in production, use ArduinoJson library)
    if (command.indexOf("\"command\":\"get_state\"") != -1) {
        // Return current network state
        String stateData = "{";
        stateData += "\"oscillators\":[";
        for (int i = 0; i < KAIABC_NUM_OSCILLATORS; i++) {
            if (i > 0) stateData += ",";
            stateData += "{";
            stateData += "\"phase\":" + String(kaiabcNetwork.oscillators[i].phase, 4) + ",";
            stateData += "\"frequency\":" + String(kaiabcNetwork.oscillators[i].frequency, 4) + ",";
            stateData += "\"amplitude\":" + String(kaiabcNetwork.oscillators[i].amplitude, 4);
            stateData += "}";
        }
        stateData += "],";
        stateData += "\"global_sync_phase\":" + String(kaiabcNetwork.globalSyncPhase, 4) + ",";
        stateData += "\"network_uptime\":" + String((millis() - kaiabcNetwork.networkStartTime) / 1000);
        stateData += "}";

        sendUARTResponse(createJSONResponse("success", stateData));
        return true;

    } else if (command.indexOf("\"command\":\"update_oscillator\"") != -1) {
        // Update specific oscillator parameters
        // Parse oscillator index and new parameters
        int oscIndex = -1;
        float newFreq = -1.0;
        float newAmp = -1.0;

        // Simple parsing - look for parameters
        int idx = command.indexOf("\"oscillator\":");
        if (idx != -1) {
            oscIndex = command.substring(idx + 13, command.indexOf(",", idx)).toInt();
        }

        idx = command.indexOf("\"frequency\":");
        if (idx != -1) {
            newFreq = command.substring(idx + 12, command.indexOf(",", idx)).toFloat();
        }

        idx = command.indexOf("\"amplitude\":");
        if (idx != -1) {
            newAmp = command.substring(idx + 12, command.indexOf("}", idx)).toFloat();
        }

        if (oscIndex >= 0 && oscIndex < KAIABC_NUM_OSCILLATORS) {
            if (newFreq > 0) kaiabcNetwork.oscillators[oscIndex].frequency = newFreq;
            if (newAmp > 0) kaiabcNetwork.oscillators[oscIndex].amplitude = newAmp;

            sendUARTResponse(createJSONResponse("success", "Oscillator updated"));
            return true;
        }

    } else if (command.indexOf("\"command\":\"reset_network\"") != -1) {
        // Reset the entire network
        initializeKaiABC();
        sendUARTResponse(createJSONResponse("success", "Network reset"));
        return true;

    } else if (command.indexOf("\"command\":\"get_sync_status\"") != -1) {
        // Return synchronization status
        float orderParameter = 0.0;
        for (int i = 0; i < KAIABC_NUM_OSCILLATORS; i++) {
            orderParameter += cos(kaiabcNetwork.oscillators[i].phase);
        }
        orderParameter /= KAIABC_NUM_OSCILLATORS;

        String syncData = "{";
        syncData += "\"order_parameter\":" + String(orderParameter, 4) + ",";
        syncData += "\"is_synchronized\":" + String(orderParameter > 0.8 ? "true" : "false") + ",";
        syncData += "\"uptime_seconds\":" + String((millis() - kaiabcNetwork.networkStartTime) / 1000);
        syncData += "}";

        sendUARTResponse(createJSONResponse("success", syncData));
        return true;
    }

    return false;
}

void sendUARTResponse(String response) {
    SerialUART.println(response);
    Serial.print("Sent response: ");
    Serial.println(response);
}

String createJSONResponse(String status, String data) {
    String response = "{";
    response += "\"status\":\"" + status + "\",";
    if (data.length() > 0) {
        response += "\"data\":" + data;
    } else {
        response += "\"data\":null";
    }
    response += "}";
    return response;
}

float calculatePhaseDifference(float phase1, float phase2) {
    float diff = phase1 - phase2;
    while (diff > PI) diff -= 2 * PI;
    while (diff < -PI) diff += 2 * PI;
    return diff;
}