-- ELM11 KaiABC UART Coprocessor Interface
-- This module provides Lua interface to ESP32 coprocessor for KaiABC calculations

local uart = require("uart")
local json = require("json")  -- Assuming JSON library is available

KaiABC_ELM11 = {}

-- Configuration
local UART_ID = 0  -- ELM11 UART0
local BAUD_RATE = 115200
local TIMEOUT_MS = 5000
local COMMAND_DELAY_MS = 100

-- Initialize UART connection to ESP32 coprocessor
function KaiABC_ELM11.init()
    -- Configure UART for ESP32 communication
    uart.setup(UART_ID, BAUD_RATE, 8, uart.PARITY_NONE, uart.STOPBITS_1)

    print("ELM11 KaiABC UART interface initialized")
    print("Connected to ESP32 coprocessor at " .. BAUD_RATE .. " baud")

    -- Test connection
    local success = KaiABC_ELM11.test_connection()
    if success then
        print("ESP32 coprocessor connection verified")
    else
        print("Warning: ESP32 coprocessor not responding")
    end

    return success
end

-- Test connection to ESP32 coprocessor
function KaiABC_ELM11.test_connection()
    local response = KaiABC_ELM11.send_command("get_sync_status")
    return response ~= nil and response.status == "success"
end

-- Send command to ESP32 coprocessor and receive response
function KaiABC_ELM11.send_command(command_type, data)
    -- Create JSON command
    local command = {
        command = command_type,
        data = data or {}
    }

    local command_json = json.encode(command)
    print("Sending command: " .. command_json)

    -- Send command via UART
    uart.write(UART_ID, command_json .. "\n")

    -- Wait for response
    local start_time = tmr.now()
    local response_json = ""

    while (tmr.now() - start_time) < (TIMEOUT_MS * 1000) do
        if uart.available(UART_ID) > 0 then
            local chunk = uart.read(UART_ID, uart.available(UART_ID))
            response_json = response_json .. chunk

            -- Check if we have a complete JSON response
            if response_json:find("}\n?$") then
                break
            end
        end
        tmr.delay(COMMAND_DELAY_MS * 1000)  -- Small delay to prevent busy waiting
    end

    if response_json == "" then
        print("Timeout waiting for ESP32 response")
        return nil
    end

    -- Parse JSON response
    local success, response = pcall(json.decode, response_json)
    if not success then
        print("Failed to parse ESP32 response: " .. response_json)
        return nil
    end

    print("Received response: " .. response_json)
    return response
end

-- Get current network state from ESP32
function KaiABC_ELM11.get_network_state()
    local response = KaiABC_ELM11.send_command("get_state")

    if response and response.status == "success" then
        return response.data
    else
        print("Failed to get network state")
        return nil
    end
end

-- Update oscillator parameters on ESP32
function KaiABC_ELM11.update_oscillator(oscillator_index, frequency, amplitude)
    local data = {
        oscillator = oscillator_index,
        frequency = frequency,
        amplitude = amplitude
    }

    local response = KaiABC_ELM11.send_command("update_oscillator", data)

    if response and response.status == "success" then
        return true
    else
        print("Failed to update oscillator " .. oscillator_index)
        return false
    end
end

-- Reset the KaiABC network on ESP32
function KaiABC_ELM11.reset_network()
    local response = KaiABC_ELM11.send_command("reset_network")

    if response and response.status == "success" then
        print("KaiABC network reset successfully")
        return true
    else
        print("Failed to reset network")
        return false
    end
end

-- Get synchronization status
function KaiABC_ELM11.get_sync_status()
    local response = KaiABC_ELM11.send_command("get_sync_status")

    if response and response.status == "success" then
        return response.data
    else
        print("Failed to get sync status")
        return nil
    end
end

-- High-level function to synchronize network
function KaiABC_ELM11.synchronize_network()
    -- Get current sync status
    local status = KaiABC_ELM11.get_sync_status()

    if not status then
        return false, "Failed to get sync status"
    end

    if status.is_synchronized then
        print("Network already synchronized (order parameter: " .. status.order_parameter .. ")")
        return true, status
    else
        print("Network not yet synchronized (order parameter: " .. status.order_parameter .. ")")
        print("Waiting for synchronization...")

        -- Wait a bit and check again (in real implementation, this would be event-driven)
        tmr.delay(5 * 1000 * 1000)  -- 5 second delay

        status = KaiABC_ELM11.get_sync_status()
        return status and status.is_synchronized, status
    end
end

-- Utility function to print network state
function KaiABC_ELM11.print_network_state()
    local state = KaiABC_ELM11.get_network_state()

    if not state then
        print("Unable to retrieve network state")
        return
    end

    print("=== KaiABC Network State ===")
    print("Global sync phase: " .. string.format("%.4f", state.global_sync_phase))
    print("Network uptime: " .. state.network_uptime .. " seconds")
    print("Oscillators:")

    for i, osc in ipairs(state.oscillators) do
        print(string.format("  %d: Phase=%.4f, Freq=%.4f, Amp=%.4f",
              i-1, osc.phase, osc.frequency, osc.amplitude))
    end
end

-- Example usage and test function
function KaiABC_ELM11.run_example()
    print("=== ELM11 KaiABC Coprocessor Example ===")

    -- Initialize connection
    if not KaiABC_ELM11.init() then
        print("Failed to initialize coprocessor connection")
        return
    end

    -- Reset network to known state
    KaiABC_ELM11.reset_network()

    -- Print initial state
    print("\nInitial network state:")
    KaiABC_ELM11.print_network_state()

    -- Wait for synchronization
    print("\nWaiting for network synchronization...")
    local synced, status = KaiABC_ELM11.synchronize_network()

    if synced then
        print("Network synchronized successfully!")
    else
        print("Network synchronization incomplete")
    end

    -- Update an oscillator
    print("\nUpdating oscillator 0 frequency to 1.1 Hz...")
    KaiABC_ELM11.update_oscillator(0, 1.1, 1.0)

    -- Print final state
    print("\nFinal network state:")
    KaiABC_ELM11.print_network_state()

    print("\nExample completed")
end

return KaiABC_ELM11