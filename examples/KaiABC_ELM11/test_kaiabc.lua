-- ELM11 KaiABC Test Script
-- Demonstrates UART coprocessor communication

print("Starting ELM11 KaiABC Test...")

-- Import the KaiABC interface module
local kaiabc = require("ELM11_KaiABC_Interface")

-- Run the example
kaiabc.run_example()

print("Test completed.")