# iTantra Performance Evaluation Summary Report

> [!IMPORTANT]
> **Simulation Environment Disclaimer**:
> All evaluations were conducted in a software simulation environment using SimulatedSDR software-loopback and stochastic AWGN/Rayleigh channel models. No physical radio hardware (USRP, HackRF, RTL-SDR) or phone-to-phone RF tests were conducted.

## 1. Overall Transmission Success & Retries

| Metric | Measured Value |
|---|---|
| Total Messages Tested | **224** |
| Successfully Delivered | **112** (50.0%) |
| Messages Failed (Retries Exhausted) | **0** |
| Messages Expired (TTL Exceeded) | **112** |
| Average Retries / Message | **0.14** (Min: 0, Max: 2) |

## 2. End-to-End Delivery Latency

| Latency Measure | Value (ms) |
|---|---|
| Average Latency (All Messages) | **95.62 ms** |
| Average Latency (Delivered Messages Only) | **190.88 ms** |
| Minimum Latency | **0.20 ms** |
| Maximum Latency | **405.48 ms** |

## 3. Radio Link Packet Loss & Integrity (CRC-16)

| Metric | Packets | Rate |
|---|---|---|
| Packets Transmitted | 6499 | - |
| Packets Lost in Transit | 799 | **12.29%** |
| Packets Checked for Integrity | 5700 | - |
| CRC Integrity Check Failures | 516 | **9.05%** |

## 4. Priority-Level Delivery Performance

| Priority Level | Total Sent | Delivered | Expired | Failed | Success Rate | Avg Latency |
|---|---|---|---|---|---|---|
| **EMERGENCY** | 112 | 56 | 56 | 0 | **50.0%** | 100.0 ms |
| **NORMAL** | 112 | 56 | 56 | 0 | **50.0%** | 91.24 ms |

## 5. Time-To-Live (TTL) Expiration Metrics

| Metric | Value |
|---|---|
| Messages with Configured TTL | **112** |
| Messages Expired Due to TTL | **112** |
| TTL Expiration Rate | **100.0%** |

## 6. Channel Performance Comparison: AWGN vs. Rayleigh

| Channel Model | SNR (dB) | Trials | Delivered | Success Rate | Avg Retries | Symbol MSE | Avg Latency |
|---|---|---|---|---|---|---|---|
| **AWGN** | 15.0 dB | 112 | 56 | **50.0%** | 0.15 | 0.015826 | 98.3 ms |
| **RAYLEIGH** | 15.0 dB | 112 | 56 | **50.0%** | 0.12 | 0.015835 | 92.9 ms |
