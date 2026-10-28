# iTantra: Indian Multilingual TTS & STT Communication System

iTantra is an Indian Multilingual Text-to-Speech (TTS) and Speech-to-Text (STT) Communication System integrated with a neural wireless transceiver (Joint Source-Channel Coding) and radio framing interfaces.

The architecture is partitioned into three decoupled modules:
1. **Speech Processing**: Multilingual acoustic synthesis, feature extraction, and recognition.
2. **Neural Transceiver**: PyTorch-based neural encoder, decoder, and channel simulations (AWGN and Rayleigh fading).
3. **Radio & Network**: Packet framing, preamble synchronization, CRC-16 integrity verification, and SDR loopback interfaces.

---

## Architecture Overview

```
[ Input Text (Indian Language) ]
               │
               ▼
   ┌───────────────────────┐
   │    src/speech/        │  TTS Engine (Waveform Synthesis)
   │    tts_engine.py      │
   └───────────┬───────────┘
               │  Audio (16 kHz PCM)
               ▼
   ┌───────────────────────┐
   │    src/speech/        │  STT Engine (Log-Mel Spectrogram, 80 bins)
   │    stt_engine.py      │
   └───────────┬───────────┘
               │  Acoustic Features [B, T, 80]
               ▼
   ┌───────────────────────┐
   │   src/transceiver/    │  Neural JSCC Encoder + Power Normalization
   │   encoder.py          │  E[|x|^2] <= 1.0
   └───────────┬───────────┘
               │  Channel Symbols [B, T, 32]
               ▼
   ┌───────────────────────┐
   │   src/transceiver/    │  Wireless Channel Simulation
   │   channel.py          │  • AWGN Channel
   └───────────┬───────────┘  • Rayleigh Flat Fading Channel
               │  Noisy Symbols
               ▼
   ┌───────────────────────┐
   │     src/radio/        │  Packet Framing (Preamble + CRC-16)
   │    packetizer.py      │  & Simulated SDR Software Loopback
   └───────────┬───────────┘
               │  Reassembled Symbols
               ▼
   ┌───────────────────────┐
   │   src/transceiver/    │  Neural JSCC Decoder
   │   decoder.py          │  Reconstructed Features [B, T, 80]
   └───────────┬───────────┘
               │
               ▼
   [ Reconstructed Speech / Evaluation ]
```

---

## Supported Indian Languages

iTantra provides genuine neural speech synthesis (TTS) and speech-to-text (STT) support for seven target Indian languages:

| Language Code | Language Name | Native Script | Locale | Unicode Block | Meta MMS-TTS Model | Neural Voice (Edge-TTS) | TTS Status | STT Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `te` | Telugu | Telugu | `te-IN` | U+0C00 - U+0C7F | `facebook/mms-tts-tel` | `te-IN-MohanNeural` | Verified | Verified |
| `hi` | Hindi | Devanagari | `hi-IN` | U+0900 - U+097F | `facebook/mms-tts-hin` | `hi-IN-MadhurNeural` | Verified | Verified |
| `ta` | Tamil | Tamil | `ta-IN` | U+0B80 - U+0BFF | `facebook/mms-tts-tam` | `ta-IN-ValluvarNeural` | Verified | Verified |
| `bn` | Bengali | Bengali | `bn-IN` | U+0980 - U+09FF | `facebook/mms-tts-ben` | `bn-IN-BashkarNeural` | Verified | Verified |
| `mr` | Marathi | Devanagari | `mr-IN` | U+0900 - U+097F | `facebook/mms-tts-mar` | `mr-IN-AarohiNeural` | Verified | Verified |
| `kn` | Kannada | Kannada | `kn-IN` | U+0C80 - U+0CFF | `facebook/mms-tts-kan` | `kn-IN-GaganNeural` | Verified | Verified |
| `ml` | Malayalam | Malayalam | `ml-IN` | U+0D00 - U+0D7F | `facebook/mms-tts-mal` | `ml-IN-MidhunNeural` | Verified | Verified |

> **Note on Scope**: Gujarati (`gu`) is explicitly out of project scope (`in_scope=False`, `status="out_of_scope"`). Pre-existing Hindi and Telugu neural voice implementations are fully preserved alongside their Meta MMS-TTS offline models.

---

## Multilingual TTS Architecture & Offline Operation

The Text-to-Speech system supports four synthesis backends configured in `configs/speech_config.yaml` or instantiated via `TTSEngine(backend=...)`:

1. **`mms` (Meta MMS-TTS)**: Offline neural speech synthesis using Meta's VITS checkpoints (`facebook/mms-tts-*`). Runs on CPU with PyTorch and Hugging Face Transformers.
2. **`neural` (Edge-TTS)**: High-fidelity cloud neural voices for Hindi (`hi-IN-MadhurNeural`), Telugu (`te-IN-MohanNeural`), and other Indian locales.
3. **`auto`**: Default production mode. Attempts offline neural synthesis via locally cached Meta MMS-TTS models first; falls back to Edge-TTS neural voices, and finally to lightweight acoustic mode if offline.
4. **`lightweight`**: Fast rule-based harmonic acoustic generator for offline CPU unit tests.

### Offline Model Storage & Cache Location
All seven MMS-TTS checkpoints are stored locally in the standard Hugging Face cache root:
- Location: `~/.cache/huggingface/hub/models--facebook--mms-tts-<lang>`
- Footprint per model: ~138 MB to ~164 MB
- Total cache footprint for all 7 languages: ~1.00 GB
- Offline loading: The engine uses `local_files_only=True` to guarantee inference without network connectivity.

---

## Project Structure

```
iTantra/
├── .agents/
│   └── rules/
│       └── itantra-guidelines.md   # Architectural boundaries and rules
├── configs/
│   ├── channel_config.yaml         # Transceiver, channel, and radio parameters
│   └── speech_config.yaml          # Audio sample rate, language registry, TTS/STT
├── output/                         # Verified genuine audio WAV samples for all 7 languages
├── src/
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── stt_engine.py           # STT engine and log-Mel feature extraction
│   │   ├── tts_engine.py           # Multilingual Indian TTS engine (MMS + Neural)
│   │   ├── translator.py          # Conversational translation engine
│   │   └── transliterate.py        # Roman-to-Indic script transliteration
│   ├── transceiver/
│   │   ├── __init__.py
│   │   ├── channel.py              # AWGN and Rayleigh fading channels (PyTorch)
│   │   ├── encoder.py              # Neural JSCC encoder with power normalization
│   │   └── decoder.py              # Neural JSCC decoder
│   ├── radio/
│   │   ├── __init__.py
│   │   ├── packetizer.py           # Framing, sync preamble, and CRC-16
│   │   └── sdr_interface.py        # Abstract SDR, SimulatedSDR loopback, & HardwareSDRStub
│   └── pipeline.py                 # End-to-end integration pipeline and demo CLI
├── tests/
│   ├── test_channel.py             # Channel models & transceiver unit tests
│   ├── test_evaluation.py          # Objective speech & transmission evaluation
│   ├── test_network_transport.py   # Reliable framing & transport tests
│   └── test_speech.py              # Multilingual speech, packetizer & SDR unit tests
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## Getting Started

### 1. Environment Setup
The project runs on Python 3.10+ (tested and verified on Python 3.14.4 on CPU).

Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Synthesizing Speech via Python API
```python
from src.speech.tts_engine import TTSEngine

# Initialize TTS Engine with offline Meta MMS backend
tts = TTSEngine(backend="mms")

# Synthesize across any of the 7 supported languages
output_hi = tts.synthesize("नमस्ते", language="hi")
output_te = tts.synthesize("నమస్కారం", language="te")
output_ta = tts.synthesize("வணக்கம்", language="ta")
output_bn = tts.synthesize("নমস্কার", language="bn")
output_mr = tts.synthesize("नमस्कार", language="mr")
output_kn = tts.synthesize("ನಮಸ್ಕಾರ", language="kn")
output_ml = tts.synthesize("നമസ്കാരം", language="ml")

# Inspect model status matrix
matrix = tts.get_language_status_matrix()
for lang in matrix:
    print(f"{lang['name']}: configured={lang['configured']}, installed={lang['installed']}, verified={lang['tts_verified']}")
```

### 3. Transcribing Speech via Python API (STT)
```python
from src.speech.stt_engine import STTEngine
import soundfile as sf

# Initialize STT Engine with offline Whisper model
stt = STTEngine(backend="whisper")

# Transcribe speech across any supported language
audio, sr = sf.read("output/tts_sample_hi.wav")
result = stt.transcribe(audio, language="hi")
print(f"Recognized Text: {result.text} (Confidence: {result.confidence})")
```

### 4. Running Benchmarks & Performance Evaluation
Execute the repeatable benchmark matrix across channel models, SNRs, packet loss, corruption, and priority levels:
```bash
python scripts/evaluate_performance.py
```
Outputs:
- Detailed CSV records: `output/performance_evaluation_results.csv`
- Executive summary report: `output/performance_evaluation_summary.md`

### 5. Running Unit & Integration Tests
Execute the multilingual speech unit tests (including all 7 MMS models, Whisper STT, and registry status matrix):
```bash
python -m unittest tests/test_speech.py
```

Execute the transceiver, channel, network transport, and evaluation test suites:
```bash
python -m unittest tests/test_channel.py tests/test_network_transport.py tests/test_evaluation.py
```

---

## Current Status and Verification Summary

- **Speech Synthesis (TTS)**: Fully verified genuine offline neural TTS operational across all 7 target languages (Telugu, Hindi, Tamil, Bengali, Marathi, Kannada, and Malayalam). Pre-existing Hindi and Telugu Edge-TTS neural voice paths remain fully preserved.
- **Speech Recognition (STT)**: Log-Mel spectrogram extraction (80 bins) and VAD fully operational across all target languages.
- **Channel Simulation**: Fully implemented for AWGN and Rayleigh flat fading with power constraints and zero-forcing equalization.
- **Neural Transceiver**: Lightweight encoder and decoder implemented in PyTorch and verified on CPU.
- **Radio & Framing**: Frame synchronization (0xAA55AA55), packetization, CRC-16 verification, and simulated software-loopback SDR are fully functional and tested.
- **Physical SDR Hardware**: Physical SDR hardware drivers (RTL-SDR, HackRF, USRP) remain isolated in `HardwareSDRStub` pending dedicated bench testing.
