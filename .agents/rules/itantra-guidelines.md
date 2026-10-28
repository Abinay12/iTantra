# iTantra Development Guidelines

## Project Overview
iTantra is an Indian Multilingual Text-to-Speech (TTS) and Speech-to-Text (STT) Communication System integrated with a neural wireless transceiver and simulated/software-defined radio (SDR) interfaces.

## Architectural Boundaries & Modularity
The codebase maintains strict separation of concerns across three core pillars:
1. **Speech Processing (`src/speech/`)**:
   - Handles TTS, STT, multilingual tokenization, phoneme mapping, and acoustic feature extraction.
   - Decoupled from transmission channels and radio framing.
2. **Neural Transceiver (`src/transceiver/`)**:
   - Implements PyTorch-based neural joint source-channel coding (JSCC), semantic/channel encoders, and decoders.
   - Implements channel simulations (AWGN, Rayleigh fading) with average power constraints.
   - Operates purely on numerical/tensor representations without direct radio hardware dependencies.
3. **Radio & Network (`src/radio/`)**:
   - Manages packet framing, synchronization words, header metadata, CRC error detection, and SDR interfaces.
   - Keeps simulated loopback SDR separate from future physical hardware drivers (e.g., USRP, HackRF, RTL-SDR).

## Multilingual Support for Indian Languages
- Planned primary languages:
  - Telugu (`te`)
  - Hindi (`hi`)
  - Tamil (`ta`)
  - Bengali (`bn`)
  - Marathi (`mr`)
- Maintain Unicode and language code standardization across all interfaces.
- Design engines to support pluggable acoustic and language backends (e.g., IndicTTS, IndicWav2Vec, Whisper, or lightweight local models).

## CPU & Resource Constraints
- All core models, signal processing algorithms, and channel simulations must function seamlessly on standard CPU hardware.
- Avoid mandatory GPU-only operations or hardcoded `.cuda()` calls; always use device-agnostic code (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- Keep default neural transceiver models lightweight (compact feedforward or lightweight 1D-CNN layers) for rapid prototyping and simulation.

## Engineering Standards
- **Integrity**: Do not claim any feature is operational unless it is fully implemented and verified with tests.
- **Hardware Isolation**: Physical SDR interfaces must remain stubs until actual hardware integration and bench testing take place.
- **Testing**: All components must have corresponding unit tests under `tests/` executable via standard `unittest` or `pytest`.
- **Typing & Documentation**: Use Python type annotations (`typing`) and clear docstrings for public classes and methods.
