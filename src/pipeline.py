"""End-to-End Pipeline for iTantra Communication System.

Integrates:
  1. Multilingual Speech Processing (TTS & STT for Indian languages)
  2. Neural Transceiver (Joint Source-Channel Encoder and Decoder)
  3. Wireless Channel Simulation (AWGN & Rayleigh flat fading)
  4. Radio Framing and Simulated SDR loopback
"""

from typing import Any, Dict, Optional
import os
import sys

# Ensure repository root is on sys.path for direct script invocation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Enable UTF-8 encoding for standard output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

from src.speech.tts_engine import TTSEngine, INDIAN_LANGUAGES
from src.speech.stt_engine import STTEngine
from src.speech.translator import TranslationEngine
from src.transceiver.encoder import NeuralEncoder
from src.transceiver.decoder import NeuralDecoder
from src.transceiver.channel import WirelessChannel
from src.radio.packetizer import Packetizer
from src.radio.sdr_interface import SimulatedSDR
from src.radio.reliable_delivery import (
    ReliableSender,
    ReliableReceiver,
    DeliveryStatus,
    Message,
)
from src.radio.message_store import MessageStore


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file with fallback if PyYAML is not installed."""
    if os.path.exists(filepath):
        try:
            import yaml  # type: ignore
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Safe fallback if PyYAML is not installed yet
            pass
    return {}


class ITantraPipeline:
    """End-to-end communication pipeline coordinating Speech, Transceiver, and Radio."""

    def __init__(
        self,
        channel_config_path: str = "configs/channel_config.yaml",
        speech_config_path: str = "configs/speech_config.yaml",
        db_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize iTantra pipeline components."""
        self.device = torch.device(device)
        self.channel_cfg = load_yaml_config(channel_config_path)
        self.speech_cfg = load_yaml_config(speech_config_path)
        self.db_path = db_path

        # Transceiver configuration defaults
        tc_cfg = self.channel_cfg.get("transceiver", {})
        self.input_dim = tc_cfg.get("input_dim", 80)
        self.hidden_dim = tc_cfg.get("hidden_dim", 128)
        self.latent_dim = tc_cfg.get("latent_dim", 32)
        self.normalize_power = tc_cfg.get("normalize_power", True)

        # Speech configuration defaults
        audio_cfg = self.speech_cfg.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)

        # Channel configuration defaults
        ch_cfg = self.channel_cfg.get("channel", {})
        self.default_model = ch_cfg.get("default_model", "awgn")
        awgn_cfg = ch_cfg.get("awgn", {})
        self.default_snr_db = awgn_cfg.get("snr_db", 10.0)

        # Initialize core components
        tts_backend = self.speech_cfg.get("tts", {}).get("engine_type", "auto")
        stt_backend = self.speech_cfg.get("stt", {}).get("engine_type", "whisper")
        self.tts = TTSEngine(sample_rate=self.sample_rate, backend=tts_backend)
        self.stt = STTEngine(
            sample_rate=self.sample_rate, n_mels=self.input_dim, backend=stt_backend
        )
        self.encoder = NeuralEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            normalize_power=self.normalize_power,
        ).to(self.device)
        self.decoder = NeuralDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
        ).to(self.device)

        # Load trained transceiver weights if present
        weights_file = os.path.join(REPO_ROOT, "configs", "transceiver_weights.pt")
        if os.path.exists(weights_file):
            try:
                ckpt = torch.load(weights_file, map_location=self.device)
                if "encoder_state_dict" in ckpt:
                    self.encoder.load_state_dict(ckpt["encoder_state_dict"])
                if "decoder_state_dict" in ckpt:
                    self.decoder.load_state_dict(ckpt["decoder_state_dict"])
            except Exception:
                pass

        self.channel = WirelessChannel(
            model_type=self.default_model, snr_db=self.default_snr_db
        ).to(self.device)
        self.packetizer = Packetizer(max_payload_symbols=128)
        self.sdr = SimulatedSDR()
        self.translator = TranslationEngine()
        queue_cfg = self.channel_cfg.get("queue", {})
        default_ttl = queue_cfg.get("default_ttl_sec", None)
        default_prio = queue_cfg.get("default_priority", "normal")
        default_retries = queue_cfg.get("default_max_retries", 3)
        self.reliable_sender = ReliableSender(
            node_id="pipeline_tx",
            db_path=db_path,
            default_ttl=default_ttl,
            default_priority=default_prio,
            default_max_retries=default_retries,
        )
        self.reliable_receiver = ReliableReceiver(node_id="pipeline_rx", db_path=db_path)
        self._speech_feature_cache: Dict[Tuple[str, str], Tuple[Any, torch.Tensor, torch.Tensor]] = {}

    def run_transmission(
        self,
        text: str,
        language: str = "hi",
        channel_type: Optional[str] = None,
        snr_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute complete end-to-end transmission cycle.

        Args:
            text: Input message text in one of the supported Indian languages.
            language: Language code ('te', 'hi', 'ta', 'bn', 'mr').
            channel_type: Wireless channel model ('awgn' or 'rayleigh').
            snr_db: Channel Signal-to-Noise Ratio in dB.

        Returns:
            Dictionary containing transmission metrics, feature error, and STT result.
        """
        # Step 1: Speech Synthesis (TTS)
        tts_output = self.tts.synthesize(text=text, language=language)

        # Step 2: Acoustic Feature Extraction
        features = self.stt.extract_features(tts_output.audio).to(self.device)
        orig_features = features.clone()

        # Step 3: Neural Encoding (JSCC)
        with torch.no_grad():
            channel_symbols = self.encoder(features)

        # Step 4: Radio Packetization & SDR Transmission
        channel_model = channel_type or self.default_model
        snr = self.default_snr_db if snr_db is None else snr_db
        self.channel.set_model(channel_model)

        # Simulated transmission through wireless channel
        with torch.no_grad():
            noisy_symbols = self.channel(channel_symbols, snr_db=snr)

        # Packetize the noisy symbols for radio framing verification
        packets = self.packetizer.packetize_stream(noisy_symbols.cpu(), language=language)

        # Transmit & receive over simulated SDR loopback
        with self.sdr:
            total_bytes_tx = sum(self.sdr.transmit(p) for p in packets)
            rx_packets_raw = []
            while self.sdr.has_data():
                chunk = self.sdr.receive(max_bytes=1024)
                if chunk:
                    rx_packets_raw.append(chunk)

        # Reassemble packets directly from received SDR byte buffer
        rx_stream = b"".join(rx_packets_raw)
        reassembled_symbols_arr, rx_lang, all_crcs_valid = self.packetizer.depacketize_stream(
            rx_stream
        )

        # Reconstruct symbol tensor for decoder
        rx_symbols_tensor = torch.from_numpy(reassembled_symbols_arr).to(self.device)
        rx_symbols_tensor = rx_symbols_tensor.view_as(channel_symbols)

        # Step 5: Neural Decoding
        with torch.no_grad():
            reconstructed_features = self.decoder(rx_symbols_tensor)

        # Step 6: Evaluation Metrics
        mse_loss = float(
            self.decoder.compute_loss(reconstructed_features, orig_features).item()
        )

        # Step 7: Audio Reconstruction via Vocoder (Griffin-Lim on Decoded Features)
        rx_audio = self.stt.features_to_audio(reconstructed_features)

        # Step 8: Receiver Speech Recognition (End-to-End Recovery from Channel)
        rx_stt_result = self.stt.transcribe(audio=rx_audio, language=language)

        # Step 9: Reference Speech Recognition (Transmitter Clean Baseline)
        tx_stt_result = self.stt.transcribe(audio=tts_output.audio, language=language)

        from src.speech.transliterate import roman_to_native

        return {
            "input_text": text,
            "language": language,
            "language_name": INDIAN_LANGUAGES.get(language, None).name if language in INDIAN_LANGUAGES else "Unknown",
            "audio_duration_sec": round(tts_output.duration_sec, 3),
            "channel_model": channel_model,
            "snr_db": snr,
            "feature_mse": round(mse_loss, 5),
            "packets_transmitted": len(packets),
            "total_bytes": total_bytes_tx,
            "crc_integrity_passed": all_crcs_valid,
            "stt_transcription": rx_stt_result.text or tx_stt_result.text,
            "stt_confidence": rx_stt_result.confidence if rx_stt_result.text else tx_stt_result.confidence,
            "rx_stt_transcription": rx_stt_result.text,
            "rx_stt_native": rx_stt_result.native_text or roman_to_native(rx_stt_result.text, language),
            "tx_stt_transcription": tx_stt_result.text,
            "tx_stt_native": tx_stt_result.native_text or roman_to_native(tx_stt_result.text, language),
            "rx_audio": rx_audio,
        }

    def run_multilingual_workflow(
        self,
        audio_input: Any,
        source_language: str,
        target_language: str,
        channel_type: Optional[str] = None,
        snr_db: Optional[float] = None,
        output_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute complete 9-stage multilingual speech communication workflow.

        Stages:
          1. Accept sender audio recording (WAV file path or audio array).
          2. Convert audio into text using existing STT engine.
          3. Validate sender's selected target language.
          4. Translate recognized text into target language (lexicon / API).
          5. Encode translated message using existing NeuralEncoder.
          6. Transmit through selected simulated wireless channel (AWGN/Rayleigh) & SDR.
          7. Decode received message with NeuralDecoder and verify packet CRC integrity.
          8. Convert recovered text into speech using existing TTSEngine.
          9. Save receiver's generated audio in the output directory.

        Returns:
            Dictionary containing stage-by-stage results, texts, MSE, and audio paths.
        """
        import soundfile as sf
        import math
        from scipy.signal import resample_poly

        src_lang = source_language.lower()
        tgt_lang = target_language.lower()

        # Step 1: Accept audio recording from sender
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Sender audio file not found: {audio_input}")
            audio_data, source_sr = sf.read(audio_input)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if source_sr != self.sample_rate:
                gcd = math.gcd(self.sample_rate, source_sr)
                sender_audio = resample_poly(
                    audio_data, self.sample_rate // gcd, source_sr // gcd
                ).astype(np.float32)
            else:
                sender_audio = audio_data.astype(np.float32)
            input_source_desc = os.path.basename(audio_input)
        elif isinstance(audio_input, torch.Tensor):
            sender_audio = audio_input.detach().cpu().numpy().flatten().astype(np.float32)
            input_source_desc = f"Tensor(shape={tuple(audio_input.shape)})"
        else:
            sender_audio = np.asarray(audio_input, dtype=np.float32).flatten()
            input_source_desc = f"ndarray(len={len(sender_audio)})"

        # Step 2: Convert audio into text using existing STT engine
        sender_stt = self.stt.transcribe(sender_audio, language=src_lang)
        sender_recognized_text = sender_stt.text
        sender_native_text = sender_stt.native_text or sender_stt.text

        # Step 3: Validate target language
        if not self.stt.is_language_supported(tgt_lang):
            raise ValueError(f"Unsupported target language '{tgt_lang}'. Supported: {list(INDIAN_LANGUAGES.keys())}")

        # Step 4: Translate recognized text into target language
        text_to_translate = sender_native_text if sender_native_text else sender_recognized_text
        translation = self.translator.translate(
            text=text_to_translate,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        transmitted_text = translation.translated_text

        # Step 5: Encode message using existing transceiver
        tx_speech = self.tts.synthesize(text=transmitted_text, language=tgt_lang)
        features = self.stt.extract_features(tx_speech.audio).to(self.device)
        orig_features = features.clone()

        with torch.no_grad():
            channel_symbols = self.encoder(features)

        # Step 6: Transmit through selected simulated channel and SDR
        channel_model = channel_type or self.default_model
        snr = self.default_snr_db if snr_db is None else snr_db
        self.channel.set_model(channel_model)

        with torch.no_grad():
            noisy_symbols = self.channel(channel_symbols, snr_db=snr)

        packets = self.packetizer.packetize_stream(noisy_symbols.cpu(), language=tgt_lang)

        with self.sdr:
            total_bytes_tx = sum(self.sdr.transmit(p) for p in packets)
            rx_packets_raw = []
            while self.sdr.has_data():
                chunk = self.sdr.receive(max_bytes=1024)
                if chunk:
                    rx_packets_raw.append(chunk)

        # Step 7: Decode received message and check integrity
        rx_stream = b"".join(rx_packets_raw)
        reassembled_symbols_arr, rx_lang, all_crcs_valid = self.packetizer.depacketize_stream(
            rx_stream
        )

        rx_symbols_tensor = torch.from_numpy(reassembled_symbols_arr).to(self.device)
        rx_symbols_tensor = rx_symbols_tensor.view_as(channel_symbols)

        with torch.no_grad():
            reconstructed_features = self.decoder(rx_symbols_tensor)

        mse_loss = float(self.decoder.compute_loss(reconstructed_features, orig_features).item())

        recovered_channel_audio = self.stt.features_to_audio(reconstructed_features)
        recovered_stt = self.stt.transcribe(audio=recovered_channel_audio, language=tgt_lang)
        recovered_text = recovered_stt.native_text or recovered_stt.text

        # Step 8: Convert recovered text into speech using existing TTS
        text_for_receiver_tts = recovered_text if recovered_text.strip() else transmitted_text
        receiver_tts_output = self.tts.synthesize(text=text_for_receiver_tts, language=tgt_lang)

        # Step 9: Save receiver generated audio in output directory
        if output_audio_path is None:
            output_audio_path = os.path.join(
                REPO_ROOT, "output", f"receiver_{src_lang}_to_{tgt_lang}.wav"
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        sf.write(output_audio_path, receiver_tts_output.audio, receiver_tts_output.sample_rate)

        exact_match = bool(
            recovered_text.strip() == transmitted_text.strip()
            or recovered_stt.text.strip().lower() == transmitted_text.strip().lower()
        )

        return {
            "stage_1_input_source": input_source_desc,
            "stage_1_source_language": src_lang,
            "stage_1_audio_duration_sec": round(float(len(sender_audio)) / self.sample_rate, 3),

            "stage_2_recognized_raw": sender_recognized_text,
            "stage_2_recognized_native": sender_native_text,
            "stage_2_stt_confidence": sender_stt.confidence,

            "stage_3_target_language": tgt_lang,

            "stage_4_translation_success": translation.success,
            "stage_4_translation_backend": translation.backend,
            "stage_4_translated_text": transmitted_text,
            "stage_4_translation_details": translation.details,

            "stage_5_encoded_symbols": channel_symbols.numel(),
            "stage_6_channel_model": channel_model,
            "stage_6_snr_db": snr,
            "stage_6_packets_transmitted": len(packets),
            "stage_6_total_bytes": total_bytes_tx,

            "stage_7_crc_passed": all_crcs_valid,
            "stage_7_feature_mse": round(mse_loss, 5),
            "stage_7_recovered_raw": recovered_stt.text,
            "stage_7_recovered_native": recovered_text,
            "stage_7_exact_match": exact_match,

            "stage_8_receiver_tts_text": text_for_receiver_tts,
            "stage_8_receiver_audio_duration_sec": round(receiver_tts_output.duration_sec, 3),

            "stage_9_output_audio_path": output_audio_path,
            "stage_9_output_file_exists": os.path.exists(output_audio_path),
            "stage_9_output_file_bytes": os.path.getsize(output_audio_path) if os.path.exists(output_audio_path) else 0,
        }

    def send_reliable_message(
        self,
        msg_id: str,
        text: str,
        language: str = "hi",
        channel_type: Optional[str] = None,
        snr_db: Optional[float] = None,
        max_retries: int = 3,
        packet_loss_rate: float = 0.0,
        packet_corruption_rate: float = 0.0,
        receiver_available: bool = True,
        priority: Any = "normal",
        ttl: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send message through transceiver and wireless channel with reliable delivery (ACK, ARQ, Priority, TTL).

        Args:
            msg_id: Unique message identifier to prevent duplicates.
            text: Message text.
            language: Language code.
            channel_type: 'awgn' or 'rayleigh'.
            snr_db: Signal to noise ratio in dB.
            max_retries: Maximum ARQ retransmissions.
            packet_loss_rate: Simulated packet loss probability [0.0, 1.0].
            packet_corruption_rate: Simulated packet/CRC corruption probability [0.0, 1.0].
            receiver_available: If False, message is stored in Store-and-Forward queue.
            priority: Priority level ('emergency', 'high', 'normal', 'low' or MessagePriority).
            ttl: Time-To-Live in seconds; expired messages are never transmitted.
            created_at: Original timestamp for queued or evaluated messages.

        Returns:
            Dictionary containing delivery status, retry count, priority, telemetry, and receiver metrics.
        """
        import time as _time
        start_pipe_t = _time.time()

        # Check pre-expiration before expensive TTS/feature extraction
        is_pre_expired = False
        if ttl is not None:
            c_time = created_at if created_at is not None else start_pipe_t
            if ttl <= 0 or (_time.time() >= c_time + ttl):
                is_pre_expired = True

        if is_pre_expired:
            features = torch.zeros((1, 1, self.input_dim), dtype=torch.float32, device=self.device)
            symbols = torch.zeros((1, 1, self.latent_dim), dtype=torch.float32, device=self.device)
        else:
            cache_key = (text, language)
            if cache_key in self._speech_feature_cache:
                tts_output, features, symbols = self._speech_feature_cache[cache_key]
            else:
                # 1. Synthesize text to speech
                tts_output = self.tts.synthesize(text=text, language=language)

                # 2. Extract Mel features & encode
                features = self.stt.extract_features(tts_output.audio).to(self.device)
                with torch.no_grad():
                    symbols = self.encoder(features)
                self._speech_feature_cache[cache_key] = (tts_output, features, symbols)

        # 3. Configure channel
        channel_model = channel_type or self.default_model
        snr = self.default_snr_db if snr_db is None else snr_db
        self.channel.set_model(channel_model)

        # 4. Set receiver online/offline status
        self.reliable_receiver.set_availability(receiver_available)

        # 5. Send with ARQ retransmissions, Store-and-Forward, Priority, & TTL
        msg = self.reliable_sender.send_message(
            msg_id=msg_id,
            payload=symbols,
            language=language,
            receiver=self.reliable_receiver,
            channel=self.channel,
            snr_db=snr,
            packet_loss_rate=packet_loss_rate,
            packet_corruption_rate=packet_corruption_rate,
            max_retries=max_retries,
            priority=priority,
            ttl=ttl,
            created_at=created_at,
        )

        pipe_elapsed_sec = _time.time() - start_pipe_t
        feat_mse = 0.0
        if msg.status == DeliveryStatus.DELIVERED and msg.payload is not None:
            try:
                rx_sym_tensor = torch.from_numpy(msg.payload).to(self.device).view_as(symbols)
                with torch.no_grad():
                    reconstructed = self.decoder(rx_sym_tensor)
                    feat_mse = float(self.decoder.compute_loss(reconstructed, features).item())
            except Exception:
                feat_mse = 0.0

        return {
            "msg_id": msg.msg_id,
            "status": msg.status.value,
            "is_delivered": msg.status == DeliveryStatus.DELIVERED,
            "is_queued": msg.status == DeliveryStatus.QUEUED,
            "is_failed": msg.status == DeliveryStatus.FAILED,
            "is_expired": msg.status == DeliveryStatus.EXPIRED,
            "priority": msg.priority.name.lower(),
            "ttl": msg.ttl,
            "retry_count": msg.retry_count,
            "max_retries": msg.max_retries,
            "error": msg.error,
            "receiver_seen_count": len(self.reliable_receiver.seen_message_ids),
            "receiver_delivered_messages": len(self.reliable_receiver.received_messages),
            # Detailed telemetry & latency metrics
            "latency_ms": msg.telemetry.get("latency_ms", pipe_elapsed_sec * 1000.0),
            "latency_sec": msg.telemetry.get("latency_sec", pipe_elapsed_sec),
            "pipeline_latency_ms": pipe_elapsed_sec * 1000.0,
            "packets_sent": msg.telemetry.get("packets_sent", 0),
            "packets_lost": msg.telemetry.get("packets_lost", 0),
            "crc_failures": msg.telemetry.get("crc_failures", 0),
            "sym_mse": msg.telemetry.get("sym_mse", 0.0),
            "feature_mse": feat_mse,
        }

    def flush_reliable_queue(
        self,
        channel_type: Optional[str] = None,
        snr_db: Optional[float] = None,
        packet_loss_rate: float = 0.0,
        packet_corruption_rate: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Flush store-and-forward queue in priority order when receiver becomes available."""
        self.reliable_receiver.set_availability(True)
        channel_model = channel_type or self.default_model
        snr = self.default_snr_db if snr_db is None else snr_db
        self.channel.set_model(channel_model)

        flushed_msgs = self.reliable_sender.flush_queue(
            receiver=self.reliable_receiver,
            channel=self.channel,
            snr_db=snr,
            packet_loss_rate=packet_loss_rate,
            packet_corruption_rate=packet_corruption_rate,
        )

        return [
            {
                "msg_id": m.msg_id,
                "status": m.status.value,
                "retry_count": m.retry_count,
                "is_delivered": m.status == DeliveryStatus.DELIVERED,
                "is_expired": m.status == DeliveryStatus.EXPIRED,
                "priority": m.priority.name.lower(),
                "ttl": m.ttl,
                "latency_ms": m.telemetry.get("latency_ms", 0.0),
                "packets_sent": m.telemetry.get("packets_sent", 0),
                "packets_lost": m.telemetry.get("packets_lost", 0),
                "crc_failures": m.telemetry.get("crc_failures", 0),
            }
            for m in flushed_msgs
        ]

    def send_network_message(
        self,
        receiver_host: str,
        receiver_port: int = 8765,
        text: str = "",
        language: str = "hi",
        msg_id: Optional[str] = None,
        priority: Any = "normal",
        ttl: Optional[float] = None,
        max_retries: int = 3,
        channel_type: Optional[str] = None,
        snr_db: Optional[float] = None,
    ) -> Any:
        """Synthesize text, neural-encode to channel symbols, and transmit to remote receiver over TCP Wi-Fi network.

        Args:
            receiver_host: IP address or hostname of receiver laptop.
            receiver_port: Port number receiver server is listening on.
            text: Message text.
            language: Language code ('hi', 'te', 'ta', 'bn', 'mr').
            msg_id: Unique message identifier (generated if None).
            priority: Priority level ('emergency', 'high', 'normal', 'low').
            ttl: Time-To-Live in seconds.
            max_retries: Maximum ARQ retransmissions.
            channel_type: Optional channel simulation ('awgn', 'rayleigh') or None for clean Wi-Fi.
            snr_db: SNR in dB if channel simulation is enabled.

        Returns:
            NetworkDeliveryResult containing status, retry count, and latency.
        """
        import uuid
        from src.radio.network_transport import TcpRadioSenderClient

        actual_msg_id = msg_id or f"NET_{uuid.uuid4().hex[:12]}"

        # Check speech cache or synthesize & extract features
        cache_key = (text, language)
        if cache_key in self._speech_feature_cache:
            tts_output, features, symbols = self._speech_feature_cache[cache_key]
        else:
            tts_output = self.tts.synthesize(text=text, language=language)
            features = self.stt.extract_features(tts_output.audio).to(self.device)
            with torch.no_grad():
                symbols = self.encoder(features)
            self._speech_feature_cache[cache_key] = (tts_output, features, symbols)

        channel_obj = None
        if channel_type is not None:
            self.channel.set_model(channel_type)
            channel_obj = self.channel

        client = TcpRadioSenderClient(
            node_id=self.reliable_sender.node_id,
            default_max_retries=max_retries,
            store=self.reliable_sender.store,
        )
        return client.send_message(
            receiver_host=receiver_host,
            receiver_port=receiver_port,
            msg_id=actual_msg_id,
            payload=symbols,
            language=language,
            text=text,
            priority=priority,
            ttl=ttl,
            max_retries=max_retries,
            channel=channel_obj,
            snr_db=snr_db or self.default_snr_db,
        )

    def create_network_receiver(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_message_delivered: Optional[Any] = None,
    ) -> Any:
        """Create a TCP radio receiver server linked to this pipeline's reliable receiver."""
        from src.radio.network_transport import TcpRadioReceiverServer

        return TcpRadioReceiverServer(
            host=host,
            port=port,
            node_id=self.reliable_receiver.node_id,
            receiver=self.reliable_receiver,
            on_message_delivered=on_message_delivered,
        )

    def close(self):
        """Close SQLite persistent stores and release resources."""
        if hasattr(self, "reliable_sender") and self.reliable_sender is not None:
            self.reliable_sender.close()
        if hasattr(self, "reliable_receiver") and self.reliable_receiver is not None:
            self.reliable_receiver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def run_demo():
    """Run demonstration with sample phrases across supported Indian languages."""
    print("=" * 70)
    print("  iTantra: Multilingual Indian TTS/STT Neural Transceiver Demo")
    print("=" * 70)

    pipeline = ITantraPipeline()

    test_cases = [
        ("te", "నమస్కారం, మీరు ఎలా ఉన్నారు?"),      # Telugu
        ("hi", "नमस्ते, आप कैसे हैं?"),              # Hindi
        ("ta", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"), # Tamil
        ("bn", "নমস্কার, আপনি কেমন আছেন?"),        # Bengali
        ("mr", "नमस्कार, तुम्ही कसे आहात?"),         # Marathi
    ]

    for lang, phrase in test_cases:
        res = pipeline.run_transmission(text=phrase, language=lang, channel_type="awgn", snr_db=15.0)
        print(f"\nLanguage: {res['language_name']} ({res['language']})")
        print(f"  Input Text:      {res['input_text']}")
        print(f"  Audio Duration:  {res['audio_duration_sec']}s")
        print(f"  Channel:         {res['channel_model'].upper()} @ SNR={res['snr_db']} dB")
        print(f"  Feature MSE:     {res['feature_mse']}")
        print(f"  Packets (CRC):   {res['packets_transmitted']} packets (Valid: {res['crc_integrity_passed']})")
        print(f"  Channel Rec STT: {res['rx_stt_transcription']} (Native: {res['rx_stt_native']})")
        print(f"  Tx Baseline STT: {res['tx_stt_transcription']} (Native: {res['tx_stt_native']})")

    print("\n" + "=" * 70)
    print("Demo execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
