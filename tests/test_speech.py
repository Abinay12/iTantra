"""Unit tests for Indian multilingual speech processing and radio packetization."""

import unittest
import numpy as np
import torch

from src.speech.tts_engine import TTSEngine, INDIAN_LANGUAGES
from src.speech.stt_engine import STTEngine
from src.speech.translator import TranslationEngine
from src.radio.packetizer import Packetizer, PREAMBLE_SYNC
from src.radio.sdr_interface import SimulatedSDR, HardwareSDRStub


class TestIndianLanguages(unittest.TestCase):
    """Test planned Indian language support registry."""

    def test_all_eight_languages_present(self):
        target_languages = ["te", "hi", "ta", "bn", "mr", "kn", "ml", "gu"]
        for lang in target_languages:
            self.assertIn(lang, INDIAN_LANGUAGES)
            info = INDIAN_LANGUAGES[lang]
            self.assertTrue(len(info.name) > 0)
            self.assertTrue(len(info.script) > 0)
            self.assertTrue(len(info.locale) > 0)
            self.assertIsInstance(info.unicode_range, tuple)
            self.assertEqual(len(info.unicode_range), 2)

    def test_independent_stt_tts_availability_tracking(self):
        """All seven target languages are fully supported and verified; Gujarati is out of project scope."""
        # Hindi, Telugu, Tamil, Bengali, Marathi, Kannada & Malayalam: fully supported and verified
        for lang in ["hi", "te", "ta", "bn", "mr", "kn", "ml"]:
            info = INDIAN_LANGUAGES[lang]
            self.assertTrue(info.tts_available)
            self.assertTrue(info.stt_available)
            self.assertTrue(info.is_fully_supported)
            self.assertTrue(info.in_scope)
            self.assertTrue(info.tts_verified)
            self.assertEqual(info.status, "fully_supported")
            self.assertEqual(info.verification_status, "verified")

        # Gujarati is out of project scope
        info = INDIAN_LANGUAGES["gu"]
        self.assertFalse(info.in_scope)
        self.assertFalse(info.tts_available)
        self.assertFalse(info.is_fully_supported)
        self.assertFalse(info.tts_verified)
        self.assertEqual(info.status, "out_of_scope")
        self.assertEqual(info.verification_status, "out_of_scope")

    def test_engine_availability_query_methods(self):
        tts = TTSEngine(backend="lightweight")
        self.assertTrue(tts.is_language_supported("hi"))
        self.assertTrue(tts.is_language_supported("kn"))
        self.assertTrue(tts.is_in_scope("hi"))
        self.assertTrue(tts.is_in_scope("kn"))
        self.assertFalse(tts.is_in_scope("gu"))
        self.assertTrue(tts.is_tts_supported("hi"))
        self.assertTrue(tts.is_tts_supported("te"))
        self.assertTrue(tts.is_tts_supported("ta"))
        self.assertTrue(tts.is_tts_supported("bn"))
        self.assertTrue(tts.is_tts_supported("mr"))
        self.assertTrue(tts.is_tts_supported("kn"))
        self.assertTrue(tts.is_tts_supported("ml"))
        self.assertFalse(tts.is_tts_supported("gu"))
        self.assertTrue(tts.is_stt_supported("gu"))
        self.assertTrue(tts.is_fully_supported("hi"))
        self.assertTrue(tts.is_fully_supported("te"))
        self.assertTrue(tts.is_fully_supported("ta"))
        self.assertTrue(tts.is_fully_supported("bn"))
        self.assertTrue(tts.is_fully_supported("mr"))
        self.assertTrue(tts.is_fully_supported("kn"))
        self.assertTrue(tts.is_fully_supported("ml"))
        self.assertFalse(tts.is_fully_supported("gu"))

        supported_list = tts.get_supported_languages()
        self.assertEqual(len(supported_list), 8)
        hi_entry = next(item for item in supported_list if item["code"] == "hi")
        self.assertTrue(hi_entry["is_fully_supported"])
        self.assertTrue(hi_entry["tts_verified"])
        self.assertEqual(hi_entry["verification_status"], "verified")
        ta_entry = next(item for item in supported_list if item["code"] == "ta")
        self.assertTrue(ta_entry["is_fully_supported"])
        bn_entry = next(item for item in supported_list if item["code"] == "bn")
        self.assertTrue(bn_entry["is_fully_supported"])
        mr_entry = next(item for item in supported_list if item["code"] == "mr")
        self.assertTrue(mr_entry["is_fully_supported"])
        kn_entry = next(item for item in supported_list if item["code"] == "kn")
        self.assertTrue(kn_entry["is_fully_supported"])
        ml_entry = next(item for item in supported_list if item["code"] == "ml")
        self.assertTrue(ml_entry["is_fully_supported"])
        self.assertTrue(ml_entry["tts_available"])
        self.assertTrue(ml_entry["stt_available"])
        gu_entry = next(item for item in supported_list if item["code"] == "gu")
        self.assertFalse(gu_entry["is_fully_supported"])
        self.assertFalse(gu_entry["in_scope"])
        self.assertTrue(gu_entry["stt_available"])
        self.assertFalse(gu_entry["tts_available"])
        self.assertEqual(gu_entry["status"], "out_of_scope")
        self.assertEqual(gu_entry["verification_status"], "out_of_scope")

        # Test filtering by in_scope
        in_scope_list = tts.get_supported_languages(in_scope_only=True)
        self.assertEqual(len(in_scope_list), 7)
        self.assertTrue(all(item["in_scope"] for item in in_scope_list))

        # Test status matrix
        matrix = tts.get_language_status_matrix()
        self.assertEqual(len(matrix), 8)
        for item in matrix:
            if item["code"] in ["hi", "te", "ta", "bn", "mr", "kn", "ml"]:
                self.assertTrue(item["configured"])
                self.assertTrue(item["installed"])
                self.assertTrue(item["tts_available"])
                self.assertTrue(item["tts_verified"])
                self.assertEqual(item["status"], "fully_supported")
                self.assertEqual(item["verification_status"], "verified")
            elif item["code"] == "gu":
                self.assertFalse(item["in_scope"])
                self.assertFalse(item["installed"])
                self.assertFalse(item["tts_available"])
                self.assertEqual(item["status"], "out_of_scope")


class TestTTSEngine(unittest.TestCase):
    """Test Text-to-Speech engine."""

    def setUp(self):
        self.tts = TTSEngine(sample_rate=16000, backend="lightweight")

    def test_synthesize_telugu(self):
        res = self.tts.synthesize("నమస్కారం", language="te")
        self.assertEqual(res.language, "te")
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 0)
        self.assertTrue(np.all(np.abs(res.audio) <= 1.0))

    def test_synthesize_hindi(self):
        res = self.tts.synthesize("नमस्ते", language="hi")
        self.assertEqual(res.language, "hi")
        self.assertGreater(res.duration_sec, 0.0)

    def test_synthesize_all_seven_target_languages(self):
        phrases = {
            "te": "నమస్కారం",
            "hi": "नमस्ते",
            "ta": "வணக்கம்",
            "bn": "নমস্কার",
            "mr": "नमस्कार",
            "kn": "ನಮಸ್ಕಾರ",
            "ml": "നമസ്കാരം",
        }
        for lang, text in phrases.items():
            res = self.tts.synthesize(text, language=lang)
            self.assertIsInstance(res.audio, np.ndarray)
            self.assertGreater(len(res.audio), 500)
            self.assertEqual(res.language, lang)

    def test_out_of_scope_language_error(self):
        """Synthesizing an out-of-scope language raises ValueError with informative message."""
        with self.assertRaises(ValueError) as ctx:
            self.tts.synthesize("નમસ્તે", language="gu")
        self.assertIn("out of project scope", str(ctx.exception))

    def test_invalid_language(self):
        with self.assertRaises(ValueError):
            self.tts.synthesize("Hello", language="fr")

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            self.tts.synthesize("   ", language="hi")

    def test_missing_mms_model_handling(self):
        """Calling MMS synthesis on an uncached model raises FileNotFoundError without pretending it worked."""
        tts_mms = TTSEngine(backend="mms")
        self.assertFalse(tts_mms.is_mms_model_cached("gu"))
        with self.assertRaises(FileNotFoundError):
            tts_mms._synthesize_mms("નમસ્તે", "gu")

    def test_preserve_hindi_telugu_neural_voices(self):
        """Verify pre-existing Edge-TTS neural voice identifiers are preserved."""
        self.assertEqual(INDIAN_LANGUAGES["hi"].neural_voice, "hi-IN-MadhurNeural")
        self.assertEqual(INDIAN_LANGUAGES["te"].neural_voice, "te-IN-MohanNeural")

    def test_hindi_mms_cached_and_synthesizes(self):
        """Hindi MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("hi"))
        res = tts_mms.synthesize("नमस्ते", language="hi")
        self.assertEqual(res.language, "hi")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_telugu_mms_cached_and_synthesizes(self):
        """Telugu MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("te"))
        res = tts_mms.synthesize("నమస్కారం", language="te")
        self.assertEqual(res.language, "te")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_tamil_mms_cached_and_synthesizes(self):
        """Tamil MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("ta"))
        res = tts_mms.synthesize("வணக்கம்", language="ta")
        self.assertEqual(res.language, "ta")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_bengali_mms_cached_and_synthesizes(self):
        """Bengali MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("bn"))
        res = tts_mms.synthesize("নমস্কার", language="bn")
        self.assertEqual(res.language, "bn")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_marathi_mms_cached_and_synthesizes(self):
        """Marathi MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("mr"))
        res = tts_mms.synthesize("नमस्कार", language="mr")
        self.assertEqual(res.language, "mr")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_kannada_mms_cached_and_synthesizes(self):
        """Kannada MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("kn"))
        res = tts_mms.synthesize("ನಮಸ್ಕಾರ", language="kn")
        self.assertEqual(res.language, "kn")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)

    def test_malayalam_mms_cached_and_synthesizes(self):
        """Malayalam MMS-TTS model is cached locally and synthesizes offline neural audio."""
        tts_mms = TTSEngine(backend="mms")
        self.assertTrue(tts_mms.is_mms_model_cached("ml"))
        res = tts_mms.synthesize("നമസ്കാരം", language="ml")
        self.assertEqual(res.language, "ml")
        self.assertGreater(res.duration_sec, 0.5)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(len(res.audio), 8000)
        self.assertGreater(np.max(np.abs(res.audio)), 0.1)


class TestSTTEngine(unittest.TestCase):
    """Test Speech-to-Text engine."""

    def setUp(self):
        self.stt = STTEngine(sample_rate=16000, n_mels=80, backend="lightweight")

    def test_feature_extraction_shape(self):
        # 1 second of audio
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        features = self.stt.extract_features(dummy_audio)

        # Output shape should be (batch_size=1, time_frames, n_mels=80)
        self.assertEqual(features.dim(), 3)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 80)
        self.assertGreater(features.shape[1], 10)

    def test_vad_and_transcribe(self):
        # Active audio
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        res = self.stt.transcribe(audio, language="hi", reference_text="नमस्ते")
        self.assertEqual(res.language, "hi")
        self.assertGreater(res.confidence, 0.5)
        self.assertEqual(res.text, "नमस्ते")

    def test_silence_vad(self):
        silence = np.zeros(8000, dtype=np.float32)
        res = self.stt.transcribe(silence, language="te")
        self.assertEqual(res.text, "")
        self.assertEqual(res.confidence, 0.0)

    def test_features_to_audio_inversion(self):
        # Create audio sinusoid
        t = np.linspace(0, 0.25, 4000, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        # Extract features and invert back to audio
        features = self.stt.extract_features(audio)
        reconstructed_audio = self.stt.features_to_audio(features)

        self.assertIsInstance(reconstructed_audio, np.ndarray)
        self.assertGreater(len(reconstructed_audio), 0)
        self.assertGreater(float(np.sum(reconstructed_audio ** 2)), 0.0)

    def test_all_eight_languages_whisper_map(self):
        from src.speech.stt_engine import WHISPER_LANG_MAP
        target = ["hi", "te", "ta", "bn", "mr", "kn", "ml", "gu"]
        for lang in target:
            self.assertIn(lang, WHISPER_LANG_MAP)
            self.assertEqual(WHISPER_LANG_MAP[lang], lang)
            self.assertTrue(self.stt.is_stt_supported(lang))
            self.assertTrue(self.stt.is_language_supported(lang))


class TestTransliteration(unittest.TestCase):
    """Test rule-based script transliteration for all 8 languages."""

    def test_transliterate_all_languages(self):
        from src.speech.transliterate import roman_to_native

        test_cases = [
            ("hi", "namaste", "नमस्ते"),
            ("te", "namaskaram", "నమస్కారం"),
            ("ta", "vanakkam", "வணக்கம்"),
            ("bn", "namaskar", "নমস্কার"),
            ("mr", "namaskar", "नमस्कार"),
            ("kn", "namaskara", "ನಮಸ್ಕಾರ"),
            ("ml", "namaskaram", "നമസ്കാരം"),
            ("gu", "namaste", "નમસ્તે"),
        ]
        for lang, roman, expected in test_cases:
            native = roman_to_native(roman, lang)
            self.assertEqual(native, expected)

    def test_transliterate_unmapped_words_preserve_input(self):
        from src.speech.transliterate import roman_to_native
        res = roman_to_native("unknownword123", "kn")
        self.assertEqual(res, "unknownword123")


class TestRadioPacketizer(unittest.TestCase):
    """Test radio framing, CRC integrity, and multi-packet streaming."""

    def setUp(self):
        self.packetizer = Packetizer(max_payload_symbols=64)

    def test_packetize_and_depacketize(self):
        symbols = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
        raw_pkt = self.packetizer.packetize(symbols, language="te", packet_id=1, total_packets=3)

        self.assertTrue(raw_pkt.startswith(PREAMBLE_SYNC))

        decoded = self.packetizer.depacketize(raw_pkt)
        self.assertEqual(decoded.language, "te")
        self.assertEqual(decoded.packet_id, 1)
        self.assertEqual(decoded.total_packets, 3)
        self.assertTrue(decoded.crc_valid)
        self.assertTrue(np.allclose(decoded.payload_symbols, symbols, atol=1e-6))

    def test_crc_detects_corruption(self):
        symbols = np.ones(16, dtype=np.float32)
        raw_pkt = bytearray(self.packetizer.packetize(symbols, language="hi"))

        # Corrupt one payload byte
        raw_pkt[15] ^= 0xFF

        decoded = self.packetizer.depacketize(bytes(raw_pkt))
        self.assertFalse(decoded.crc_valid)

    def test_stream_packetization_and_reassembly(self):
        symbols = np.random.randn(150).astype(np.float32)
        packets = self.packetizer.packetize_stream(symbols, language="ta")

        # 150 symbols with chunk size 64 -> 3 packets
        self.assertEqual(len(packets), 3)

        reassembled, lang, all_valid = self.packetizer.depacketize_stream(packets)
        self.assertEqual(lang, "ta")
        self.assertTrue(all_valid)
        self.assertTrue(np.allclose(reassembled, symbols, atol=1e-6))

    def test_stream_contiguous_bytes_reassembly(self):
        symbols = np.random.randn(120).astype(np.float32)
        packets = self.packetizer.packetize_stream(symbols, language="hi")
        contiguous_stream = b"".join(packets)

        reassembled, lang, all_valid = self.packetizer.depacketize_stream(contiguous_stream)
        self.assertEqual(lang, "hi")
        self.assertTrue(all_valid)
        self.assertTrue(np.allclose(reassembled, symbols, atol=1e-6))

    def test_all_eight_languages_framing_and_id_mapping(self):
        """Verify 1-byte header language ID mapping and CRC integrity across all 8 target languages."""
        expected_ids = {
            "te": 1,
            "hi": 2,
            "ta": 3,
            "bn": 4,
            "mr": 5,
            "kn": 6,
            "ml": 7,
            "gu": 8,
        }
        symbols = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        for lang, expected_id in expected_ids.items():
            raw_pkt = self.packetizer.packetize(symbols, language=lang, packet_id=0, total_packets=1)

            # Check 1-byte header field offset: [0:4] sync, [4:5] version, [5:6] lang_id
            self.assertEqual(raw_pkt[5], expected_id)
            self.assertLessEqual(raw_pkt[5], 255) # Fits in 1-byte unsigned char

            decoded = self.packetizer.depacketize(raw_pkt)
            self.assertEqual(decoded.language, lang)
            self.assertTrue(decoded.crc_valid)
            self.assertTrue(np.allclose(decoded.payload_symbols, symbols, atol=1e-6))

    def test_unknown_language_packet_fallback(self):
        """Unknown languages map to ID 0 and recover as 'unknown' without CRC corruption."""
        symbols = np.array([0.5, -0.5], dtype=np.float32)
        raw_pkt = self.packetizer.packetize(symbols, language="xyz", packet_id=0, total_packets=1)

        self.assertEqual(raw_pkt[5], 0)
        decoded = self.packetizer.depacketize(raw_pkt)
        self.assertEqual(decoded.language, "unknown")
        self.assertTrue(decoded.crc_valid)

    def test_new_languages_stream_reassembly(self):
        """Verify stream packetization and reassembly for the 3 new languages (kn, ml, gu)."""
        symbols = np.random.randn(80).astype(np.float32)
        for lang in ["kn", "ml", "gu"]:
            packets = self.packetizer.packetize_stream(symbols, language=lang)
            reassembled, rx_lang, all_valid = self.packetizer.depacketize_stream(packets)
            self.assertEqual(rx_lang, lang)
            self.assertTrue(all_valid)
            self.assertTrue(np.allclose(reassembled, symbols, atol=1e-6))


class TestSDRInterfaces(unittest.TestCase):
    """Test simulated SDR loopback and hardware stub isolation."""

    def test_simulated_sdr_loopback(self):
        sdr = SimulatedSDR()
        with sdr:
            test_bytes = b"iTantra_Test_Transmission_0123"
            sent_len = sdr.transmit(test_bytes)
            self.assertEqual(sent_len, len(test_bytes))
            self.assertTrue(sdr.has_data())

            received = sdr.receive(max_bytes=100)
            self.assertEqual(received, test_bytes)
            self.assertFalse(sdr.has_data())

    def test_hardware_stub_raises_not_implemented(self):
        hw_sdr = HardwareSDRStub()
        with self.assertRaises(NotImplementedError):
            hw_sdr.open()
        with self.assertRaises(NotImplementedError):
            hw_sdr.transmit(b"data")


class TestTranslationEngine(unittest.TestCase):
    """Test multilingual Indian translation engine."""

    def setUp(self):
        self.translator = TranslationEngine()

    def test_hindi_to_telugu_greeting(self):
        res = self.translator.translate("नमस्ते, मेरा नाम अभिनय है।", "hi", "te")
        self.assertTrue(res.success)
        self.assertEqual(res.source_lang, "hi")
        self.assertEqual(res.target_lang, "te")
        self.assertIn("నమస్కారం", res.translated_text)

    def test_telugu_to_hindi_question(self):
        res = self.translator.translate("మీరు ఎలా ఉన్నారు?", "te", "hi")
        self.assertTrue(res.success)
        self.assertIn("आप कैसे हैं", res.translated_text)

    def test_identity_translation(self):
        res = self.translator.translate("नमस्ते", "hi", "hi")
        self.assertTrue(res.success)
        self.assertEqual(res.translated_text, "नमस्ते")
        self.assertEqual(res.backend, "identity")

    def test_out_of_vocabulary_fails_gracefully(self):
        res = self.translator.translate("कुछ याद नहीं आ रहा है", "hi", "te")
        self.assertFalse(res.success)
        self.assertIn("out of offline conversational lexicon", res.details)


class TestAutomaticLanguageDetection(unittest.TestCase):
    """Test Whisper-based automatic language identification and transcription."""

    @classmethod
    def setUpClass(cls):
        import os
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.stt = STTEngine(backend="whisper")

    def test_automatic_detection_hindi_telugu_tamil_bengali_kannada(self):
        """Test automatic language detection on standard audio samples."""
        import os
        import soundfile as sf

        test_cases = [
            ("hi", os.path.join(self.repo_root, "output", "tts_samples", "hi.wav"), "hi"),
            ("ta", os.path.join(self.repo_root, "output", "tts_samples", "ta.wav"), "ta"),
            ("bn", os.path.join(self.repo_root, "output", "tts_samples", "bn.wav"), "bn"),
            ("kn", os.path.join(self.repo_root, "output", "tts_samples", "kn.wav"), "kn"),
            ("te", os.path.join(self.repo_root, "output", "tts_samples", "te.wav"), "te"),
        ]

        for expected_code, path, target_detection in test_cases:
            if not os.path.exists(path):
                continue
            audio, sr = sf.read(path)
            detected_lang, conf, probs = self.stt.detect_language(audio)
            self.assertEqual(detected_lang, target_detection)
            self.assertGreater(conf, 0.20)
            self.assertIn(target_detection, probs)

    def test_transcribe_without_language_argument_auto_detects(self):
        """When language is None, transcribe() automatically identifies language and transcribes."""
        import os
        import soundfile as sf

        path = os.path.join(self.repo_root, "output", "demo_hindi.wav")
        audio, sr = sf.read(path)

        # Call transcribe without language argument
        res = self.stt.transcribe(audio)
        self.assertEqual(res.language, "hi")
        self.assertEqual(res.detected_language, "hi")
        self.assertIsNotNone(res.language_confidence)
        self.assertGreater(res.language_confidence, 0.50)
        self.assertGreater(len(res.text), 0)

    def test_transcribe_with_explicit_language_preserves_behavior(self):
        """When language is explicitly supplied, existing behavior is preserved."""
        import os
        import soundfile as sf

        path = os.path.join(self.repo_root, "output", "tts_samples", "te.wav")
        audio, sr = sf.read(path)

        res = self.stt.transcribe(audio, language="te")
        self.assertEqual(res.language, "te")
        self.assertIsNone(res.detected_language)  # Not detected because explicitly provided
        self.assertGreater(len(res.text), 0)

    def test_all_seven_languages_identification_audit(self):
        """Audit language detection across all seven in-scope languages and record metrics."""
        import os
        import soundfile as sf

        all_langs = [
            ("hi", "output/tts_samples/hi.wav"),
            ("te", "output/tts_samples/te.wav"),
            ("ta", "output/tts_samples/ta.wav"),
            ("bn", "output/tts_samples/bn.wav"),
            ("mr", "output/tts_samples/mr.wav"),
            ("kn", "output/tts_samples/kn.wav"),
            ("ml", "output/tts_samples/ml.wav"),
        ]

        for code, rel_path in all_langs:
            path = os.path.join(self.repo_root, rel_path)
            if not os.path.exists(path):
                continue
            audio, sr = sf.read(path)
            detected, conf, probs = self.stt.detect_language(audio)
            # Must return one of the 7 supported languages or None (never an out-of-scope code)
            self.assertIn(detected, ["hi", "te", "ta", "bn", "mr", "kn", "ml", None])
            # Target probabilities dictionary must contain all 7 supported languages
            self.assertEqual(len(probs), 7)

    def test_noise_or_unsupported_audio_returns_uncertain(self):
        """Noise or non-speech audio returns uncertain result without crashing or defaulting to Hindi."""
        sr = 16000
        noise = (np.random.randn(sr) * 0.05).astype(np.float32)

        # Direct detection should reject noise
        detected, conf, probs = self.stt.detect_language(noise)
        self.assertIsNone(detected)

        # Transcribe without language argument should return explicit uncertain result
        res = self.stt.transcribe(noise)
        self.assertEqual(res.language, "unknown")
        self.assertIn("Uncertain", res.text)
        self.assertEqual(res.confidence, 0.0)

    def test_lightweight_backend_language_detection(self):
        """Lightweight backend supports language detection in offline unit tests."""
        stt_lw = STTEngine(backend="lightweight")
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000))).astype(np.float32)

        detected, conf, probs = stt_lw.detect_language(audio)
        self.assertEqual(detected, "hi")
        self.assertGreater(conf, 0.5)

        res = stt_lw.transcribe(audio)
        self.assertEqual(res.language, "hi")
        self.assertEqual(res.detected_language, "hi")


if __name__ == "__main__":
    unittest.main()
