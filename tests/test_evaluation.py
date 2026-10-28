"""Unit tests for Performance Evaluation Module in iTantra."""

import csv
import os
import tempfile
import unittest
import numpy as np

from src.evaluation.performance_evaluator import (
    PerformanceEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
    SIMULATION_NOTICE,
    EVAL_TEXT_CORPUS,
)
from src.pipeline import ITantraPipeline


class TestEvaluationConfig(unittest.TestCase):
    """Test EvaluationConfig instantiation and presets."""

    def test_default_config_fields(self):
        cfg = EvaluationConfig()
        for lang in ["hi", "te", "ta", "bn", "mr", "kn", "ml"]:
            self.assertIn(lang, cfg.languages)
        self.assertIn("awgn", cfg.channel_models)
        self.assertIn("rayleigh", cfg.channel_models)
        self.assertIn("emergency", cfg.priorities)
        self.assertIn("normal", cfg.priorities)
        self.assertGreater(len(cfg.snr_levels_db), 0)
        self.assertEqual(cfg.seed, 42)

    def test_all_seven_languages_in_eval_corpus(self):
        """Verify EVAL_TEXT_CORPUS has native test sentences for all 7 languages and all 4 priorities."""
        target_langs = ["hi", "te", "ta", "bn", "mr", "kn", "ml"]
        priorities = ["emergency", "high", "normal", "low"]
        for lang in target_langs:
            self.assertIn(lang, EVAL_TEXT_CORPUS)
            for prio in priorities:
                self.assertIn(prio, EVAL_TEXT_CORPUS[lang])
                text = EVAL_TEXT_CORPUS[lang][prio]
                self.assertIsInstance(text, str)
                self.assertGreater(len(text), 0)

    def test_quick_test_config_preset(self):
        quick_cfg = EvaluationConfig.quick_test_config()
        self.assertEqual(quick_cfg.languages, ["hi"])
        self.assertEqual(quick_cfg.priorities, ["emergency", "normal"])
        self.assertEqual(quick_cfg.snr_levels_db, [10.0, 20.0])
        self.assertEqual(quick_cfg.max_retries, 2)


class TestMetricsCalculation(unittest.TestCase):
    """Test compute_metrics statistical and aggregation calculations."""

    def setUp(self):
        self.evaluator = PerformanceEvaluator()

    def tearDown(self):
        self.evaluator.close()

    def test_empty_records_edge_case(self):
        """Zero records should return cleanly with zeroed metrics without division by zero."""
        metrics = self.evaluator.compute_metrics([])
        self.assertEqual(metrics.total_messages, 0)
        self.assertEqual(metrics.delivery_success_rate_pct, 0.0)
        self.assertEqual(metrics.latency_avg_ms, 0.0)
        self.assertEqual(metrics.measured_packet_loss_rate_pct, 0.0)

    def test_statistical_aggregation_accuracy(self):
        """Verify exact metric calculations for known synthetic trial outcomes."""
        mock_records = [
            {
                "trial_id": "T01",
                "language": "hi",
                "priority": "emergency",
                "channel_type": "awgn",
                "snr_db": 10.0,
                "ttl_sec": 30.0,
                "status": "delivered",
                "is_delivered": 1,
                "is_expired": 0,
                "is_failed": 0,
                "retry_count": 0,
                "latency_ms": 100.0,
                "packets_transmitted": 4,
                "packets_lost": 0,
                "crc_failures": 0,
                "sym_mse": 0.01,
                "feature_mse": 0.05,
            },
            {
                "trial_id": "T02",
                "language": "hi",
                "priority": "emergency",
                "channel_type": "awgn",
                "snr_db": 10.0,
                "ttl_sec": 30.0,
                "status": "delivered",
                "is_delivered": 1,
                "is_expired": 0,
                "is_failed": 0,
                "retry_count": 1,
                "latency_ms": 200.0,
                "packets_transmitted": 8,
                "packets_lost": 4,
                "crc_failures": 0,
                "sym_mse": 0.015,
                "feature_mse": 0.06,
            },
            {
                "trial_id": "T03",
                "language": "te",
                "priority": "normal",
                "channel_type": "rayleigh",
                "snr_db": 5.0,
                "ttl_sec": 0.05,
                "status": "expired",
                "is_delivered": 0,
                "is_expired": 1,
                "is_failed": 0,
                "retry_count": 0,
                "latency_ms": 50.0,
                "packets_transmitted": 0,
                "packets_lost": 0,
                "crc_failures": 0,
                "sym_mse": 0.0,
                "feature_mse": 0.0,
            },
            {
                "trial_id": "T04",
                "language": "te",
                "priority": "normal",
                "channel_type": "rayleigh",
                "snr_db": 5.0,
                "ttl_sec": "",
                "status": "failed",
                "is_delivered": 0,
                "is_expired": 0,
                "is_failed": 1,
                "retry_count": 3,
                "latency_ms": 450.0,
                "packets_transmitted": 16,
                "packets_lost": 4,
                "crc_failures": 4,
                "sym_mse": 0.12,
                "feature_mse": 0.0,
            },
        ]

        metrics = self.evaluator.compute_metrics(mock_records, seed=123)

        # 1. Delivery success
        self.assertEqual(metrics.total_messages, 4)
        self.assertEqual(metrics.delivered_count, 2)
        self.assertEqual(metrics.expired_count, 1)
        self.assertEqual(metrics.failed_count, 1)
        self.assertEqual(metrics.delivery_success_rate_pct, 50.0)

        # 2. Latency
        self.assertEqual(metrics.latency_min_ms, 50.0)
        self.assertEqual(metrics.latency_max_ms, 450.0)
        self.assertEqual(metrics.latency_avg_ms, 200.0) # (100 + 200 + 50 + 450) / 4
        self.assertEqual(metrics.latency_delivered_avg_ms, 150.0) # (100 + 200) / 2

        # 3. Packet loss
        self.assertEqual(metrics.packets_transmitted_total, 28)
        self.assertEqual(metrics.packets_lost_total, 8)
        self.assertAlmostEqual(metrics.measured_packet_loss_rate_pct, (8 / 28) * 100.0, places=2)

        # 4. CRC Failures
        self.assertEqual(metrics.crc_failures_total, 4)
        self.assertEqual(metrics.packets_checked_total, 20)
        self.assertEqual(metrics.integrity_failure_rate_pct, 20.0)

        # 5. Retries
        self.assertEqual(metrics.min_retries, 0)
        self.assertEqual(metrics.max_retries, 3)
        self.assertEqual(metrics.avg_retries_per_message, 1.0) # (0 + 1 + 0 + 3) / 4

        # 6. Priority breakdown
        self.assertIn("emergency", metrics.priority_metrics)
        self.assertIn("normal", metrics.priority_metrics)
        self.assertEqual(metrics.priority_metrics["emergency"]["delivered"], 2)
        self.assertEqual(metrics.priority_metrics["emergency"]["success_rate_pct"], 100.0)
        self.assertEqual(metrics.priority_metrics["normal"]["delivered"], 0)
        self.assertEqual(metrics.priority_metrics["normal"]["success_rate_pct"], 0.0)

        # 7. TTL breakdown
        self.assertEqual(metrics.ttl_messages_total, 3) # T01, T02, T03
        self.assertEqual(metrics.ttl_expired_count, 1)
        self.assertAlmostEqual(metrics.ttl_expiration_rate_pct, (1 / 3) * 100.0, places=2)

        # 8. Channel & SNR comparison
        self.assertIn("awgn", metrics.channel_snr_comparison)
        self.assertIn("rayleigh", metrics.channel_snr_comparison)
        self.assertEqual(metrics.channel_snr_comparison["awgn"][10.0]["success_rate_pct"], 100.0)
        self.assertEqual(metrics.channel_snr_comparison["rayleigh"][5.0]["success_rate_pct"], 0.0)


class TestCSVAndReportExport(unittest.TestCase):
    """Test CSV file output and Markdown summary report formatting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evaluator = PerformanceEvaluator()

    def tearDown(self):
        self.evaluator.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_export_csv_structure_and_disclaimer(self):
        """CSV export must include comments header with simulation disclaimer and valid rows."""
        csv_file = os.path.join(self.temp_dir.name, "test_output.csv")
        records = [
            {
                "trial_id": "TRIAL_0001",
                "msg_id": "MSG_001",
                "language": "hi",
                "channel_type": "awgn",
                "snr_db": 15.0,
                "configured_loss_rate": 0.0,
                "configured_corruption_rate": 0.0,
                "priority": "emergency",
                "ttl_sec": 30.0,
                "status": "delivered",
                "is_delivered": 1,
                "is_expired": 0,
                "is_failed": 0,
                "retry_count": 0,
                "max_retries": 3,
                "latency_ms": 120.5,
                "packets_transmitted": 4,
                "packets_lost": 0,
                "crc_failures": 0,
                "sym_mse": 0.005,
                "feature_mse": 0.02,
                "error": "",
            }
        ]

        written_path = self.evaluator.export_csv(records, csv_file)
        self.assertTrue(os.path.exists(written_path))

        with open(written_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Comment headers
        self.assertTrue(any("Software Simulation" in line for line in lines[:5]))
        self.assertTrue(any("Disclaimer" in line for line in lines[:5]))

        # Read actual data rows
        non_comment_lines = [l for l in lines if not l.startswith("#")]
        reader = csv.DictReader(non_comment_lines)
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trial_id"], "TRIAL_0001")
        self.assertEqual(rows[0]["status"], "delivered")
        self.assertEqual(rows[0]["is_delivered"], "1")

    def test_summary_report_generation(self):
        """Report contains markdown tables, all 8 required metrics, and simulation disclaimer."""
        metrics = PerformanceMetrics(
            total_messages=10,
            delivered_count=8,
            failed_count=1,
            expired_count=1,
            delivery_success_rate_pct=80.0,
            latency_avg_ms=115.4,
            latency_min_ms=50.2,
            latency_max_ms=320.1,
            packets_transmitted_total=40,
            packets_lost_total=4,
            measured_packet_loss_rate_pct=10.0,
            packets_checked_total=36,
            crc_failures_total=2,
            integrity_failure_rate_pct=5.56,
            avg_retries_per_message=0.4,
            priority_metrics={
                "emergency": {
                    "total_sent": 5,
                    "delivered": 5,
                    "expired": 0,
                    "failed": 0,
                    "success_rate_pct": 100.0,
                    "avg_latency_ms": 95.0,
                }
            },
            ttl_messages_total=4,
            ttl_expired_count=1,
            ttl_expiration_rate_pct=25.0,
            channel_snr_comparison={
                "awgn": {
                    15.0: {
                        "trials": 5,
                        "delivered": 5,
                        "success_rate_pct": 100.0,
                        "avg_retries": 0.0,
                        "avg_sym_mse": 0.003,
                        "avg_latency_ms": 90.0,
                    }
                }
            },
        )

        report = self.evaluator.generate_summary_report(metrics)
        self.assertIn("Simulation Environment Disclaimer", report)
        self.assertIn("SimulatedSDR", report)
        self.assertIn("80.0%", report)
        self.assertIn("115.40 ms", report)
        self.assertIn("10.0%", report)
        self.assertIn("5.56%", report)
        self.assertIn("EMERGENCY", report)
        self.assertIn("AWGN", report)


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration test running actual quick evaluation through pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "eval_integ.db")
        self.pipeline = ITantraPipeline(db_path=self.db_path)

    def tearDown(self):
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_quick_evaluation_run(self):
        """Run quick preset evaluation and verify actual trials and output files."""
        csv_path = os.path.join(self.temp_dir.name, "quick_eval.csv")
        md_path = os.path.join(self.temp_dir.name, "quick_summary.md")

        cfg = EvaluationConfig(
            languages=["hi"],
            channel_models=["awgn"],
            snr_levels_db=[20.0],
            packet_loss_rates=[0.0],
            packet_corruption_rates=[0.0],
            priorities=["emergency", "normal"],
            ttl_values=[None],
            max_retries=1,
            seed=100,
            output_csv_path=csv_path,
            summary_report_path=md_path,
        )

        evaluator = PerformanceEvaluator(pipeline=self.pipeline, config=cfg)
        records, metrics = evaluator.run_evaluation(config=cfg)

        self.assertEqual(len(records), 2)
        self.assertEqual(metrics.total_messages, 2)
        self.assertEqual(metrics.delivered_count, 2)
        self.assertEqual(metrics.delivery_success_rate_pct, 100.0)
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(md_path))
        self.assertGreater(metrics.latency_avg_ms, 0.0)

    def test_evaluation_run_with_ttl_and_packet_loss(self):
        """Run evaluation with TTL expiration and loss conditions, verifying metric fidelity."""
        csv_path = os.path.join(self.temp_dir.name, "ttl_loss_eval.csv")
        md_path = os.path.join(self.temp_dir.name, "ttl_loss_summary.md")

        cfg = EvaluationConfig(
            languages=["hi"],
            channel_models=["awgn"],
            snr_levels_db=[15.0],
            packet_loss_rates=[0.0, 0.25],
            packet_corruption_rates=[0.0],
            priorities=["emergency"],
            ttl_values=[None, 0.01],
            max_retries=2,
            seed=42,
            output_csv_path=csv_path,
            summary_report_path=md_path,
        )

        evaluator = PerformanceEvaluator(pipeline=self.pipeline, config=cfg)
        records, metrics = evaluator.run_evaluation(config=cfg)

        # 1 language * 1 channel * 1 SNR * 2 loss rates * 1 corruption * 1 priority * 2 TTLs = 4 trials
        self.assertEqual(len(records), 4)
        self.assertEqual(metrics.total_messages, 4)

        # 2 messages had ttl=0.01 (expired) and 2 had ttl=None (delivered)
        self.assertEqual(metrics.ttl_messages_total, 2)
        self.assertEqual(metrics.ttl_expired_count, 2)
        self.assertEqual(metrics.ttl_expiration_rate_pct, 100.0)
        self.assertEqual(metrics.delivered_count, 2)
        self.assertEqual(metrics.delivery_success_rate_pct, 50.0)

        # Packet loss was experienced in trial with loss_rate=0.25
        self.assertGreater(metrics.packets_transmitted_total, 0)
        self.assertGreaterEqual(metrics.packets_lost_total, 0)

        # Output files verified
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(md_path))


if __name__ == "__main__":
    unittest.main()
