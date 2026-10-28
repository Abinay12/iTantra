"""Performance Evaluation Module for iTantra Neural Transceiver & Radio Stack.

Measures:
  1. End-to-end message delivery success rate.
  2. Average, minimum, and maximum delivery latency.
  3. Packet loss rate.
  4. Packet corruption or integrity-check failure rate.
  5. Number of retries per message.
  6. Successful delivery rate for each priority level.
  7. Number of messages expired due to TTL.
  8. AWGN versus Rayleigh channel performance at configurable SNRs.

NOTE: All evaluations are conducted in software simulation using SimulatedSDR loopback
and stochastic mathematical channel models (AWGNChannel and RayleighChannel).
No physical radio transmissions or hardware RF testing are performed or claimed.
"""

from dataclasses import dataclass, field
import csv
import io
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.pipeline import ITantraPipeline
from src.radio.reliable_delivery import DeliveryStatus, MessagePriority


# Standard reproducible multilingual test corpus across priority levels
EVAL_TEXT_CORPUS: Dict[str, Dict[str, str]] = {
    "hi": {
        "emergency": "आपातकालीन चिकित्सा सहायता की आवश्यकता है!",
        "high": "बाढ़ की चेतावनी: तुरंत सुरक्षित स्थान पर जाएं",
        "normal": "नमस्ते, आप कैसे हैं?",
        "low": "दैनिक मौसम रिपोर्ट: आज धूप खिली रहेगी",
    },
    "te": {
        "emergency": "అత్యవసర వైద్య సహాయం తక్షణమే కావాలి!",
        "high": "వరద హెచ్చరిక: వెంటనే సురక్షిత ప్రాంతానికి వెళ్ళండి",
        "normal": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "low": "వాతావరణ సమాచారం: ఈరోజు ఎండగా ఉంటుంది",
    },
    "ta": {
        "emergency": "அவசர மருத்துவ உதவி தேவைப்படுகிறது!",
        "high": "வெள்ள எச்சரிக்கை: உடனடியாக பாதுகாப்பான இடத்திற்கு செல்லவும்",
        "normal": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "low": "தினசரி வானிலை அறிக்கை: இன்று வெயில் இருக்கும்",
    },
    "bn": {
        "emergency": "জরুরি চিকিৎসা সহায়তা প্রয়োজন!",
        "high": "বন্যা সতর্কতা: অবিলম্বে নিরাপদ স্থানে যান",
        "normal": "নমস্কার, আপনি কেমন আছেন?",
        "low": "দৈনিক আবহাওয়া রিপোর্ট: আজ রোদ থাকবে",
    },
    "mr": {
        "emergency": "तातडीची वैद्यकीय मदत आवश्यक आहे!",
        "high": "पुराचा इशारा: त्वरित सुरक्षित स्थळी जा",
        "normal": "नमस्कार, तुम्ही कसे आहात?",
        "low": "दैनिक हवामान अहवाल: आज सूर्यप्रकाश असेल",
    },
    "kn": {
        "emergency": "ತುರ್ತು ವೈದ್ಯಕೀಯ ನೆರವು ಅಗತ್ಯವಿದೆ!",
        "high": "ಪ್ರವಾಹ ಎಚ್ಚರಿಕೆ: ತಕ್ಷಣ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ತೆರಳಿ",
        "normal": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "low": "ದೈನಂದಿನ ಹವಾಮಾನ ವರದಿ: ಇಂದು ಬಿಸಿಲು ಇರುತ್ತದೆ",
    },
    "ml": {
        "emergency": "അടിയന്തര വൈദ്യസഹായം ആവശ്യമാണ്!",
        "high": "വെള്ളപ്പൊക്ക മുന്നറിയിപ്പ്: ഉടൻ സുരക്ഷിത സ്ഥാനത്തേക്ക് മാറുക",
        "normal": "നമസ്കാരം, സുഖമാണോ?",
        "low": "ദൈനംദിന കാലാവസ്ഥാ റിപ്പോർട്ട്: ഇന്ന് വെയിൽ ഉണ്ടായിരിക്കും",
    },
}

SIMULATION_NOTICE = (
    "All evaluations were conducted in a software simulation environment using "
    "SimulatedSDR software-loopback and stochastic AWGN/Rayleigh channel models. "
    "No physical radio hardware (USRP, HackRF, RTL-SDR) or phone-to-phone RF tests "
    "were conducted."
)


@dataclass
class EvaluationConfig:
    """Configurable parameters for repeatable performance evaluation."""
    languages: List[str] = field(default_factory=lambda: ["hi", "te", "ta", "bn", "mr", "kn", "ml"])
    channel_models: List[str] = field(default_factory=lambda: ["awgn", "rayleigh"])
    snr_levels_db: List[float] = field(default_factory=lambda: [5.0, 15.0, 25.0])
    packet_loss_rates: List[float] = field(default_factory=lambda: [0.0, 0.25])
    packet_corruption_rates: List[float] = field(default_factory=lambda: [0.0, 0.1])
    priorities: List[str] = field(default_factory=lambda: ["emergency", "high", "normal", "low"])
    ttl_values: List[Optional[float]] = field(default_factory=lambda: [None, 30.0, 0.05])
    max_retries: int = 3
    seed: int = 42
    output_csv_path: str = "output/performance_evaluation_results.csv"
    summary_report_path: str = "output/performance_evaluation_summary.md"

    @classmethod
    def quick_test_config(cls) -> "EvaluationConfig":
        """Fast configuration preset for quick integration and unit testing."""
        return cls(
            languages=["hi"],
            channel_models=["awgn", "rayleigh"],
            snr_levels_db=[10.0, 20.0],
            packet_loss_rates=[0.0],
            packet_corruption_rates=[0.0],
            priorities=["emergency", "normal"],
            ttl_values=[None, 0.01],
            max_retries=2,
            seed=42,
            output_csv_path="output/quick_test_results.csv",
            summary_report_path="output/quick_test_summary.md",
        )


@dataclass
class PerformanceMetrics:
    """Structured container for aggregated performance metrics."""
    # 1. Message delivery success
    total_messages: int = 0
    delivered_count: int = 0
    failed_count: int = 0
    expired_count: int = 0
    delivery_success_rate_pct: float = 0.0

    # 2. Latency metrics (milliseconds)
    latency_avg_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_delivered_avg_ms: float = 0.0

    # 3. Packet loss metrics
    packets_transmitted_total: int = 0
    packets_lost_total: int = 0
    measured_packet_loss_rate_pct: float = 0.0

    # 4. Integrity check / CRC failure metrics
    packets_checked_total: int = 0
    crc_failures_total: int = 0
    integrity_failure_rate_pct: float = 0.0

    # 5. Retries per message
    avg_retries_per_message: float = 0.0
    min_retries: int = 0
    max_retries: int = 0
    retries_distribution: Dict[int, int] = field(default_factory=dict)

    # 6. Priority level breakdown
    priority_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 7. TTL expiration metrics
    ttl_messages_total: int = 0
    ttl_expired_count: int = 0
    ttl_expiration_rate_pct: float = 0.0

    # 8. AWGN vs Rayleigh comparison at configurable SNRs
    channel_snr_comparison: Dict[str, Dict[float, Dict[str, Any]]] = field(default_factory=dict)

    # Simulation metadata
    simulation_disclaimer: str = SIMULATION_NOTICE
    seed_used: int = 42
    timestamp: float = field(default_factory=time.time)


class PerformanceEvaluator:
    """Repeatable benchmark and performance evaluation engine for iTantra."""

    def __init__(
        self,
        pipeline: Optional[ITantraPipeline] = None,
        config: Optional[EvaluationConfig] = None,
        db_path: Optional[str] = None,
    ):
        """Initialize performance evaluator.

        Args:
            pipeline: Existing ITantraPipeline instance or None (creates a default instance).
            config: Benchmark configuration options.
            db_path: Optional custom path to SQLite database for evaluation messages.
        """
        self.config = config or EvaluationConfig()
        self._owned_pipeline = pipeline is None
        self.pipeline = pipeline or ITantraPipeline(db_path=db_path)

    def set_seed(self, seed: int):
        """Seed random number generators for strict repeatability."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run_evaluation(
        self,
        config: Optional[EvaluationConfig] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], PerformanceMetrics]:
        """Execute complete matrix of reproducible benchmark trials.

        Args:
            config: Optional override of benchmark configuration.
            progress_callback: Optional callback receiving (current_idx, total_trials, trial_dict).

        Returns:
            Tuple of (trial_records, aggregated_metrics).
        """
        cfg = config or self.config
        self.set_seed(cfg.seed)

        # Build list of trial configurations
        trial_plans = []
        trial_id = 1

        for lang in cfg.languages:
            corpus = EVAL_TEXT_CORPUS.get(lang, EVAL_TEXT_CORPUS["hi"])
            for prio in cfg.priorities:
                text = corpus.get(prio, corpus["normal"])
                for ch in cfg.channel_models:
                    for snr in cfg.snr_levels_db:
                        for loss_rate in cfg.packet_loss_rates:
                            for corrupt_rate in cfg.packet_corruption_rates:
                                for ttl in cfg.ttl_values:
                                    trial_plans.append({
                                        "trial_id": f"TRIAL_{trial_id:04d}",
                                        "lang": lang,
                                        "text": text,
                                        "priority": prio,
                                        "channel": ch,
                                        "snr": snr,
                                        "loss_rate": loss_rate,
                                        "corruption_rate": corrupt_rate,
                                        "ttl": ttl,
                                    })
                                    trial_id += 1

        total_trials = len(trial_plans)
        trial_records: List[Dict[str, Any]] = []

        for idx, plan in enumerate(trial_plans):
            msg_id = f"EVAL_{plan['trial_id']}"
            
            # If testing expired TTL (e.g. <= 0.05s) to verify expiration under queue delay,
            # simulate arrival timestamp that exceeds the configured TTL deadline.
            created_at = None
            if plan["ttl"] is not None and plan["ttl"] <= 0.05:
                created_at = time.time() - (plan["ttl"] + 1.0)

            # Reset receiver for clean per-trial measurements
            self.pipeline.reliable_receiver.reset()

            res = self.pipeline.send_reliable_message(
                msg_id=msg_id,
                text=plan["text"],
                language=plan["lang"],
                channel_type=plan["channel"],
                snr_db=plan["snr"],
                max_retries=cfg.max_retries,
                packet_loss_rate=plan["loss_rate"],
                packet_corruption_rate=plan["corruption_rate"],
                priority=plan["priority"],
                ttl=plan["ttl"],
                created_at=created_at,
            )

            record = {
                "trial_id": plan["trial_id"],
                "msg_id": msg_id,
                "language": plan["lang"],
                "text": plan["text"],
                "channel_type": plan["channel"],
                "snr_db": plan["snr"],
                "configured_loss_rate": plan["loss_rate"],
                "configured_corruption_rate": plan["corruption_rate"],
                "priority": plan["priority"],
                "ttl_sec": "" if plan["ttl"] is None else plan["ttl"],
                "status": res["status"],
                "is_delivered": int(res["is_delivered"]),
                "is_expired": int(res["is_expired"]),
                "is_failed": int(res["is_failed"]),
                "retry_count": res["retry_count"],
                "max_retries": res["max_retries"],
                "latency_ms": round(res["latency_ms"], 2),
                "packets_transmitted": res.get("packets_sent", 0),
                "packets_lost": res.get("packets_lost", 0),
                "crc_failures": res.get("crc_failures", 0),
                "sym_mse": round(res.get("sym_mse", 0.0), 6),
                "feature_mse": round(res.get("feature_mse", 0.0), 6),
                "error": res.get("error") or "",
            }
            trial_records.append(record)

            if progress_callback:
                progress_callback(idx + 1, total_trials, record)

        metrics = self.compute_metrics(trial_records, seed=cfg.seed)

        # Export CSV and Summary Report
        if cfg.output_csv_path:
            self.export_csv(trial_records, cfg.output_csv_path)
        if cfg.summary_report_path:
            report_text = self.generate_summary_report(metrics)
            self.save_summary_report(report_text, cfg.summary_report_path)

        return trial_records, metrics

    def compute_metrics(
        self,
        trial_records: List[Dict[str, Any]],
        seed: int = 42,
    ) -> PerformanceMetrics:
        """Calculate aggregated statistical metrics from raw trial records.

        Args:
            trial_records: List of per-trial execution dictionaries.
            seed: Seed value used during benchmark.

        Returns:
            Populated PerformanceMetrics instance.
        """
        metrics = PerformanceMetrics(seed_used=seed)
        total = len(trial_records)
        metrics.total_messages = total

        if total == 0:
            return metrics

        delivered = [r for r in trial_records if r["is_delivered"] == 1]
        expired = [r for r in trial_records if r["is_expired"] == 1]
        failed = [r for r in trial_records if r["is_failed"] == 1]

        metrics.delivered_count = len(delivered)
        metrics.expired_count = len(expired)
        metrics.failed_count = len(failed)
        metrics.delivery_success_rate_pct = round((len(delivered) / total) * 100.0, 2)

        # Latencies
        all_latencies = [r["latency_ms"] for r in trial_records]
        metrics.latency_avg_ms = round(float(np.mean(all_latencies)), 2)
        metrics.latency_min_ms = round(float(np.min(all_latencies)), 2)
        metrics.latency_max_ms = round(float(np.max(all_latencies)), 2)

        if delivered:
            delivered_latencies = [r["latency_ms"] for r in delivered]
            metrics.latency_delivered_avg_ms = round(float(np.mean(delivered_latencies)), 2)

        # Packet Loss
        pkts_sent = sum(r.get("packets_transmitted", 0) for r in trial_records)
        pkts_lost = sum(r.get("packets_lost", 0) for r in trial_records)
        metrics.packets_transmitted_total = pkts_sent
        metrics.packets_lost_total = pkts_lost
        metrics.measured_packet_loss_rate_pct = (
            round((pkts_lost / pkts_sent) * 100.0, 2) if pkts_sent > 0 else 0.0
        )

        # Packet Corruption / CRC Check Failures
        crc_fails = sum(r.get("crc_failures", 0) for r in trial_records)
        # Packets received at physical/link layer for checking
        pkts_checked = max(pkts_sent - pkts_lost, 0)
        metrics.packets_checked_total = pkts_checked
        metrics.crc_failures_total = crc_fails
        metrics.integrity_failure_rate_pct = (
            round((crc_fails / pkts_checked) * 100.0, 2) if pkts_checked > 0 else 0.0
        )

        # Retries
        retries_list = [r["retry_count"] for r in trial_records]
        metrics.avg_retries_per_message = round(float(np.mean(retries_list)), 2)
        metrics.min_retries = int(np.min(retries_list))
        metrics.max_retries = int(np.max(retries_list))

        dist: Dict[int, int] = {}
        for ret in retries_list:
            dist[ret] = dist.get(ret, 0) + 1
        metrics.retries_distribution = dict(sorted(dist.items()))

        # Priority Level Breakdown
        prio_metrics: Dict[str, Dict[str, Any]] = {}
        for p in ["emergency", "high", "normal", "low"]:
            sub = [r for r in trial_records if r["priority"].lower() == p]
            sub_total = len(sub)
            if sub_total > 0:
                sub_deliv = len([r for r in sub if r["is_delivered"] == 1])
                sub_exp = len([r for r in sub if r["is_expired"] == 1])
                sub_fail = len([r for r in sub if r["is_failed"] == 1])
                rate = round((sub_deliv / sub_total) * 100.0, 2)
                sub_lat = round(float(np.mean([r["latency_ms"] for r in sub])), 2)
                prio_metrics[p] = {
                    "total_sent": sub_total,
                    "delivered": sub_deliv,
                    "expired": sub_exp,
                    "failed": sub_fail,
                    "success_rate_pct": rate,
                    "avg_latency_ms": sub_lat,
                }
        metrics.priority_metrics = prio_metrics

        # TTL Metrics
        ttl_records = [r for r in trial_records if r["ttl_sec"] != ""]
        metrics.ttl_messages_total = len(ttl_records)
        ttl_expired = len([r for r in ttl_records if r["is_expired"] == 1])
        metrics.ttl_expired_count = ttl_expired
        metrics.ttl_expiration_rate_pct = (
            round((ttl_expired / len(ttl_records)) * 100.0, 2) if ttl_records else 0.0
        )

        # Channel & SNR Comparison (AWGN vs Rayleigh)
        comp: Dict[str, Dict[float, Dict[str, Any]]] = {}
        for ch in ["awgn", "rayleigh"]:
            comp[ch] = {}
            for r in trial_records:
                if r["channel_type"].lower() == ch:
                    snr = float(r["snr_db"])
                    if snr not in comp[ch]:
                        comp[ch][snr] = []
                    comp[ch][snr].append(r)

            # Aggregate per SNR
            for snr, sub_list in list(comp[ch].items()):
                n_sub = len(sub_list)
                deliv_n = len([x for x in sub_list if x["is_delivered"] == 1])
                comp[ch][snr] = {
                    "trials": n_sub,
                    "delivered": deliv_n,
                    "success_rate_pct": round((deliv_n / n_sub) * 100.0, 2),
                    "avg_latency_ms": round(float(np.mean([x["latency_ms"] for x in sub_list])), 2),
                    "avg_retries": round(float(np.mean([x["retry_count"] for x in sub_list])), 2),
                    "avg_sym_mse": round(float(np.mean([x["sym_mse"] for x in sub_list])), 6),
                    "avg_feat_mse": round(float(np.mean([x["feature_mse"] for x in sub_list if x["is_delivered"] == 1] or [0.0])), 6),
                }
        metrics.channel_snr_comparison = comp

        return metrics

    def export_csv(
        self,
        trial_records: List[Dict[str, Any]],
        filepath: str,
    ) -> str:
        """Write trial outcomes to a standard CSV file with metadata header.

        Args:
            trial_records: List of execution trial dictionaries.
            filepath: Target CSV destination path.

        Returns:
            Absolute filepath written.
        """
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if not trial_records:
            fieldnames = ["trial_id", "status"]
        else:
            fieldnames = list(trial_records[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            f.write("# iTantra Neural Transceiver Performance Evaluation Results\n")
            f.write(f"# Evaluation Type: Software Simulation (SimulatedSDR + AWGN/Rayleigh)\n")
            f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"# Total Trials: {len(trial_records)}\n")
            f.write("# Disclaimer: No physical radio or phone hardware was utilized.\n")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_records)

        return os.path.abspath(filepath)

    def generate_summary_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a concise, human-readable markdown summary of benchmark metrics.

        Args:
            metrics: PerformanceMetrics instance.

        Returns:
            Markdown formatted report string.
        """
        lines = []
        lines.append("# iTantra Performance Evaluation Summary Report")
        lines.append("")
        lines.append("> [!IMPORTANT]")
        lines.append("> **Simulation Environment Disclaimer**:")
        lines.append(f"> {metrics.simulation_disclaimer}")
        lines.append("")
        lines.append("## 1. Overall Transmission Success & Retries")
        lines.append("")
        lines.append("| Metric | Measured Value |")
        lines.append("|---|---|")
        lines.append(f"| Total Messages Tested | **{metrics.total_messages}** |")
        lines.append(f"| Successfully Delivered | **{metrics.delivered_count}** ({metrics.delivery_success_rate_pct}%) |")
        lines.append(f"| Messages Failed (Retries Exhausted) | **{metrics.failed_count}** |")
        lines.append(f"| Messages Expired (TTL Exceeded) | **{metrics.expired_count}** |")
        lines.append(f"| Average Retries / Message | **{metrics.avg_retries_per_message}** (Min: {metrics.min_retries}, Max: {metrics.max_retries}) |")
        lines.append("")

        lines.append("## 2. End-to-End Delivery Latency")
        lines.append("")
        lines.append("| Latency Measure | Value (ms) |")
        lines.append("|---|---|")
        lines.append(f"| Average Latency (All Messages) | **{metrics.latency_avg_ms:.2f} ms** |")
        lines.append(f"| Average Latency (Delivered Messages Only) | **{metrics.latency_delivered_avg_ms:.2f} ms** |")
        lines.append(f"| Minimum Latency | **{metrics.latency_min_ms:.2f} ms** |")
        lines.append(f"| Maximum Latency | **{metrics.latency_max_ms:.2f} ms** |")
        lines.append("")

        lines.append("## 3. Radio Link Packet Loss & Integrity (CRC-16)")
        lines.append("")
        lines.append("| Metric | Packets | Rate |")
        lines.append("|---|---|---|")
        lines.append(f"| Packets Transmitted | {metrics.packets_transmitted_total} | - |")
        lines.append(f"| Packets Lost in Transit | {metrics.packets_lost_total} | **{metrics.measured_packet_loss_rate_pct}%** |")
        lines.append(f"| Packets Checked for Integrity | {metrics.packets_checked_total} | - |")
        lines.append(f"| CRC Integrity Check Failures | {metrics.crc_failures_total} | **{metrics.integrity_failure_rate_pct}%** |")
        lines.append("")

        lines.append("## 4. Priority-Level Delivery Performance")
        lines.append("")
        lines.append("| Priority Level | Total Sent | Delivered | Expired | Failed | Success Rate | Avg Latency |")
        lines.append("|---|---|---|---|---|---|---|")
        for p, data in metrics.priority_metrics.items():
            lines.append(
                f"| **{p.upper()}** | {data['total_sent']} | {data['delivered']} | "
                f"{data['expired']} | {data['failed']} | **{data['success_rate_pct']}%** | "
                f"{data['avg_latency_ms']} ms |"
            )
        lines.append("")

        lines.append("## 5. Time-To-Live (TTL) Expiration Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Messages with Configured TTL | **{metrics.ttl_messages_total}** |")
        lines.append(f"| Messages Expired Due to TTL | **{metrics.ttl_expired_count}** |")
        lines.append(f"| TTL Expiration Rate | **{metrics.ttl_expiration_rate_pct}%** |")
        lines.append("")

        lines.append("## 6. Channel Performance Comparison: AWGN vs. Rayleigh")
        lines.append("")
        lines.append("| Channel Model | SNR (dB) | Trials | Delivered | Success Rate | Avg Retries | Symbol MSE | Avg Latency |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ch in ["awgn", "rayleigh"]:
            for snr, snr_data in sorted(metrics.channel_snr_comparison.get(ch, {}).items()):
                lines.append(
                    f"| **{ch.upper()}** | {snr:.1f} dB | {snr_data['trials']} | "
                    f"{snr_data['delivered']} | **{snr_data['success_rate_pct']}%** | "
                    f"{snr_data['avg_retries']:.2f} | {snr_data['avg_sym_mse']:.6f} | "
                    f"{snr_data['avg_latency_ms']:.1f} ms |"
                )
        lines.append("")

        return "\n".join(lines)

    def save_summary_report(self, report_text: str, filepath: str) -> str:
        """Write summary markdown report to disk.

        Args:
            report_text: Report content.
            filepath: Destination file path.

        Returns:
            Absolute filepath written.
        """
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        return os.path.abspath(filepath)

    def close(self):
        """Clean up pipeline resources if owned."""
        if self._owned_pipeline and hasattr(self.pipeline, "close"):
            self.pipeline.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
