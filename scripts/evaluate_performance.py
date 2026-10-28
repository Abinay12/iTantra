"""Repeatable Performance Evaluation Script for iTantra.

Measures:
  1. End-to-end message delivery success rate.
  2. Average, minimum, and maximum delivery latency.
  3. Packet loss rate.
  4. Packet corruption or integrity-check failure rate.
  5. Number of retries per message.
  6. Successful delivery rate for each priority level.
  7. Number of messages expired due to TTL.
  8. AWGN versus Rayleigh channel performance at configurable SNRs.

Outputs:
  - CSV file recording individual trial outcomes.
  - Markdown summary report with comprehensive statistics.
  - Formatted console summary output.

Usage:
  python scripts/evaluate_performance.py --help
  python scripts/evaluate_performance.py --quick
  python scripts/evaluate_performance.py --snrs 5,15,25 --channels awgn,rayleigh
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.evaluation.performance_evaluator import (
    PerformanceEvaluator,
    EvaluationConfig,
    SIMULATION_NOTICE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable performance evaluation on iTantra neural transceiver stack."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast evaluation preset with minimal matrix for quick sanity check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible simulation runs (default: 42).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="hi,te",
        help="Comma-separated language codes to evaluate (e.g. 'hi,te').",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="awgn,rayleigh",
        help="Comma-separated channel models ('awgn,rayleigh').",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="5.0,15.0,25.0",
        help="Comma-separated SNR levels in dB (e.g. '5.0,15.0,25.0').",
    )
    parser.add_argument(
        "--loss-rates",
        type=str,
        default="0.0,0.25",
        help="Comma-separated simulated packet loss rates (e.g. '0.0,0.25').",
    )
    parser.add_argument(
        "--corruption-rates",
        type=str,
        default="0.0,0.1",
        help="Comma-separated simulated packet corruption rates (e.g. '0.0,0.1').",
    )
    parser.add_argument(
        "--priorities",
        type=str,
        default="emergency,high,normal,low",
        help="Comma-separated priority levels ('emergency,high,normal,low').",
    )
    parser.add_argument(
        "--ttls",
        type=str,
        default="none,30.0,0.05",
        help="Comma-separated TTL values in seconds or 'none' (e.g. 'none,30.0,0.05').",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum ARQ retransmission attempts per message (default: 3).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional path to isolated SQLite message database for benchmark runs.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output/performance_evaluation_results.csv",
        help="Destination path for detailed trial results CSV.",
    )
    parser.add_argument(
        "--summary-report",
        type=str,
        default="output/performance_evaluation_summary.md",
        help="Destination path for markdown summary report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("      iTantra: Performance Evaluation & Benchmark Engine")
    print("=" * 80)
    print(f"Environment: Software Simulation (SimulatedSDR loopback + AWGN/Rayleigh)")
    print(f"Hardware Disclaimer: No physical radio or phone-to-phone RF testing conducted.")
    print(f"Random Seed: {args.seed}")
    print("=" * 80)

    if args.quick:
        print("\n[*] Running in --quick mode with streamlined benchmark matrix...")
        config = EvaluationConfig.quick_test_config()
        config.seed = args.seed
        if args.output_csv:
            config.output_csv_path = args.output_csv
        if args.summary_report:
            config.summary_report_path = args.summary_report
    else:
        langs = [s.strip() for s in args.languages.split(",") if s.strip()]
        channels = [s.strip().lower() for s in args.channels.split(",") if s.strip()]
        snrs = [float(s.strip()) for s in args.snrs.split(",") if s.strip()]
        loss_rates = [float(s.strip()) for s in args.loss_rates.split(",") if s.strip()]
        corruption_rates = [float(s.strip()) for s in args.corruption_rates.split(",") if s.strip()]
        prios = [s.strip().lower() for s in args.priorities.split(",") if s.strip()]
        ttl_vals = []
        for s in args.ttls.split(","):
            s_clean = s.strip().lower()
            if not s_clean:
                continue
            if s_clean in ("none", "null", "nil"):
                ttl_vals.append(None)
            else:
                ttl_vals.append(float(s_clean))

        config = EvaluationConfig(
            languages=langs,
            channel_models=channels,
            snr_levels_db=snrs,
            packet_loss_rates=loss_rates,
            packet_corruption_rates=corruption_rates,
            priorities=prios,
            ttl_values=ttl_vals,
            max_retries=args.max_retries,
            seed=args.seed,
            output_csv_path=args.output_csv,
            summary_report_path=args.summary_report,
        )

    evaluator = PerformanceEvaluator(config=config, db_path=args.db_path)

    def on_progress(current, total, trial):
        prio = trial["priority"].upper()
        ch = trial["channel_type"].upper()
        snr = trial["snr_db"]
        status = trial["status"].upper()
        loss = trial["configured_loss_rate"]
        corrupt = trial["configured_corruption_rate"]
        lat = trial["latency_ms"]
        ret = trial["retry_count"]
        print(
            f"[{current:03d}/{total:03d}] {trial['trial_id']} | "
            f"Lang: {trial['language']} | Prio: {prio:<9} | Ch: {ch} @ {snr:>4.1f}dB | "
            f"Loss: {loss:.2f} Corrupt: {corrupt:.2f} | "
            f"Retries: {ret} | Lat: {lat:>6.1f}ms -> {status}"
        )

    start_eval_time = time.time()
    trial_records, metrics = evaluator.run_evaluation(config=config, progress_callback=on_progress)
    total_elapsed = time.time() - start_eval_time

    print("\n" + "=" * 80)
    print("                     EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Elapsed Time:      {total_elapsed:.2f} seconds")
    print(f"Total Trials:      {metrics.total_messages}")
    print(f"CSV Output:        {os.path.abspath(config.output_csv_path)}")
    print(f"Summary Report:    {os.path.abspath(config.summary_report_path)}")
    print("=" * 80)

    # Print summary report to console
    report = evaluator.generate_summary_report(metrics)
    print("\n" + report)
    print("=" * 80)

    evaluator.close()


if __name__ == "__main__":
    main()
