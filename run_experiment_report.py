"""
Run SOVEREIGN rulebook experiments independently and write summaries to a text file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from Sovereign_agent import run_protocol, RULEBOOK_EXPERIMENT_ORDER


def _format_counts(counts: Dict[str, int], order: List[str]) -> str:
    lines = []
    total = sum(int(v) for v in counts.values())
    for key in order:
        value = int(counts.get(key, 0))
        pct = (100.0 * value / total) if total > 0 else 0.0
        lines.append(f"    - {key:<22s}: {value:5d} ({pct:5.1f}%)")
    return "\n".join(lines)


def write_report(
    output_file: str,
    steps: int,
    seeds: List[int],
    eval_episodes: int,
) -> Path:
    all_results: Dict[str, Dict[str, Any]] = {}

    # Run each condition independently to keep each experiment isolated.
    for condition in RULEBOOK_EXPERIMENT_ORDER:
        result = run_protocol(
            total_steps=steps,
            seeds=seeds,
            eval_episodes=eval_episodes,
            condition_names=[condition],
            verbose=True,
        )
        all_results[condition] = result[condition]

    path = Path(output_file)
    lines: List[str] = []
    lines.append("SOVEREIGN Experiment Summary Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Steps per condition: {steps}")
    lines.append(f"Seeds: {seeds}")
    lines.append(f"Eval episodes per seed: {eval_episodes}")
    lines.append("=" * 72)

    train_term_order = [
        "negotiated_settlement",
        "conquest",
        "legitimacy_collapse",
        "military_defeat",
        "timeout",
    ]

    eval_term_order = [
        "negotiated_settlement",
        "conquest",
        "legitimacy_collapse",
        "military_defeat",
        "timeout",
        "unknown",
    ]

    for condition in RULEBOOK_EXPERIMENT_ORDER:
        data = all_results[condition]
        aggregate_eval = data.get("aggregate_eval", {})
        runs = data.get("runs", [])
        expected_policy = data.get("config", {}).get("expected_policy", "N/A")

        lines.append("")
        lines.append(f"Condition: {condition}")
        lines.append(f"Expected policy (rulebook): {expected_policy}")
        lines.append("-" * 72)

        for run in runs:
            seed = run.get("seed")
            train_summary = run.get("train_summary", {})
            train_term_counts = run.get("train_termination_counts", {})
            eval_summary = run.get("eval", {})
            eval_term_counts = eval_summary.get("termination_counts", {})

            lines.append(f"  Seed {seed}")
            lines.append(
                "  Train summary: "
                f"avg_reward={train_summary.get('avg_reward', 0.0):.3f}, "
                f"avg_invasion_rate={train_summary.get('avg_invasion_rate', 0.0):.3f}, "
                f"avg_diplomacy_rate={train_summary.get('avg_diplomacy_rate', 0.0):.3f}, "
                f"avg_non_home_territories={train_summary.get('avg_non_home_territories', 0.0):.3f}"
            )
            lines.append("  Train termination counts:")
            lines.append(_format_counts(train_term_counts, train_term_order))

            lines.append(
                "  Eval summary: "
                f"R={eval_summary.get('mean_reward', 0.0):.3f}, "
                f"Inv={eval_summary.get('mean_invasion_rate', 0.0):.3f}, "
                f"Dip={eval_summary.get('mean_diplomacy_rate', 0.0):.3f}, "
                f"OccTerr={eval_summary.get('mean_non_home_territories', 0.0):.3f}, "
                f"L_final={eval_summary.get('mean_L_final', 0.0):.3f}, "
                f"theta_final={eval_summary.get('mean_theta_final', 0.0):.3f}"
            )
            lines.append("  Eval termination counts:")
            lines.append(_format_counts(eval_term_counts, eval_term_order))
            lines.append("")

        lines.append(
            "  Aggregate eval across seeds: "
            f"R={aggregate_eval.get('mean_reward_mean', 0.0):.3f}+-{aggregate_eval.get('mean_reward_std', 0.0):.3f}, "
            f"Inv={aggregate_eval.get('mean_invasion_rate_mean', 0.0):.3f}+-{aggregate_eval.get('mean_invasion_rate_std', 0.0):.3f}, "
            f"Dip={aggregate_eval.get('mean_diplomacy_rate_mean', 0.0):.3f}+-{aggregate_eval.get('mean_diplomacy_rate_std', 0.0):.3f}, "
            f"OccTerr={aggregate_eval.get('mean_non_home_territories_mean', 0.0):.3f}+-{aggregate_eval.get('mean_non_home_territories_std', 0.0):.3f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="experiment_summary_report.txt")
    args = parser.parse_args()

    output_path = write_report(
        output_file=args.out,
        steps=args.steps,
        seeds=args.seeds,
        eval_episodes=args.eval_episodes,
    )
    print(f"Wrote report to: {output_path}")
