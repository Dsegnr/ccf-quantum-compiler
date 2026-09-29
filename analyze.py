#!/usr/bin/env python3
"""
Results Analysis and Visualization Script
=========================================

Analyzes evaluation results, generates comparison plots between
different compilation strategies, and produces summary statistics.

Usage:
  python analyze.py --results results/public_summary.json --out analysis/
"""
from __future__ import annotations

import argparse
import json
import sys

# Fix Unicode output on Windows
if sys.platform == 'win32':
  sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path


def load_summary(path: Path) -> dict:
  """Load evaluation summary JSON."""
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def print_summary_table(summary: dict) -> None:
  """Print a formatted summary table of evaluation results."""
  print(f"\n{'='*80}")
  print(f"Compiler: {summary.get('compiler_name', 'unknown')}")
  print(f"Split:    {summary.get('split', 'unknown')}")
  print(f"{'='*80}")

  samples = summary.get("sample_results", [])
  if not samples:
    print("No sample results found.")
    return

  # Group by track
  track_a = [s for s in samples if s.get("track") == "track_a"]
  track_b = [s for s in samples if s.get("track") == "track_b"]

  # Track A table
  if track_a:
    print(f"\n{'─'*80}")
    print("Track A Results")
    print(f"{'─'*80}")
    print(f"{'Benchmark':<20} {'Valid':<7} {'Success Rate':<13} "
          f"{'Correct':<9} {'Shots':<7} {'2Q Gates':<10} "
          f"{'Log Rel':<10} {'Time':<8}")
    print(f"{'─'*80}")
    for s in track_a:
      print(
        f"{s['benchmark_name']:<20} "
        f"{'✓' if s['valid'] else '✗':<7} "
        f"{s['success_rate']:>10.4f}   "
        f"{s['correct_shots']:>5}   "
        f"{s['shots']:>5} "
        f"{s['two_qubit_gate_count']:>8} "
        f"{s['estimated_log_reliability']:>8.2f} "
        f"{s['compile_time_sec']:>6.2f}s"
      )

    track_a_score = summary.get("track_a_score", 0)
    print(f"{'─'*80}")
    print(f"Track A Score: {track_a_score:.4f}")

  # Track B table
  if track_b:
    print(f"\n{'─'*80}")
    print("Track B Results")
    print(f"{'─'*80}")
    print(f"{'Benchmark':<20} {'Valid':<7} {'Success Rate':<13} "
          f"{'Correct':<9} {'Shots':<7} {'2Q Gates':<10} "
          f"{'Log Rel':<10} {'Time':<8}")
    print(f"{'─'*80}")
    for s in track_b:
      print(
        f"{s['benchmark_name']:<20} "
        f"{'✓' if s['valid'] else '✗':<7} "
        f"{s['success_rate']:>10.4f}   "
        f"{s['correct_shots']:>5}   "
        f"{s['shots']:>5} "
        f"{s['two_qubit_gate_count']:>8} "
        f"{s['estimated_log_reliability']:>8.2f} "
        f"{s['compile_time_sec']:>6.2f}s"
      )

    track_b_score = summary.get("track_b_score", 0)
    print(f"{'─'*80}")
    print(f"Track B Score: {track_b_score:.4f}")

  # Overall
  final_score = summary.get("final_score", 0)
  print(f"\n{'='*80}")
  print(f"FINAL SCORE: {final_score:.4f} "
        f"(0.75 × {summary.get('track_a_score', 0):.4f} "
        f"+ 0.25 × {summary.get('track_b_score', 0):.4f})")
  print(f"{'='*80}")

  # Validation errors
  invalid = [s for s in samples if not s.get("valid", True)]
  if invalid:
    print(f"\nValidation Errors:")
    for s in invalid:
      errors = s.get("validation_errors", [])
      print(f"  {s['benchmark_name']}: {errors}")


def compare_summaries(summaries: dict[str, dict]) -> None:
  """Compare multiple evaluation summaries side by side."""
  if not summaries:
    return

  print(f"\n{'='*80}")
  print("Comparison of Compilers")
  print(f"{'='*80}")
  print(f"{'Compiler':<30} {'Track A':<10} {'Track B':<10} {'Final':<10}")
  print(f"{'─'*80}")

  for name, summary in summaries.items():
    print(
      f"{name:<30} "
      f"{summary.get('track_a_score', 0):>8.4f}  "
      f"{summary.get('track_b_score', 0):>8.4f}  "
      f"{summary.get('final_score', 0):>8.4f}"
    )


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Analyze quantum compiler evaluation results"
  )
  parser.add_argument(
    "--results", type=Path, nargs="+",
    help="Evaluation summary JSON file(s)",
  )
  parser.add_argument(
    "--compare", type=Path, nargs="+",
    help="Compare multiple summary JSON files",
  )
  parser.add_argument(
    "--out", type=Path, default=Path("analysis"),
    help="Output directory for analysis artifacts",
  )
  args = parser.parse_args()

  if args.compare:
    summaries = {}
    for path in args.compare:
      name = path.stem
      summaries[name] = load_summary(path)
    compare_summaries(summaries)
    return 0

  if args.results:
    for path in args.results:
      summary = load_summary(path)
      print_summary_table(summary)
    return 0

  parser.print_help()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
