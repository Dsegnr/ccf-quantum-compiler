#!/usr/bin/env python3
"""
Batch Evaluation Script for Noise-Adaptive SABRE Compiler
=========================================================

Runs the compiler on all public benchmarks and generates evaluation results.
Can be used both for local testing and for generating submission results.

Usage:
  # Evaluate on all public benchmarks
  python evaluate_all.py --root /path/to/release_zh \\
      --compiler "$(pwd)/compiler.py" \\
      --out results/

  # Quick test on a single benchmark
  python evaluate_all.py --root /path/to/release_zh \\
      --compiler "$(pwd)/compiler.py" \\
      --out results/ \\
      --benchmarks track_a_001
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkResult:
  """Result for a single benchmark."""
  benchmark_name: str
  track: str
  success: bool
  compile_time_sec: float
  error_message: str = ""


def find_release_root(root: Path) -> Path:
  """Locate the release_zh root directory."""
  # Accept either the release_zh directory itself or its parent
  if (root / "benchmarks").exists() and (root / "backends").exists():
    return root
  # Maybe they passed the parent directory
  release = root / "release_zh"
  if release.exists():
    # Check for nested structure
    inner = release / "release_zh"
    if inner.exists():
      return inner
    return release
  return root


def load_benchmarks(root: Path, tracks: list[str]) -> list[dict]:
  """Load all public benchmark metadata."""
  benchmarks = []
  for track in tracks:
    track_dir = root / "benchmarks" / track / "public"
    if not track_dir.exists():
      print(f"WARNING: Track directory not found: {track_dir}")
      continue
    for bench_dir in sorted(track_dir.iterdir()):
      if not bench_dir.is_dir():
        continue
      bench_json = bench_dir / "benchmark.json"
      if not bench_json.exists():
        continue
      with open(bench_json) as f:
        meta = json.load(f)
      meta["directory"] = str(bench_dir)
      meta["track"] = track
      meta["circuit_path"] = str(bench_dir / "circuit.json")
      benchmarks.append(meta)
  return benchmarks


def find_backend_path(root: Path, benchmark: dict) -> Path:
  """Find the backend JSON for a benchmark."""
  size_class = benchmark.get("backend_size_class", "medium")
  snapshot_id = benchmark.get("backend_snapshot_id", f"{size_class}_snapshot_a")
  backend_path = (
    root / "backends" / size_class / f"{snapshot_id}.json"
  )
  return backend_path


def run_compiler(
  command: list[str],
  circuit_path: Path,
  backend_path: Path,
  output_path: Path,
  timeout_sec: int = 60,
) -> tuple[bool, float, str]:
  """Run the compiler on a single benchmark.

  Returns:
    (success, compile_time_sec, error_message)
  """
  started = time.perf_counter()
  try:
    result = subprocess.run(
      command + [
        "--circuit", str(circuit_path),
        "--backend", str(backend_path),
        "--out", str(output_path),
      ],
      capture_output=True,
      text=True,
      timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
      stderr = result.stderr.strip() or result.stdout.strip()
      error_msg = stderr[:500] if stderr else f"exit code {result.returncode}"
      return False, elapsed, error_msg
    if not output_path.exists():
      return False, elapsed, "output file not created"
    return True, elapsed, ""
  except subprocess.TimeoutExpired:
    elapsed = time.perf_counter() - started
    return False, elapsed, f"timeout after {timeout_sec}s"
  except Exception as exc:
    elapsed = time.perf_counter() - started
    return False, elapsed, str(exc)


def run_official_evaluation(
  root: Path,
  command: list[str],
  output_path: Path,
  shots_a: int = 256,
  shots_b: int = 1024,
  seed: int = 42,
  timeout_sec: int = 60,
) -> bool:
  """Run the official nah-evaluate-submission evaluator.

  Returns True if evaluation completed successfully.
  """
  try:
    result = subprocess.run(
      [
        sys.executable, "-m",
        "noise_adaptive_hackathon.cli.evaluate_submission",
        "--root", str(root),
        "--shots-a", str(shots_a),
        "--shots-b", str(shots_b),
        "--seed", str(seed),
        "--timeout-sec", str(timeout_sec),
        "--out", str(output_path),
        "--command",
      ] + command,
      capture_output=True,
      text=True,
      timeout=timeout_sec * 15,  # Allow more time for full evaluation
    )
    if result.returncode != 0:
      print(f"Evaluator error: {result.stderr[:500]}")
      return False
    return True
  except Exception as exc:
    print(f"Evaluation failed: {exc}")
    return False


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Batch evaluate noise-adaptive SABRE compiler"
  )
  parser.add_argument(
    "--root", required=True, type=Path,
    help="Path to release_zh root directory",
  )
  parser.add_argument(
    "--compiler", required=True, type=str,
    help="Path to compiler script",
  )
  parser.add_argument(
    "--out", required=True, type=Path,
    help="Output directory for results",
  )
  parser.add_argument(
    "--benchmarks", type=str, nargs="*",
    help="Specific benchmarks to run (default: all public)",
  )
  parser.add_argument(
    "--track", type=str, choices=["track_a", "track_b", "both"],
    default="both",
    help="Which track(s) to evaluate (default: both)",
  )
  parser.add_argument(
    "--timeout-sec", type=int, default=60,
    help="Timeout per benchmark in seconds (default: 60)",
  )
  parser.add_argument(
    "--official-eval", action="store_true",
    help="Run official nah-evaluate-submission evaluator",
  )
  parser.add_argument(
    "--shots-a", type=int, default=256,
    help="Shots for Track A (default: 256)",
  )
  parser.add_argument(
    "--shots-b", type=int, default=1024,
    help="Shots for Track B (default: 1024)",
  )
  parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed (default: 42)",
  )
  args = parser.parse_args()

  root = find_release_root(args.root)
  print(f"Release root: {root}")

  # Determine tracks
  tracks = []
  if args.track in ("track_a", "both"):
    tracks.append("track_a")
  if args.track in ("track_b", "both"):
    tracks.append("track_b")

  # Load benchmarks
  all_benchmarks = load_benchmarks(root, tracks)
  if args.benchmarks:
    all_benchmarks = [
      b for b in all_benchmarks
      if b["name"] in args.benchmarks
    ]
  print(f"Found {len(all_benchmarks)} benchmarks")

  # Prepare command
  command = [sys.executable, args.compiler]
  if not Path(args.compiler).exists():
    print(f"ERROR: Compiler not found: {args.compiler}")
    return 1

  # Create output directory
  args.out.mkdir(parents=True, exist_ok=True)

  # Run each benchmark
  results: list[BenchmarkResult] = []
  total_started = time.perf_counter()

  for i, bench in enumerate(all_benchmarks):
    name = bench["name"]
    track = bench["track"]
    print(f"\n[{i+1}/{len(all_benchmarks)}] {name} ({track}) ... ", end="", flush=True)

    backend_path = find_backend_path(root, bench)
    if not backend_path.exists():
      print(f"SKIP: backend not found: {backend_path}")
      continue

    circuit_path = Path(bench["circuit_path"])
    output_dir = args.out / track
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_compiled.json"

    success, elapsed, error = run_compiler(
      command, circuit_path, backend_path, output_path,
      timeout_sec=args.timeout_sec,
    )

    result = BenchmarkResult(
      benchmark_name=name,
      track=track,
      success=success,
      compile_time_sec=elapsed,
      error_message=error,
    )
    results.append(result)

    status = " OK " if success else "FAIL"
    print(f"{status} ({elapsed:.2f}s)", end="")
    if error:
      print(f" — {error[:100]}", end="")
    print(flush=True)

  total_elapsed = time.perf_counter() - total_started

  # Summary
  success_count = sum(1 for r in results if r.success)
  print(f"\n{'='*60}")
  print(f"Summary: {success_count}/{len(results)} succeeded")
  print(f"Total time: {total_elapsed:.1f}s")

  # Save results
  results_path = args.out / "compilation_results.json"
  with open(results_path, "w", encoding="utf-8") as f:
    json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
  print(f"Results saved to {results_path}")

  # Run official evaluation if requested
  if args.official_eval:
    print(f"\n{'='*60}")
    print("Running official nah-evaluate-submission ...")
    summary_path = args.out / "public_summary.json"
    ok = run_official_evaluation(
      root, command, summary_path,
      shots_a=args.shots_a, shots_b=args.shots_b,
      seed=args.seed, timeout_sec=args.timeout_sec,
    )
    if ok and summary_path.exists():
      with open(summary_path) as f:
        summary = json.load(f)
      print(f"\nOfficial Evaluation Results:")
      print(f"  Track A score: {summary.get('track_a_score', 'N/A')}")
      print(f"  Track B score: {summary.get('track_b_score', 'N/A')}")
      print(f"  Final score:   {summary.get('final_score', 'N/A')}")
      print(f"Summary saved to {summary_path}")
    else:
      print("Official evaluation failed or summary not found")

  return 0 if success_count == len(results) else 1


if __name__ == "__main__":
  raise SystemExit(main())
