#!/usr/bin/env python3
"""
Noise-Adaptive SABRE Quantum Compiler
=====================================

Main entry point for the CCF Quantum Computing Programming Challenge
(量旋杯) noise-adaptive quantum compilation track.

This compiler implements a noise-adaptive variant of the SABRE
(SWAP-based BidiREctional heuristic search) algorithm for qubit
mapping and routing on NISQ devices.

Usage:
  python compiler.py \\
    --circuit path/to/circuit.json \\
    --backend path/to/backend.json \\
    --out path/to/compiled_circuit.json

The compiler writes two output files:
  1. compiled_circuit.json  — Cirq JSON compiled circuit
  2. compiled_circuit.layout.json — layout sidecar

Reference:
  Li, Ding, Xie. "Tackling the Qubit Mapping Problem for NISQ-Era
  Quantum Devices." ASPLOS 2019.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cirq

from noise_adaptive_hackathon.evaluator.equivalence import write_layout_sidecar
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from noise_adaptive_sabre import NoiseAdaptiveSabreCompiler


def build_parser() -> argparse.ArgumentParser:
  """Build command-line argument parser."""
  parser = argparse.ArgumentParser(
    description="Noise-Adaptive SABRE quantum circuit compiler"
  )
  parser.add_argument(
    "--circuit", required=True, type=Path,
    help="Path to input circuit JSON (Cirq format)",
  )
  parser.add_argument(
    "--backend", required=True, type=Path,
    help="Path to backend JSON (CompetitionBackend format)",
  )
  parser.add_argument(
    "--out", required=True, type=Path,
    help="Path for output compiled circuit JSON",
  )
  parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility (default: 42)",
  )
  parser.add_argument(
    "--num-layout-candidates", type=int, default=20,
    help="Number of initial layout candidates to try (default: 20)",
  )
  parser.add_argument(
    "--sabre-iterations", type=int, default=3,
    help="Number of SABRE reverse-traversal refinement iterations (default: 3)",
  )
  parser.add_argument(
    "--lookahead-window", type=int, default=10,
    help="SABRE look-ahead window size (default: 10)",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  """Main entry point for the compiler.

  Returns 0 on success, non-zero on failure.
  """
  args = build_parser().parse_args(argv)

  started = time.perf_counter()

  # Read input files
  try:
    circuit = cirq.read_json(args.circuit)
  except Exception as exc:
    print(f"ERROR: Failed to read circuit file: {exc}", flush=True)
    return 1

  try:
    backend = CompetitionBackend.parse_raw(
      args.backend.read_text(encoding="utf-8")
    )
  except Exception as exc:
    print(f"ERROR: Failed to read backend file: {exc}", flush=True)
    return 1

  # Auto-tune parameters based on circuit size
  num_qubits = len(set(q.x for q in circuit.all_qubits()))
  is_turbo = (num_qubits > 20)  # Turbo mode for large Track B circuits

  if is_turbo:
    # Turbo: massive search within time budget (currently ~8s of 180s available)
    num_candidates = max(args.num_layout_candidates, 80)
    sabre_iters = max(args.sabre_iterations, 10)
    lookahead = max(args.lookahead_window, 40)
  elif num_qubits <= 8:
    num_candidates = max(args.num_layout_candidates, 25)
    sabre_iters = max(args.sabre_iterations, 4)
    lookahead = args.lookahead_window
  elif num_qubits <= 16:
    num_candidates = max(args.num_layout_candidates, 30)
    sabre_iters = max(args.sabre_iterations, 5)
    lookahead = max(args.lookahead_window, 15)
  else:
    num_candidates = max(args.num_layout_candidates, 40)
    sabre_iters = max(args.sabre_iterations, 6)
    lookahead = max(args.lookahead_window, 25)

  use_sa = num_qubits > 12

  # Turbo mode: try multiple seeds, pick best
  if is_turbo:
    best_compiled = None
    best_layout = None
    best_cost = float('inf')
    turbo_seeds = [args.seed, args.seed + 100, args.seed + 200]

    for ts in turbo_seeds:
      try:
        compiler = NoiseAdaptiveSabreCompiler(
          seed=ts,
          num_layout_candidates=num_candidates,
          sabre_iterations=sabre_iters,
          lookahead_window=lookahead,
          use_simulated_annealing=use_sa,
          use_resynthesis=False,
          use_parallel=False,
        )
        compiled, initial_layout = compiler.compile(circuit, backend)
        q2 = sum(1 for o in compiled.all_operations()
                 if len(o.qubits) == 2
                 and not isinstance(o.gate, cirq.MeasurementGate))
        # Prefer result with fewer 2Q gates
        if q2 < best_cost or (q2 == best_cost and len(compiled) < len(best_compiled or [])):
          best_cost = q2
          best_compiled = compiled
          best_layout = initial_layout
      except Exception:
        continue

    if best_compiled is None:
      raise RuntimeError("all turbo seeds failed")
    compiled, initial_layout = best_compiled, best_layout
  else:
    try:
      compiler = NoiseAdaptiveSabreCompiler(
        seed=args.seed,
        num_layout_candidates=num_candidates,
        sabre_iterations=sabre_iters,
        lookahead_window=lookahead,
        use_simulated_annealing=use_sa,
        use_resynthesis=False,
        use_parallel=False,
      )
      compiled, initial_layout = compiler.compile(circuit, backend)
    except Exception as exc:
      print(f"ERROR: Compilation failed: {exc}", flush=True)
      import traceback
      traceback.print_exc()
      return 1

  elapsed = time.perf_counter() - started

  # Write output files
  try:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cirq.to_json(compiled, args.out)
    write_layout_sidecar(args.out, initial_layout)
  except Exception as exc:
    print(f"ERROR: Failed to write output files: {exc}", flush=True)
    return 1

  # Summary
  num_2q = sum(
    1 for op in compiled.all_operations()
    if len(op.qubits) == 2 and not isinstance(op.gate, cirq.MeasurementGate)
  )
  depth = len(compiled)
  print(
    f"Compilation complete: {depth} moments, "
    f"{num_2q} two-qubit gates, "
    f"{elapsed:.3f}s",
    flush=True,
  )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
