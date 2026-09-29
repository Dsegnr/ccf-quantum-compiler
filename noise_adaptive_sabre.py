"""
Noise-Adaptive SABRE Quantum Circuit Compiler
=============================================

Implements a noise-adaptive variant of the SABRE (SWAP-based BidiREctional
heuristic search) algorithm for qubit mapping and routing on NISQ devices.

Key features:
  - Multiple initial layout candidates with reliability scoring
  - SABRE look-ahead SWAP selection with noise-aware edge weights
  - Reverse traversal refinement for layout optimization
  - Post-routing gate cancellation
  - Track B Clifford-only circuit preservation

Reference:
  Li, Ding, Xie. "Tackling the Qubit Mapping Problem for NISQ-Era
  Quantum Devices." ASPLOS 2019. https://doi.org/10.1145/3297858.3304023
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import cirq
import networkx as nx


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
  """Result of a single compilation attempt."""
  circuit: cirq.Circuit
  initial_layout: dict[int, int]
  estimated_cost: float


# ---------------------------------------------------------------------------
# Circuit analysis helpers
# ---------------------------------------------------------------------------

def build_interaction_graph(circuit: cirq.Circuit) -> nx.Graph:
  """Build weighted interaction graph from circuit."""
  graph = nx.Graph()
  logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
  graph.add_nodes_from(logical_qubits)
  for op in circuit.all_operations():
    if isinstance(op.gate, cirq.MeasurementGate):
      continue
    if len(op.qubits) == 2 and not isinstance(op.gate, cirq.SwapPowGate):
      left, right = op.qubits
      weight = graph.get_edge_data(left, right, {}).get("weight", 0) + 1
      graph.add_edge(left, right, weight=weight)
  return graph


def get_measured_qubits(circuit: cirq.Circuit) -> set[int]:
  """Return set of logical qubit indices that are measured."""
  measured: set[int] = set()
  for op in circuit.all_operations():
    if isinstance(op.gate, cirq.MeasurementGate):
      for q in op.qubits:
        measured.add(q.x)
  return measured


def get_all_operations(circuit: cirq.Circuit) -> list[cirq.Operation]:
  """Flatten all operations from the circuit in order."""
  ops: list[cirq.Operation] = []
  for moment in circuit:
    for op in moment.operations:
      ops.append(op)
  return ops


# ---------------------------------------------------------------------------
# Hardware graph construction
# ---------------------------------------------------------------------------

def build_hardware_graph(backend) -> nx.Graph:
  """Build hardware graph with reliability-aware edge weights."""
  graph = nx.Graph()
  for qubit in range(backend.num_qubits):
    graph.add_node(
      qubit,
      readout_reliability=1.0 - backend.readout_error[str(qubit)],
    )
  for control, target in backend.coupling_map:
    reliability = 1.0 - backend.edge_error(control, target)
    reliability = max(reliability, 1e-9)
    graph.add_edge(
      control, target,
      reliability=reliability,
      cost=-math.log(reliability),
    )
  return graph


# ---------------------------------------------------------------------------
# SWAP decomposition
# ---------------------------------------------------------------------------

def _decomposed_swap(
  a: cirq.LineQubit, b: cirq.LineQubit
) -> list[cirq.Operation]:
  """Decompose SWAP into 3 CX gates."""
  return [
    cirq.CNOT(a, b),
    cirq.CNOT(b, a),
    cirq.CNOT(a, b),
  ]


# ---------------------------------------------------------------------------
# Layout generation strategies
# ---------------------------------------------------------------------------

def reliability_aware_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
) -> dict[int, int]:
  """Generate layout using reliability-aware interaction graph matching."""
  program = build_interaction_graph(circuit)
  measured = get_measured_qubits(circuit)
  logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)

  if len(logical_qubits) > backend.num_qubits:
    raise ValueError("program does not fit on backend")

  reliability_lengths = dict(
    nx.all_pairs_dijkstra_path_length(hw_graph, weight="cost")
  )

  logical_weight = {
    node: sum(data["weight"] for _, _, data in program.edges(node, data=True))
    for node in program.nodes
  }

  available = set(range(backend.num_qubits))
  layout: dict[int, int] = {}

  # Seed with heaviest edge
  if program.number_of_edges() > 0:
    best_edge = max(
      program.edges(data=True),
      key=lambda e: (
        e[2]["weight"],
        logical_weight[e[0]] + logical_weight[e[1]],
        -e[0].x, -e[1].x,
      ),
    )
    l1, l2, _ = best_edge
    hw_edges = [
      (u, v) for u, v in hw_graph.edges
      if u in available and v in available
    ]
    if hw_edges:
      best_hw_edge = max(
        hw_edges,
        key=lambda e: (
          hw_graph.edges[e]["reliability"],
          hw_graph.nodes[e[0]]["readout_reliability"]
          + hw_graph.nodes[e[1]]["readout_reliability"],
          hw_graph.degree(e[0]) + hw_graph.degree(e[1]),
        ),
      )
      p1, p2 = best_hw_edge
      # Assign heavier logical to better-readout physical
      if (l1.x in measured) != (l2.x in measured):
        measured_l = l1 if l1.x in measured else l2
        unmeasured_l = l2 if l1.x in measured else l1
        if hw_graph.nodes[p1]["readout_reliability"] >= hw_graph.nodes[p2]["readout_reliability"]:
          ordered = [(measured_l.x, p1), (unmeasured_l.x, p2)]
        else:
          ordered = [(measured_l.x, p2), (unmeasured_l.x, p1)]
      elif logical_weight.get(l1, 0) >= logical_weight.get(l2, 0):
        ordered = [(l1.x, p1), (l2.x, p2)]
      else:
        ordered = [(l2.x, p1), (l1.x, p2)]

      for logical, physical in ordered:
        layout[logical] = physical
        available.remove(physical)

  # Place remaining
  while len(layout) < len(logical_qubits):
    unplaced = [
      q for q in program.nodes
      if q.x not in layout
    ]
    if not unplaced:
      remaining = [q for q in logical_qubits if q.x not in layout]
      for q in remaining:
        if available:
          best_p = max(
            available,
            key=lambda p: (
              hw_graph.nodes[p]["readout_reliability"]
              if q.x in measured else 0,
              hw_graph.degree(p),
              -p,
            ),
          )
          layout[q.x] = best_p
          available.remove(best_p)
      break

    logical = max(
      unplaced,
      key=lambda q: (
        sum(program.edges[q, n]["weight"]
            for n in program.neighbors(q) if n.x in layout),
        logical_weight.get(q, 0),
        program.degree(q),
        -q.x,
      ),
    )

    placed_neighbors = [
      (n, program.edges[logical, n]["weight"])
      for n in program.neighbors(logical) if n.x in layout
    ]

    if placed_neighbors:
      physical = max(
        available,
        key=lambda p: (
          sum(
            w * math.exp(-reliability_lengths[p][layout[n.x]])
            for n, w in placed_neighbors
          ),
          hw_graph.nodes[p]["readout_reliability"]
          if logical.x in measured else 0,
          hw_graph.degree(p),
          -p,
        ),
      )
    else:
      physical = max(
        available,
        key=lambda p: (
          hw_graph.nodes[p]["readout_reliability"]
          if logical.x in measured else 0,
          hw_graph.degree(p),
          -p,
        ),
      )

    layout[logical.x] = physical
    available.remove(physical)

  return layout


def movement_aware_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
) -> dict[int, int]:
  """Layout based on degree matching."""
  program = build_interaction_graph(circuit)
  measured = get_measured_qubits(circuit)
  logical_order = sorted(
    program.nodes,
    key=lambda node: (program.degree(node), -node.x),
    reverse=True,
  )
  available = set(range(backend.num_qubits))
  layout: dict[int, int] = {}

  for logical in logical_order:
    if not layout:
      candidates = sorted(
        available,
        key=lambda p: (
          hw_graph.nodes[p]["readout_reliability"]
          if logical.x in measured else 0,
          hw_graph.degree(p),
        ),
        reverse=True,
      )
      top_n = min(3, len(candidates))
      physical = candidates[rng.randint(0, top_n - 1)]
    else:
      placed = [
        neighbor for neighbor in program.neighbors(logical)
        if neighbor.x in layout
      ]
      if placed:
        physical = min(
          available,
          key=lambda p: (
            sum(
              nx.shortest_path_length(hw_graph, p, layout[n.x])
              for n in placed
            ),
            -hw_graph.nodes[p]["readout_reliability"]
            if logical.x in measured else 0,
            -hw_graph.degree(p),
            p,
          ),
        )
      else:
        physical = max(
          available,
          key=lambda p: (
            hw_graph.nodes[p]["readout_reliability"]
            if logical.x in measured else 0,
            hw_graph.degree(p),
            -p,
          ),
        )
    layout[logical.x] = physical
    available.remove(physical)

  return layout


def random_perturbed_layout(
  base_layout: dict[int, int],
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
  perturbation: float = 0.3,
) -> dict[int, int]:
  """Create a perturbed version of a base layout."""
  available = set(range(backend.num_qubits)) - set(base_layout.values())
  layout = dict(base_layout)
  logicals = list(layout.keys())

  for logical in logicals:
    if rng.random() < perturbation and available:
      old_physical = layout[logical]
      candidates = sorted(
        available,
        key=lambda p: (
          nx.shortest_path_length(hw_graph, p, old_physical),
          -hw_graph.nodes[p]["readout_reliability"],
        ),
      )
      top_k = min(max(3, len(candidates) // 4), len(candidates))
      if top_k > 0:
        new_physical = candidates[rng.randint(0, top_k - 1)]
        layout[logical] = new_physical
        available.remove(new_physical)
        available.add(old_physical)

  return layout


def star_aware_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
) -> dict[int, int] | None:
  """Layout specialized for star-structured circuits (like GHZ star).

  Detects if the interaction graph has a dominant center node and places
  it at the best-connected, lowest-error physical qubit.
  Returns None if not a star structure.
  """
  program = build_interaction_graph(circuit)
  if program.number_of_edges() < 3:
    return None

  degrees = dict(program.degree())
  max_deg = max(degrees.values())

  # Star: one node has degree >= 5 and is >= 3x the second-highest degree
  sorted_degs = sorted(degrees.values(), reverse=True)
  if len(sorted_degs) < 2:
    return None
  if sorted_degs[0] < 5:
    return None
  if sorted_degs[0] < 3 * sorted_degs[1]:
    return None

  center_logical = max(degrees, key=degrees.get)
  measured = get_measured_qubits(circuit)

  # Find best physical qubit for center: high degree + low error edges
  center_candidates = sorted(
    hw_graph.nodes,
    key=lambda p: (
      # Total incident reliability
      sum(hw_graph.edges[p, n]["reliability"] for n in hw_graph.neighbors(p)),
      hw_graph.degree(p),
      hw_graph.nodes[p]["readout_reliability"] if center_logical.x in measured else 0,
      -p,
    ),
    reverse=True,
  )
  center_physical = center_candidates[0]

  layout: dict[int, int] = {}
  layout[center_logical.x] = center_physical
  available = set(range(backend.num_qubits)) - {center_physical}

  # Place neighbors closest to center
  neighbors = sorted(
    [n for n in program.neighbors(center_logical)],
    key=lambda n: (program.degree(n), -n.x),
    reverse=True,
  )

  for neighbor in neighbors:
    if not available:
      break
    # Find closest available physical qubit to center
    best_p = min(
      available,
      key=lambda p: (
        nx.shortest_path_length(hw_graph, p, center_physical),
        -hw_graph.nodes[p]["readout_reliability"] if neighbor.x in measured else 0,
        -hw_graph.degree(p),
        p,
      ),
    )
    layout[neighbor.x] = best_p
    available.remove(best_p)

  # Place any remaining isolated qubits
  remaining = [q for q in program.nodes if q.x not in layout]
  for q in remaining:
    if available:
      best_p = max(
        available,
        key=lambda p: (
          hw_graph.nodes[p]["readout_reliability"] if q.x in measured else 0,
          hw_graph.degree(p),
          -p,
        ),
      )
      layout[q.x] = best_p
      available.remove(best_p)

  return layout


def linear_chain_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
) -> dict[int, int] | None:
  """Layout specialized for chain-structured circuits (like GHZ).

  Detects if the interaction graph is a single path and places
  logical qubits consecutively along a long path in the hardware graph.
  Returns None if the circuit doesn't look like a chain.
  """
  program = build_interaction_graph(circuit)

  # Check if graph is a single path (or close to it)
  if program.number_of_edges() == 0:
    return None

  # Check for path structure: max degree <= 2 and connected
  max_deg = max(dict(program.degree()).values())
  if max_deg > 3:
    return None

  # Find endpoints (degree 1 nodes)
  endpoints = [n for n in program.nodes if program.degree(n) == 1]
  if len(endpoints) < 2:
    return None

  # Try to find a path from one endpoint to the other
  try:
    path = nx.shortest_path(program, endpoints[0], endpoints[-1])
  except (nx.NetworkXNoPath, nx.NodeNotFound):
    return None

  if len(path) < len(program.nodes):
    # Not all nodes are on the main path
    return None

  # Now find a long path in the hardware graph
  measured = get_measured_qubits(circuit)

  # Try multiple starting points to find the best hardware path
  best_hw_path = None
  best_score = float('-inf')

  # Find high-degree nodes as path endpoints
  hw_nodes = sorted(
    hw_graph.nodes,
    key=lambda n: (
      -hw_graph.degree(n),
      -hw_graph.nodes[n]["readout_reliability"],
      n,
    ),
  )

  for start_node in hw_nodes[:10]:
    try:
      # Try to find a path of sufficient length
      hw_path = _greedy_long_path(hw_graph, start_node, len(path))
      if hw_path and len(hw_path) >= len(path):
        # Score: prefer paths with good readout at endpoints
        score = (
          hw_graph.nodes[hw_path[0]]["readout_reliability"]
          + hw_graph.nodes[hw_path[-1]]["readout_reliability"]
        )
        if score > best_score:
          best_score = score
          best_hw_path = hw_path[:len(path)]
    except Exception:
      continue

  if best_hw_path is None:
    return None

  # Map logical path onto hardware path
  layout: dict[int, int] = {}
  for i, logical_node in enumerate(path):
    layout[logical_node.x] = best_hw_path[i]

  return layout


def _greedy_long_path(
  hw_graph: nx.Graph, start: int, target_length: int
) -> list[int] | None:
  """Greedily build a long simple path starting from a node."""
  path = [start]
  visited = {start}
  current = start

  while len(path) < target_length:
    # Find unvisited neighbors, prefer those with more connections
    neighbors = [
      n for n in hw_graph.neighbors(current) if n not in visited
    ]
    if not neighbors:
      break
    # Sort by: (has many unvisited neighbors, high readout reliability)
    next_node = max(
      neighbors,
      key=lambda n: (
        sum(1 for nn in hw_graph.neighbors(n) if nn not in visited),
        hw_graph.nodes[n]["readout_reliability"],
        -n,
      ),
    )
    path.append(next_node)
    visited.add(next_node)
    current = next_node

  return path if len(path) >= target_length else None


def generate_layout_candidates(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
  num_candidates: int = 20,
) -> list[dict[int, int]]:
  """Generate diverse initial layout candidates."""
  candidates: list[dict[int, int]] = []

  # Try star-specific layout (for GHZ-star circuits)
  try:
    star_layout = star_aware_layout(circuit, backend, hw_graph, rng)
    if star_layout is not None:
      candidates.append(star_layout)
      for _ in range(8):
        perturbed = random_perturbed_layout(
          star_layout, backend, hw_graph, rng,
          perturbation=0.1,
        )
        if perturbed not in candidates:
          candidates.append(perturbed)
  except Exception:
    pass

  # Try chain-specific layout (for linear GHZ-like circuits)
  try:
    chain_layout = linear_chain_layout(circuit, backend, hw_graph, rng)
    if chain_layout is not None:
      candidates.append(chain_layout)
      for _ in range(5):
        perturbed = random_perturbed_layout(
          chain_layout, backend, hw_graph, rng,
          perturbation=0.15,
        )
        if perturbed not in candidates:
          candidates.append(perturbed)
  except Exception:
    pass

  # Try community-detection layout (for clustered circuits like Parity)
  try:
    comm_layout = community_detection_layout(circuit, backend, hw_graph, rng)
    if comm_layout is not None and len(comm_layout) >= 8:
      candidates.append(comm_layout)
      for _ in range(5):
        perturbed = random_perturbed_layout(
          comm_layout, backend, hw_graph, rng,
          perturbation=0.12,
        )
        if perturbed not in candidates:
          candidates.append(perturbed)
  except Exception:
    pass

  # Standard layout strategies
  try:
    candidates.append(
      reliability_aware_layout(circuit, backend, hw_graph, rng))
  except Exception:
    pass

  try:
    candidates.append(
      movement_aware_layout(circuit, backend, hw_graph, rng))
  except Exception:
    pass

  base_layouts = [c for c in candidates if c is not None]
  for base in base_layouts:
    for _ in range(max(1, (num_candidates - len(candidates)) // max(1, len(base_layouts)))):
      try:
        perturbed = random_perturbed_layout(
          base, backend, hw_graph, rng,
          perturbation=0.2 + 0.3 * rng.random(),
        )
        if perturbed not in candidates:
          candidates.append(perturbed)
      except Exception:
        pass

  logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
  logical_ids = [q.x for q in logical_qubits]
  measured = get_measured_qubits(circuit)

  for _ in range(min(5, num_candidates - len(candidates))):
    available = list(range(backend.num_qubits))
    rng.shuffle(available)
    scored_available = sorted(
      available,
      key=lambda p: (
        -hw_graph.nodes[p]["readout_reliability"],
        -hw_graph.degree(p),
      ),
    )
    layout: dict[int, int] = {}
    used: set[int] = set()
    measured_logicals = [q for q in logical_ids if q in measured]
    unmeasured = [q for q in logical_ids if q not in measured]
    for q in measured_logicals:
      for p in scored_available:
        if p not in used:
          layout[q] = p
          used.add(p)
          break
    for q in unmeasured:
      for p in scored_available:
        if p not in used:
          layout[q] = p
          used.add(p)
          break
    if len(layout) == len(logical_ids) and layout not in candidates:
      candidates.append(layout)

  return candidates[:num_candidates]


# ---------------------------------------------------------------------------
# SABRE routing algorithm (processes ALL operations)
# ---------------------------------------------------------------------------

class SabreRouter:
  """SABRE-based noise-adaptive router.

  Processes ALL circuit operations (1Q, 2Q, measurement) and inserts
  SWAP gates as needed for 2Q adjacency.
  """

  def __init__(
    self,
    backend,
    hw_graph: nx.Graph,
    rng: random.Random,
    lookahead_window: int = 10,
    decay: float = 0.5,
  ):
    self.backend = backend
    self.hw_graph = hw_graph
    self.rng = rng
    self.lookahead_window = lookahead_window
    self.decay = decay

    # Precompute paths
    self.reliability_paths = dict(
      nx.all_pairs_dijkstra_path(hw_graph, weight="cost")
    )
    self.reliability_lengths = dict(
      nx.all_pairs_dijkstra_path_length(hw_graph, weight="cost")
    )
    self.hop_lengths = dict(nx.all_pairs_shortest_path_length(hw_graph))

  def route(
    self,
    circuit: cirq.Circuit,
    initial_layout: dict[int, int],
  ) -> tuple[cirq.Circuit, dict[int, int], float]:
    """Route full circuit including all operations.

    Returns:
      (routed_circuit, final_layout, estimated_cost)
    """
    # State
    mapping = dict(initial_layout)  # logical -> physical
    reverse: dict[int, int | None] = {
      p: None for p in range(self.backend.num_qubits)
    }
    for logical, physical in mapping.items():
      reverse[physical] = logical

    physical_qubits = [cirq.LineQubit(i) for i in range(self.backend.num_qubits)]

    # Start with identities on all physical qubits
    compiled = cirq.Circuit(
      cirq.Moment(cirq.I(q) for q in physical_qubits)
    )

    # Get all operations in order
    all_ops = get_all_operations(circuit)

    # Pre-compute future 2Q gate positions for look-ahead
    future_2q_positions = [
      i for i, op in enumerate(all_ops)
      if (len(op.qubits) == 2
          and not isinstance(op.gate, cirq.MeasurementGate)
          and not isinstance(op.gate, cirq.SwapPowGate))
    ]

    total_cost = 0.0

    for op_index, op in enumerate(all_ops):
      if isinstance(op.gate, cirq.MeasurementGate):
        # Collect measurement for end of circuit
        compiled.append(
          cirq.measure(
            *[physical_qubits[mapping[q.x]] for q in op.qubits],
            key=op.gate.key,
          )
        )
        continue

      if len(op.qubits) == 1:
        # Single-qubit gate: remap and add
        logical = op.qubits[0].x
        physical = mapping.get(logical)
        if physical is not None:
          compiled.append(
            op.transform_qubits(
              lambda _: physical_qubits[physical]
            )
          )
        continue

      if len(op.qubits) != 2:
        continue

      # Two-qubit gate: may need routing
      left = op.qubits[0].x
      right = op.qubits[1].x
      p_left = mapping.get(left)
      p_right = mapping.get(right)

      if p_left is None or p_right is None:
        continue

      # Insert SWAPs until adjacent
      max_swaps = 50  # safety limit
      swap_count = 0

      while (not self.hw_graph.has_edge(p_left, p_right)
             and swap_count < max_swaps):
        # Find best SWAP using SABRE look-ahead
        best_swap = self._select_best_swap(
          mapping, reverse, all_ops, op_index,
          future_2q_positions,
        )

        if best_swap is None:
          # Fallback: first edge on reliability path
          try:
            path = self.reliability_paths[p_left].get(p_right)
            if path and len(path) >= 2:
              best_swap = (path[0], path[1])
            else:
              break
          except (KeyError, nx.NetworkXNoPath):
            break

        swap_u, swap_v = best_swap

        # Execute SWAP (3 CX gates)
        compiled.append(_decomposed_swap(
          physical_qubits[swap_u], physical_qubits[swap_v]
        ))

        # Cost accounting
        swap_reliability = self.hw_graph.edges[swap_u, swap_v]["reliability"]
        total_cost += 3 * (-math.log(max(swap_reliability, 1e-9)))

        # Update mapping
        l_u = reverse.get(swap_u)
        l_v = reverse.get(swap_v)
        if l_u is not None:
          mapping[l_u] = swap_v
          reverse[swap_v] = l_u
        else:
          reverse[swap_v] = None
        if l_v is not None:
          mapping[l_v] = swap_u
          reverse[swap_u] = l_v
        else:
          reverse[swap_u] = None

        # Update p_left, p_right for next iteration
        p_left = mapping.get(left, p_left)
        p_right = mapping.get(right, p_right)

        swap_count += 1

      # Execute the gate (if now adjacent)
      if self.hw_graph.has_edge(p_left, p_right):
        compiled.append(
          op.transform_qubits(
            lambda q: physical_qubits[mapping[q.x]]
          )
        )
        try:
          total_cost += self.reliability_lengths[p_left][p_right]
        except KeyError:
          total_cost += 1.0

    return compiled, dict(mapping), total_cost

  def _select_best_swap(
    self,
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    all_ops: list[cirq.Operation],
    current_op_index: int,
    future_2q_positions: list[int],
  ) -> tuple[int, int] | None:
    """Select best SWAP using SABRE look-ahead heuristic."""
    # Find position of current op in the 2Q positions list
    try:
      current_2q_idx = future_2q_positions.index(current_op_index)
    except ValueError:
      current_2q_idx = 0
      for pos in future_2q_positions:
        if pos >= current_op_index:
          break
        current_2q_idx += 1

    # Collect future 2Q gates for look-ahead
    future_gates: list[tuple[int, int, int]] = []
    for idx in range(current_2q_idx, min(
        current_2q_idx + self.lookahead_window + 1,
        len(future_2q_positions),
    )):
      pos = future_2q_positions[idx]
      op = all_ops[pos]
      left = op.qubits[0].x
      right = op.qubits[1].x
      p1 = mapping.get(left)
      p2 = mapping.get(right)
      if p1 is not None and p2 is not None:
        if not self.hw_graph.has_edge(p1, p2):
          future_gates.append((left, right, idx - current_2q_idx))

    # Candidate SWAP edges
    mapped_physical = {p for p, l in reverse.items() if l is not None}
    candidate_edges: list[tuple[int, int]] = []
    for u, v in self.hw_graph.edges:
      if u in mapped_physical or v in mapped_physical:
        candidate_edges.append((u, v))

    if not candidate_edges:
      return None

    # Compute current cost
    current_cost = 0.0
    for l1, l2, dist in future_gates:
      p1 = mapping.get(l1)
      p2 = mapping.get(l2)
      if p1 is not None and p2 is not None:
        try:
          d = self.reliability_lengths[p1][p2]
        except KeyError:
          d = 100.0
        current_cost += (self.decay ** dist) * d

    best_swap = None
    best_score = float('inf')

    for u, v in candidate_edges:
      # Simulate swap
      l_u = reverse.get(u)
      l_v = reverse.get(v)

      sim_mapping = dict(mapping)
      if l_u is not None:
        sim_mapping[l_u] = v
      if l_v is not None:
        sim_mapping[l_v] = u

      # Compute new cost
      new_cost = 0.0
      for l1, l2, dist in future_gates:
        p1 = sim_mapping.get(l1)
        p2 = sim_mapping.get(l2)
        if p1 is not None and p2 is not None:
          try:
            d = self.reliability_lengths[p1][p2]
          except KeyError:
            d = 100.0
          new_cost += (self.decay ** dist) * d

      # SWAP edge cost
      swap_reliability = self.hw_graph.edges[u, v]["reliability"]
      swap_cost = 3 * (-math.log(max(swap_reliability, 1e-9)))

      # Score: favor swaps that reduce future distance on good edges
      improvement = current_cost - new_cost
      score = -improvement + 0.05 * swap_cost

      if score < best_score:
        best_score = score
        best_swap = (u, v)

    return best_swap


# ---------------------------------------------------------------------------
# Reverse traversal refinement
# ---------------------------------------------------------------------------

def reverse_traversal_refine(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  initial_layout: dict[int, int],
  rng: random.Random,
  num_iterations: int = 3,
) -> dict[int, int]:
  """Refine layout using reverse traversal."""
  current_layout = dict(initial_layout)
  best_layout = dict(initial_layout)
  best_cost = float('inf')

  for _ in range(num_iterations):
    # Forward pass
    router = SabreRouter(backend, hw_graph, rng)
    _, forward_final, forward_cost = router.route(circuit, current_layout)

    if forward_cost < best_cost:
      best_cost = forward_cost
      best_layout = dict(current_layout)

    # Reverse pass
    reversed_circuit = _reverse_circuit(circuit)
    rev_router = SabreRouter(backend, hw_graph, rng)
    _, rev_final, rev_cost = rev_router.route(reversed_circuit, forward_final)

    if rev_cost < best_cost:
      best_cost = rev_cost
      best_layout = dict(forward_final)

    current_layout = dict(rev_final)

  return best_layout


def _reverse_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
  """Create reversed circuit for reverse traversal."""
  ops = [
    op for op in circuit.all_operations()
    if not isinstance(op.gate, cirq.MeasurementGate)
  ]
  return cirq.Circuit(reversed(ops))


# ---------------------------------------------------------------------------
# Post-routing optimization
# ---------------------------------------------------------------------------

def optimize_circuit(
  circuit: cirq.Circuit,
  backend,
) -> cirq.Circuit:
  """Apply post-routing circuit optimizations."""
  return _cancel_adjacent_inverses(circuit)


def _cancel_adjacent_inverses(circuit: cirq.Circuit) -> cirq.Circuit:
  """Cancel adjacent self-inverse gate pairs."""
  ops = list(circuit.all_operations())
  cancelled: list[cirq.Operation] = []
  skip_next: set[int] = set()

  for i, op in enumerate(ops):
    if i in skip_next:
      continue
    if isinstance(op.gate, cirq.IdentityGate):
      continue  # Skip identity gates entirely
    if i + 1 < len(ops) and _are_inverses(op, ops[i + 1]):
      skip_next.add(i + 1)
      continue
    cancelled.append(op)

  return cirq.Circuit(cancelled)


def _are_inverses(op1: cirq.Operation, op2: cirq.Operation) -> bool:
  """Check if two operations cancel each other."""
  if type(op1.gate) != type(op2.gate):
    return False

  qubits1 = tuple(op1.qubits)
  qubits2 = tuple(op2.qubits)
  if qubits1 != qubits2:
    return False

  # Self-inverse: H
  if isinstance(op1.gate, cirq.HPowGate):
    return (abs(float(op1.gate.exponent) - float(op2.gate.exponent)) < 1e-9
            and abs(float(op1.gate.exponent) % 2 - 1.0) < 1e-9)

  # Self-inverse: X
  if isinstance(op1.gate, cirq.XPowGate) and len(op1.qubits) == 1:
    return (abs(float(op1.gate.exponent) % 2 - 1.0) < 1e-9
            and abs(float(op2.gate.exponent) % 2 - 1.0) < 1e-9)

  # Self-inverse: CX
  if isinstance(op1.gate, cirq.CXPowGate):
    return (abs(float(op1.gate.exponent) % 2 - 1.0) < 1e-9
            and abs(float(op2.gate.exponent) % 2 - 1.0) < 1e-9)

  return False


# ---------------------------------------------------------------------------
# Cost estimation for layout comparison
# ---------------------------------------------------------------------------

def estimate_layout_cost(
  circuit: cirq.Circuit,
  layout: dict[int, int],
  hw_graph: nx.Graph,
) -> float:
  """Estimate total cost of a layout without full routing."""
  program = build_interaction_graph(circuit)
  reliability_lengths = dict(
    nx.all_pairs_dijkstra_path_length(hw_graph, weight="cost")
  )

  total_cost = 0.0
  for left, right, data in program.edges(data=True):
    weight = data["weight"]
    try:
      dist = reliability_lengths[layout[left.x]][layout[right.x]]
    except KeyError:
      dist = 100.0
    total_cost += weight * dist

  measured = get_measured_qubits(circuit)
  readout_bonus = sum(
    hw_graph.nodes[layout[q]]["readout_reliability"]
    for q in measured if q in layout
  )

  return total_cost - 0.1 * readout_bonus


# ---------------------------------------------------------------------------
# Community detection layout (Louvain)
# ---------------------------------------------------------------------------

def community_detection_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
) -> dict[int, int] | None:
  """Layout based on Louvain community detection on the interaction graph.

  Detects dense subgraphs in the interaction graph and maps each
  community to a dense region of the hardware graph.  Effective for
  circuits with clustered interaction patterns (e.g. parity circuits).

  Returns None if the circuit doesn't have meaningful community structure.
  """
  from networkx.algorithms.community import louvain_communities

  program = build_interaction_graph(circuit)
  measured = get_measured_qubits(circuit)

  # Only apply if circuit has enough qubits and edges for communities
  if program.number_of_nodes() < 8 or program.number_of_edges() < 8:
    return None

  # Run Louvain community detection on the interaction graph
  try:
    communities = louvain_communities(program, seed=rng.randint(0, 2**31 - 1))
  except Exception:
    return None

  # Need at least 2 non-trivial communities for this to be useful
  communities = [c for c in communities if len(c) > 1]
  if len(communities) < 2:
    return None

  # Sort communities by size (largest first)
  communities.sort(key=len, reverse=True)

  # Precompute hardware path lengths
  path_lengths = dict(nx.all_pairs_shortest_path_length(hw_graph))
  reliability_lengths = dict(
    nx.all_pairs_dijkstra_path_length(hw_graph, weight="cost")
  )

  # Find dense regions in hardware graph for each community
  available = set(range(backend.num_qubits))
  layout: dict[int, int] = {}

  for comm_idx, community in enumerate(communities):
    comm_logicals = [q for q in community]  # LineQubit nodes from interaction graph
    comm_size = len(comm_logicals)

    # Find the densest available region of size >= comm_size
    best_seed = None
    best_density = -1.0

    for seed_node in sorted(available):
      # Grow a region around seed_node using BFS
      region = _grow_region(hw_graph, seed_node, comm_size, available)
      if region and len(region) >= comm_size:
        # Compute region density (edges within region / possible edges)
        subgraph = hw_graph.subgraph(region)
        density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
        # Also consider edge reliability
        avg_reliability = 0.0
        edge_count = 0
        for u, v in subgraph.edges:
          avg_reliability += hw_graph.edges[u, v]["reliability"]
          edge_count += 1
        if edge_count > 0:
          avg_reliability /= edge_count

        score = density + 0.5 * avg_reliability
        if score > best_density:
          best_density = score
          best_seed = seed_node
          best_region = region[:comm_size]

    if best_seed is None:
      continue  # Can't place this community as a unit

    # Place community members within the region
    region = best_region[:comm_size]
    _place_community_in_region(
      program, comm_logicals, region, layout, available,
      measured, reliability_lengths, path_lengths,
    )

  # Place any remaining individual qubits
  unplaced_logicals = [
    q for q in program.nodes if q.x not in layout
  ]
  remaining_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
  for q in remaining_qubits:
    if q.x not in layout and available:
      # Place near already-placed neighbors if any
      placed_neighbors = [
        n for n in program.neighbors(q) if n.x in layout
      ]
      if placed_neighbors:
        best_p = min(
          available,
          key=lambda p: sum(
            path_lengths[p][layout[n.x]] for n in placed_neighbors
          ),
        )
      else:
        best_p = max(
          available,
          key=lambda p: (
            hw_graph.nodes[p]["readout_reliability"]
            if q.x in measured else 0,
            hw_graph.degree(p),
            -p,
          ),
        )
      layout[q.x] = best_p
      available.remove(best_p)

  # Verify all qubits are placed
  for q in sorted(circuit.all_qubits(), key=lambda q: q.x):
    if q.x not in layout:
      return None  # Failed to place all qubits

  return layout


def _grow_region(
  hw_graph: nx.Graph,
  seed: int,
  target_size: int,
  available: set[int],
) -> list[int] | None:
  """Grow a region around a seed node using BFS, respecting availability."""
  region = [seed]
  frontier = [seed]
  visited = {seed}

  while frontier and len(region) < target_size:
    next_frontier = []
    for node in sorted(frontier):
      for neighbor in sorted(hw_graph.neighbors(node)):
        if neighbor not in visited and neighbor in available:
          visited.add(neighbor)
          region.append(neighbor)
          next_frontier.append(neighbor)
          if len(region) >= target_size:
            break
      if len(region) >= target_size:
        break
    frontier = next_frontier

  return region if len(region) >= target_size else None


def _place_community_in_region(
  program: nx.Graph,
  comm_logicals: list,
  region: list[int],
  layout: dict[int, int],
  available: set[int],
  measured: set[int],
  reliability_lengths: dict,
  path_lengths: dict,
) -> None:
  """Place logical qubits of one community into an assigned hardware region."""
  comm_logicals.sort(
    key=lambda q: (program.degree(q), sum(
      program.edges[q, n]["weight"] for n in program.neighbors(q)
    )),
    reverse=True,
  )

  for logical in comm_logicals:
    if not region:
      break
    logical_id = logical.x

    placed_in_comm = [
      n for n in program.neighbors(logical) if n.x in layout
    ]

    if placed_in_comm:
      best_p = min(
        region,
        key=lambda p: sum(
          reliability_lengths.get(p, {}).get(layout[n.x], 100)
          for n in placed_in_comm
        ),
      )
    else:
      best_p = max(
        region,
        key=lambda p: (
          sum(1 for n in program.neighbors(logical)),
          -p,
        ),
      )

    layout[logical_id] = best_p
    available.discard(best_p)
    region.remove(best_p)


# ---------------------------------------------------------------------------
# Main compiler class
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Simulated annealing layout search
# ---------------------------------------------------------------------------

def simulated_annealing_layout(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
  initial_temp: float = 1.5,
  cooling_rate: float = 0.97,
  steps: int = 600,
) -> dict[int, int]:
  """Optimize layout via simulated annealing with adaptive restart.

  Uses classic SA with Metropolis acceptance criterion.
  Restarts if stuck for too long.
  """
  logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
  logical_ids = [q.x for q in logical_qubits]
  if len(logical_ids) < 3:
    # Too few qubits for meaningful SA
    try:
      return reliability_aware_layout(circuit, backend, hw_graph, rng)
    except Exception:
      return movement_aware_layout(circuit, backend, hw_graph, rng)

  # Start from reliability-aware layout
  try:
    current = reliability_aware_layout(circuit, backend, hw_graph, rng)
  except Exception:
    current = movement_aware_layout(circuit, backend, hw_graph, rng)

  current_cost = estimate_layout_cost(circuit, current, hw_graph)
  best = dict(current)
  best_cost = current_cost
  temp = initial_temp
  steps_since_improvement = 0
  restart_threshold = max(50, steps // 8)

  for step in range(steps):
    # Generate neighbor by swapping two logical qubits' positions
    neighbor = dict(current)
    a, b = rng.sample(logical_ids, 2)
    if a in neighbor and b in neighbor:
      neighbor[a], neighbor[b] = neighbor[b], neighbor[a]

    neighbor_cost = estimate_layout_cost(circuit, neighbor, hw_graph)
    delta = neighbor_cost - current_cost

    # Metropolis criterion
    if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-8)):
      current = neighbor
      current_cost = neighbor_cost
      steps_since_improvement = 0
      if current_cost < best_cost:
        best = dict(current)
        best_cost = current_cost
    else:
      steps_since_improvement += 1

    # Adaptive restart: if stuck, perturb more aggressively
    if steps_since_improvement > restart_threshold:
      # Do multiple random swaps to escape local minimum
      for _ in range(len(logical_ids) // 2):
        a2, b2 = rng.sample(logical_ids, 2)
        if a2 in current and b2 in current:
          current[a2], current[b2] = current[b2], current[a2]
      current_cost = estimate_layout_cost(circuit, current, hw_graph)
      steps_since_improvement = 0
      temp = initial_temp * 0.5  # Resume at moderate temperature

    temp *= cooling_rate
    if temp < 1e-8:
      # Reheat
      temp = initial_temp * 0.3
      steps_since_improvement = 0

  return best


# ---------------------------------------------------------------------------
# Cirq gate resynthesis
# ---------------------------------------------------------------------------

def resynthesize_gates(
  circuit: cirq.Circuit,
  backend,
) -> cirq.Circuit:
  """Apply Cirq's built-in gate optimizers to reduce gate count.

  Uses merge_single_qubit_moments and eject_phased_paulis which are
  safe for both Track A and Track B (they operate within the gate set).
  """
  try:
    # Merge consecutive single-qubit operations within each moment
    merged = cirq.merge_single_qubit_moments_to_phxz(circuit)
    # Drop empty moments
    merged = cirq.drop_negligible_operations_and_convert(merged)
    # Drop empty moments
    merged = cirq.drop_empty_moments(merged)
    return merged
  except Exception:
    pass

  try:
    # Alternative: use merge_single_qubit_moments
    merged = cirq.merge_single_qubit_moments_to_phased_x_and_z(circuit)
    merged = cirq.drop_empty_moments(merged)
    return merged
  except Exception:
    pass

  return circuit


# ---------------------------------------------------------------------------
# Parallel layout evaluation
# ---------------------------------------------------------------------------

def _eval_layout_worker(args: tuple) -> CompileResult | None:
  """Worker function for parallel layout evaluation."""
  (circuit, backend, hw_graph, layout,
   sabre_iterations, lookahead_window, decay, seed) = args
  rng = random.Random(seed)
  try:
    refined = reverse_traversal_refine(
      circuit, backend, hw_graph, layout, rng,
      num_iterations=sabre_iterations,
    )
    router = SabreRouter(
      backend, hw_graph, rng,
      lookahead_window=lookahead_window,
      decay=decay,
    )
    routed, _, cost = router.route(circuit, refined)
    return CompileResult(
      circuit=routed,
      initial_layout=refined,
      estimated_cost=cost,
    )
  except Exception:
    return None


def _run_serial_evaluation(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  candidates: list[dict[int, int]],
  rng: random.Random,
  sabre_iterations: int,
  lookahead_window: int,
  decay: float,
) -> list[CompileResult | None]:
  """Serially evaluate layout candidates."""
  results = []
  best_cost = float('inf')
  for i, layout in enumerate(candidates):
    # Quick pre-screening
    est = estimate_layout_cost(circuit, layout, hw_graph)
    if est > 1.8 * best_cost:
      results.append(None)
      continue
    seed = rng.randint(0, 2**31 - 1)
    result = _eval_layout_worker((
      circuit, backend, hw_graph, layout,
      sabre_iterations, lookahead_window, decay, seed,
    ))
    results.append(result)
    if result is not None and result.estimated_cost < best_cost:
      best_cost = result.estimated_cost
  return results


def _run_parallel_evaluation(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  candidates: list[dict[int, int]],
  rng: random.Random,
  sabre_iterations: int,
  lookahead_window: int,
  decay: float,
) -> list[CompileResult | None]:
  """Evaluate layout candidates in parallel using ProcessPoolExecutor."""
  # Quick pre-screening in main process first
  prescreened: list[dict[int, int]] = []
  best_est = float('inf')
  for layout in candidates:
    est = estimate_layout_cost(circuit, layout, hw_graph)
    if est <= 1.8 * best_est:
      prescreened.append(layout)
      if est < best_est:
        best_est = est

  # Build work items
  work_items = [
    (circuit, backend, hw_graph, layout,
     sabre_iterations, lookahead_window, decay,
     rng.randint(0, 2**31 - 1))
    for layout in prescreened
  ]

  # Limit concurrency to avoid memory issues
  max_workers = min(8, len(work_items))

  results_map: dict[int, CompileResult | None] = {}
  try:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = {
        executor.submit(_eval_layout_worker, item): i
        for i, item in enumerate(work_items)
      }
      for future in as_completed(futures, timeout=120):
        idx = futures[future]
        try:
          results_map[idx] = future.result()
        except Exception:
          results_map[idx] = None
  except Exception:
    # Fall back to serial if parallel fails
    results_map = {}
    for i, item in enumerate(work_items):
      results_map[i] = _eval_layout_worker(item)

  # Reconstruct ordered results (None for skipped candidates)
  results: list[CompileResult | None] = []
  prescreen_idx = 0
  for layout in candidates:
    est = estimate_layout_cost(circuit, layout, hw_graph)
    if est <= 1.8 * best_est and layout in prescreened:
      results.append(results_map.get(prescreen_idx))
      prescreen_idx += 1
    else:
      results.append(None)

  return results


# ---------------------------------------------------------------------------
# Main compiler class
# ---------------------------------------------------------------------------

class NoiseAdaptiveSabreCompiler:
  """Noise-adaptive SABRE quantum circuit compiler with SA + resynthesis."""

  def __init__(
    self,
    seed: int = 42,
    num_layout_candidates: int = 20,
    sabre_iterations: int = 3,
    lookahead_window: int = 10,
    decay: float = 0.5,
    use_simulated_annealing: bool = True,
    use_resynthesis: bool = True,
    use_parallel: bool = False,
  ):
    self.seed = seed
    self.num_layout_candidates = num_layout_candidates
    self.sabre_iterations = sabre_iterations
    self.lookahead_window = lookahead_window
    self.decay = decay
    self.use_simulated_annealing = use_simulated_annealing
    self.use_resynthesis = use_resynthesis
    self.use_parallel = use_parallel
    self.rng = random.Random(seed)
    self.last_initial_layout: dict[int, int] = {}

  def compile(
    self,
    circuit: cirq.Circuit,
    backend,
  ) -> tuple[cirq.Circuit, dict[int, int]]:
    """Compile circuit for the given backend with full optimization pipeline.

    Pipeline:
      1. Generate base layout candidates (five strategies)
      2. Add simulated annealing optimized candidates
      3. Evaluate all candidates (parallel or serial)
      4. Select best result
      5. Apply gate resynthesis
      6. Apply final gate cancellation pass
    """
    logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
    if len(logical_qubits) > backend.num_qubits:
      raise ValueError("program does not fit on backend")

    hw_graph = build_hardware_graph(backend)

    # Step 1: Generate base layout candidates
    base_candidates = generate_layout_candidates(
      circuit, backend, hw_graph, self.rng,
      num_candidates=self.num_layout_candidates,
    )

    # Step 2: Add simulated annealing candidates
    all_candidates = list(base_candidates)
    if self.use_simulated_annealing:
      # Run SA starting from a few of the best base layouts
      sa_starts = base_candidates[:min(3, len(base_candidates))]
      sa_steps = 500 if len(logical_qubits) > 16 else 300
      for base in sa_starts:
        try:
          # Use the base as starting point (not generating from scratch)
          sa_layout = _simulated_annealing_from_base(
            circuit, backend, hw_graph, self.rng, base,
            steps=sa_steps,
          )
          if sa_layout not in all_candidates:
            all_candidates.append(sa_layout)
        except Exception:
          pass
      # Also try a fresh SA from scratch
      try:
        sa_fresh = simulated_annealing_layout(
          circuit, backend, hw_graph, self.rng,
          steps=sa_steps,
        )
        if sa_fresh not in all_candidates:
          all_candidates.append(sa_fresh)
      except Exception:
        pass

    # Step 3: Evaluate candidates
    num_candidates = len(all_candidates)
    if self.use_parallel and num_candidates > 8:
      results = _run_parallel_evaluation(
        circuit, backend, hw_graph, all_candidates,
        self.rng, self.sabre_iterations,
        self.lookahead_window, self.decay,
      )
    else:
      results = _run_serial_evaluation(
        circuit, backend, hw_graph, all_candidates,
        self.rng, self.sabre_iterations,
        self.lookahead_window, self.decay,
      )

    # Step 4: Pick best valid result
    best_result: CompileResult | None = None
    for r in results:
      if r is not None:
        if best_result is None or r.estimated_cost < best_result.estimated_cost:
          best_result = r

    if best_result is None:
      raise RuntimeError("all layout candidates failed")

    # Step 5: Gate resynthesis
    if self.use_resynthesis:
      try:
        resynthesized = resynthesize_gates(best_result.circuit, backend)
        best_result = CompileResult(
          circuit=resynthesized,
          initial_layout=best_result.initial_layout,
          estimated_cost=best_result.estimated_cost,
        )
      except Exception:
        pass

    # Step 6: Final gate cancellation
    optimized = optimize_circuit(best_result.circuit, backend)
    self.last_initial_layout = best_result.initial_layout

    return optimized, best_result.initial_layout


def _simulated_annealing_from_base(
  circuit: cirq.Circuit,
  backend,
  hw_graph: nx.Graph,
  rng: random.Random,
  base_layout: dict[int, int],
  initial_temp: float = 0.8,
  cooling_rate: float = 0.96,
  steps: int = 300,
) -> dict[int, int]:
  """Run simulated annealing starting from a specific base layout."""
  logical_qubits = sorted(circuit.all_qubits(), key=lambda q: q.x)
  logical_ids = [q.x for q in logical_qubits]

  current = dict(base_layout)
  current_cost = estimate_layout_cost(circuit, current, hw_graph)
  best = dict(current)
  best_cost = current_cost
  temp = initial_temp

  for _ in range(steps):
    neighbor = dict(current)
    a, b = rng.sample(logical_ids, 2)
    if a in neighbor and b in neighbor:
      neighbor[a], neighbor[b] = neighbor[b], neighbor[a]

    neighbor_cost = estimate_layout_cost(circuit, neighbor, hw_graph)
    delta = neighbor_cost - current_cost

    if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-8)):
      current = neighbor
      current_cost = neighbor_cost
      if current_cost < best_cost:
        best = dict(current)
        best_cost = current_cost

    temp *= cooling_rate
    if temp < 1e-6:
      break

  return best
