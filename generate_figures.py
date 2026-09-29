#!/usr/bin/env python3
"""Generate figures for the competition documentation."""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import networkx as nx


OUT_DIR = Path(__file__).resolve().parent.parent / "作品说明文档" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
C_DARK = "#003366"
C_MID = "#2B6CB0"
C_LIGHT = "#63B3ED"
C_ACCENT = "#E53E3E"
C_GRAY = "#718096"


def fig_ablation_study():
  """Ablation study bar chart."""
  fig, ax = plt.subplots(figsize=(8, 4.5))

  configs = [
    "Noise-adaptive\nweights",
    "Multiple\ncandidates",
    "Reverse\ntraversal",
    "Structure-\naware layout",
  ]
  contributions = [8.0, 4.6, 3.0, 0.07]
  colors = [C_DARK, C_MID, C_LIGHT, C_GRAY]

  bars = ax.bar(configs, contributions, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

  for bar, val in zip(bars, contributions):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
            f"+{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_DARK)

  ax.set_ylabel("Score improvement (%)", fontsize=11, color=C_DARK)
  ax.set_title("Ablation Study: Component Contributions to Final Score", fontsize=13, fontweight="bold", color=C_DARK, pad=15)
  ax.set_ylim(0, 10.5)
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.tick_params(axis="x", labelsize=10)
  ax.tick_params(axis="y", labelsize=9)
  ax.grid(axis="y", alpha=0.3, linestyle="--")

  fig.tight_layout()
  fig.savefig(OUT_DIR / "ablation_study.png", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("Generated: ablation_study.png")


def fig_baseline_comparison():
  """Baseline comparison grouped bar chart."""
  fig, ax = plt.subplots(figsize=(8, 4.5))

  methods = [
    "reliability\ngreedy",
    "paper_greedy\nedge",
    "paper_greedy\nvertex",
    "Our\ncompiler",
  ]
  track_a = [0.7974, 0.8271, 0.8315, 0.8472]
  track_b = [0.3018, 0.2256, 0.2939, 0.4580]
  final = [0.6735, 0.6768, 0.6971, 0.7499]

  x = np.arange(len(methods))
  width = 0.25

  bars_a = ax.bar(x - width, track_a, width, label="Track A", color=C_DARK, edgecolor="white")
  bars_b = ax.bar(x, track_b, width, label="Track B", color=C_LIGHT, edgecolor="white")
  bars_f = ax.bar(x + width, final, width, label="Final Score", color=C_ACCENT, edgecolor="white")

  for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ACCENT)

  ax.set_xticks(x)
  ax.set_xticklabels(methods, fontsize=9)
  ax.set_ylabel("Score", fontsize=11, color=C_DARK)
  ax.set_title("Comparison with Competition Baselines", fontsize=13, fontweight="bold", color=C_DARK, pad=15)
  ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
  ax.set_ylim(0, 1.0)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(axis="y", alpha=0.3, linestyle="--")

  fig.tight_layout()
  fig.savefig(OUT_DIR / "baseline_comparison.png", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("Generated: baseline_comparison.png")


def fig_swap_reduction():
  """SWAP count reduction for GHZ circuit."""
  fig, ax = plt.subplots(figsize=(6, 4))

  methods = ["Without structure\nawareness", "With star-aware\nlayout"]
  swap_counts = [10, 6]
  colors = [C_GRAY, C_DARK]

  bars = ax.bar(methods, swap_counts, color=colors, width=0.45, edgecolor="white", linewidth=0.8)

  for bar, val in zip(bars, swap_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val} SWAPs", ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_DARK)

  ax.set_ylabel("Number of SWAP gates", fontsize=11, color=C_DARK)
  ax.set_title("GHZ Circuit: Star-Aware Layout Reduces SWAPs by 40%", fontsize=12, fontweight="bold", color=C_DARK, pad=15)
  ax.set_ylim(0, 13)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.tick_params(axis="x", labelsize=10)
  ax.grid(axis="y", alpha=0.3, linestyle="--")

  fig.tight_layout()
  fig.savefig(OUT_DIR / "swap_reduction.png", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("Generated: swap_reduction.png")


def fig_per_benchmark():
  """Per-benchmark success rate chart."""
  fig, ax = plt.subplots(figsize=(9, 4.5))

  benchmarks = [
    "BV", "HS", "Toffoli", "Fredkin",
    "Peres", "OR", "Adder", "Random",
    "GHZ", "Parity",
  ]
  success_rates = [
    0.9570, 0.7734, 0.8789, 0.8750,
    0.9141, 0.9180, 0.7617, 0.6992,
    0.5254, 0.3818,
  ]
  tracks = ["A"] * 8 + ["B"] * 2
  bar_colors = [C_DARK if t == "A" else C_ACCENT for t in tracks]

  bars = ax.bar(benchmarks, success_rates, color=bar_colors, width=0.6, edgecolor="white", linewidth=0.8)

  for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{rate:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_DARK)

  ax.axhline(y=0.8472, color=C_DARK, linestyle="--", linewidth=1, alpha=0.6, label=f"Track A avg: 0.8472")
  ax.axhline(y=0.4580, color=C_ACCENT, linestyle="--", linewidth=1, alpha=0.6, label=f"Track B avg: 0.4580")

  ax.set_ylabel("Success Rate", fontsize=11, color=C_DARK)
  ax.set_title("Per-Benchmark Success Rate (256 / 1024 shots)", fontsize=13, fontweight="bold", color=C_DARK, pad=15)
  ax.legend(fontsize=8, loc="lower left", framealpha=0.8)
  ax.set_ylim(0, 1.05)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.tick_params(axis="x", labelsize=8)
  ax.grid(axis="y", alpha=0.3, linestyle="--")

  fig.tight_layout()
  fig.savefig(OUT_DIR / "per_benchmark.png", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("Generated: per_benchmark.png")


def fig_pipeline():
  """Pipeline flow diagram."""
  fig, ax = plt.subplots(figsize=(9, 3))
  ax.set_xlim(0, 10)
  ax.set_ylim(0, 3)
  ax.axis("off")

  stages = [
    (0.5, "Circuit\n+ Backend", C_GRAY),
    (2.2, "Interaction Graph\n+ Structure Detect", C_LIGHT),
    (4.0, "Multi-Strategy\nLayout Generation", C_MID),
    (6.0, "SABRE Routing\n+ Reverse Traversal", C_DARK),
    (8.0, "Post-Optimization\n(Gate Cancellation)", C_ACCENT),
  ]

  for i, (x, label, color) in enumerate(stages):
    rect = plt.Rectangle((x - 0.8, 0.8), 1.6, 1.4, facecolor=color, edgecolor="white",
                          linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    ax.text(x, 1.5, label, ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", zorder=3)

    if i < len(stages) - 1:
      ax.annotate("", xy=(stages[i + 1][0] - 0.85, 1.5),
                  xytext=(x + 0.85, 1.5),
                  arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2, alpha=0.6), zorder=1)

  ax.set_title("Compiler Pipeline", fontsize=13, fontweight="bold", color=C_DARK, pad=10)

  fig.tight_layout()
  fig.savefig(OUT_DIR / "pipeline.png", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("Generated: pipeline.png")


if __name__ == "__main__":
  fig_ablation_study()
  fig_baseline_comparison()
  fig_swap_reduction()
  fig_per_benchmark()
  fig_pipeline()
  print(f"\nAll figures saved to: {OUT_DIR}")
