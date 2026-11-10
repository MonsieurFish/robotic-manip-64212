"""
Draw the floor plan from above (like the figures in the proposal) and save it
as a PNG. Useful for a quick look without opening the 3D viewer.

Run from this folder:
    .venv/bin/python plot_top_down.py              # default environment
    .venv/bin/python plot_top_down.py --seed 7     # one randomized environment
    .venv/bin/python plot_top_down.py --gallery    # 9 randomized environments
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as PolygonPatch

from environment_params import EnvironmentParams, sample_random_params
from maze_layout import make_two_segment_layout


def draw_layout(ax, layout, title=""):
    """Draw one floor plan onto a matplotlib axis."""
    # Open areas (corridors + bay) in light blue
    for rect in layout.open_areas.values():
        ax.add_patch(PolygonPatch(rect.corners(), facecolor="#cde0f5", edgecolor="none"))

    # Walls in dark gray (each wall piece is a triangle; outlining each one in
    # the same color hides the seams between them)
    for corners in layout.walls.values():
        ax.add_patch(PolygonPatch(corners, facecolor="#737f99", edgecolor="#737f99",
                                  linewidth=0.5))

    # Block at start (solid green) and goal (green outline)
    ax.add_patch(PolygonPatch(layout.block_start.corners(),
                              facecolor="#59a64d", edgecolor="black", linewidth=0.8))
    ax.add_patch(PolygonPatch(layout.block_goal.corners(), facecolor="none",
                              edgecolor="#59a64d", linestyle="--", linewidth=1.5))

    # Roofs: hatched and see-through, drawn on top so the block shows underneath
    for rect in layout.roofs.values():
        ax.add_patch(PolygonPatch(rect.corners(), facecolor="#f29933", alpha=0.35,
                                  hatch="///", edgecolor="#b36b1f"))

    # Labels
    for label, rect in [("start", layout.block_start), ("goal", layout.block_goal)]:
        ax.annotate(label, rect.center, ha="center", va="center", fontsize=8)
    for name, heading in layout.corridor_heading.items():
        tip = layout.open_areas["bay"].center + 0.62 * np.array([np.cos(heading), np.sin(heading)])
        ax.annotate(name, tip, ha="center", va="center", fontsize=9, weight="bold")

    # Where the robot base will go
    ax.add_patch(Circle((0, 0), 0.10, facecolor="#4d4d4d"))
    ax.annotate("robot", (0, -0.15), ha="center", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9)


def title_for(params, prefix=""):
    return (f"{prefix}angle {params.corner_angle_deg:.0f}°, bay {params.bay_size * 100:.0f} cm, "
            f"roof {params.start_roof_length * 100:.0f} cm, start {params.start_corridor}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="randomize the environment with this seed")
    parser.add_argument("--gallery", action="store_true",
                        help="draw 9 randomized environments (seeds 0-8)")
    parser.add_argument("--out", default="top_down.png", help="where to save the image")
    args = parser.parse_args()

    if args.gallery:
        fig, axes = plt.subplots(3, 3, figsize=(13, 13))
        for seed, ax in enumerate(axes.flat):
            params = sample_random_params(seed=seed)
            draw_layout(ax, make_two_segment_layout(params), title_for(params, f"seed {seed}: "))
    else:
        if args.seed is None:
            params = EnvironmentParams()
        else:
            params = sample_random_params(seed=args.seed)
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_layout(ax, make_two_segment_layout(params), title_for(params))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
