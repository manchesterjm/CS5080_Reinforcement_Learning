"""
Visualization utilities for HW2 Parking Lot Q-Learning

Generates matplotlib figures for policies, learning curves,
value heatmaps, and parameter comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple

from parking_lot import (ROWS, COLS, PARKING_SPOTS, BARRIER_CELLS,
                         ALL_PARKING_POSITIONS, ENTRANCE, NUM_GOALS,
                         SPOT_NAMES, ACTION_DELTAS, ParkingLot)


# Arrow directions for policy display (dx, dy in plot coords)
# Plot coords: x = col, y = row (inverted)
ARROW_DX = {0: 0, 1: 0, 2: -0.3, 3: 0.3}   # LEFT/RIGHT
ARROW_DY = {0: 0.3, 1: -0.3, 2: 0, 3: 0}     # UP/DOWN (inverted y)


def plot_parking_lot(goal_id: int,
                     policy: Optional[Dict[Tuple[int, int], int]] = None,
                     values: Optional[Dict[Tuple[int, int], float]] = None,
                     path: Optional[List[Tuple[int, int]]] = None,
                     title: Optional[str] = None,
                     ax: Optional[plt.Axes] = None,
                     show: bool = True) -> plt.Axes:
    """Render parking lot grid with optional policy arrows and value colors.

    Args:
        goal_id: Which parking spot is the goal (0-7)
        policy: Dict mapping (row, col) -> action
        values: Dict mapping (row, col) -> float for heatmap coloring
        path: List of (row, col) to highlight as optimal path
        title: Plot title
        ax: Existing axes to draw on
        show: Whether to call plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    env = ParkingLot()
    walkable = env._get_walkable_cells(goal_id)
    goal_pos = PARKING_SPOTS[goal_id]
    path_set = set(path) if path else set()

    # Value normalization for coloring
    if values:
        vmin = min(values.values())
        vmax = max(values.values())
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlGn

    # Draw grid
    for r in range(ROWS):
        for c in range(COLS):
            pos = (r, c)
            x, y = c, ROWS - 1 - r  # Convert to plot coords

            if pos in BARRIER_CELLS:
                # Barrier
                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#404040", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y, "###", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")

            elif pos == goal_pos:
                # Goal
                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#4CAF50", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y, f"G\n{SPOT_NAMES[goal_id]}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

            elif pos == ENTRANCE:
                # Entrance
                color = "#E3F2FD"
                if values and pos in values:
                    color = cmap(norm(values[pos]))
                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y + 0.25, "E", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")
                # Draw arrow if policy exists
                if policy and pos in policy:
                    action = policy[pos]
                    ax.annotate("", xy=(x + ARROW_DX[action],
                                        y + ARROW_DY[action]),
                                xytext=(x, y - 0.1),
                                arrowprops=dict(arrowstyle="->",
                                                color="blue", lw=1.5))

            elif pos in ALL_PARKING_POSITIONS and pos not in walkable:
                # Blocked parking spot
                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#FFCDD2", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                sid = [k for k, v in PARKING_SPOTS.items() if v == pos][0]
                ax.text(x, y, f"[{SPOT_NAMES[sid]}]",
                        ha="center", va="center",
                        fontsize=7, color="#B71C1C")

            elif pos in walkable:
                # Walkable cell
                color = "white"
                if values and pos in values:
                    color = cmap(norm(values[pos]))
                elif pos in path_set:
                    color = "#FFF9C4"

                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)

                # Draw policy arrow
                if policy and pos in policy:
                    action = policy[pos]
                    ax.annotate("", xy=(x + ARROW_DX[action],
                                        y + ARROW_DY[action]),
                                xytext=(x, y),
                                arrowprops=dict(arrowstyle="->",
                                                color="blue", lw=1.5))
            else:
                # Non-walkable, non-barrier (shouldn't happen often)
                rect = mpatches.FancyBboxPatch(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#E0E0E0", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)

    # Draw path line if provided
    if path and len(path) > 1:
        px = [p[1] for p in path]
        py = [ROWS - 1 - p[0] for p in path]
        ax.plot(px, py, "o-", color="orange", markersize=4,
                linewidth=2, alpha=0.7, zorder=5)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_yticklabels([str(ROWS - 1 - i) for i in range(ROWS)])
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if title:
        ax.set_title(title, fontsize=11)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_learning_curves(rewards_dict: Dict[str, List[float]],
                         lengths_dict: Optional[Dict[str, List[float]]] = None,
                         window: int = 100,
                         title: str = "Learning Curves",
                         ax: Optional[plt.Axes] = None,
                         show: bool = True):
    """Plot reward (and optionally length) learning curves.

    Args:
        rewards_dict: {"Agent Name": [ep_rewards, ...]}
        lengths_dict: {"Agent Name": [ep_lengths, ...]}
        window: Moving average window size
        title: Plot title
    """
    n_plots = 2 if lengths_dict else 1

    if ax is None:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots),
                                 sharex=True)
        if n_plots == 1:
            axes = [axes]
    else:
        axes = [ax]

    colors = plt.cm.tab10.colors

    # Plot rewards
    for i, (name, rewards) in enumerate(rewards_dict.items()):
        episodes = np.arange(1, len(rewards) + 1)
        axes[0].plot(episodes, rewards, alpha=0.15, color=colors[i % 10])
        # Moving average
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0].plot(np.arange(window, len(rewards) + 1), ma,
                         label=name, color=colors[i % 10], linewidth=2)
        else:
            axes[0].plot(episodes, rewards, label=name,
                         color=colors[i % 10])

    axes[0].set_ylabel("Episode Reward")
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Plot lengths
    if lengths_dict and n_plots > 1:
        for i, (name, lengths) in enumerate(lengths_dict.items()):
            episodes = np.arange(1, len(lengths) + 1)
            axes[1].plot(episodes, lengths, alpha=0.15,
                         color=colors[i % 10])
            if len(lengths) >= window:
                ma = np.convolve(lengths, np.ones(window) / window,
                                 mode="valid")
                axes[1].plot(np.arange(window, len(lengths) + 1), ma,
                             label=name, color=colors[i % 10], linewidth=2)
        axes[1].set_ylabel("Episode Length")
        axes[1].set_xlabel("Episode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


def plot_value_heatmap(values: Dict[Tuple[int, int], float],
                       goal_id: int,
                       title: Optional[str] = None,
                       ax: Optional[plt.Axes] = None,
                       show: bool = True) -> plt.Axes:
    """Plot state values as a heatmap on the parking lot grid."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    env = ParkingLot()
    walkable = env._get_walkable_cells(goal_id)
    goal_pos = PARKING_SPOTS[goal_id]

    # Build value matrix
    grid = np.full((ROWS, COLS), np.nan)
    for (r, c), v in values.items():
        grid[r, c] = v

    # Mask barriers and blocked spots
    masked = np.ma.masked_invalid(grid)

    # Plot with inverted y-axis
    im = ax.imshow(masked, cmap="RdYlGn", origin="upper",
                   extent=[-0.5, COLS - 0.5, -0.5, ROWS - 0.5])
    ax.invert_yaxis()

    # Annotate cells
    for r in range(ROWS):
        for c in range(COLS):
            pos = (r, c)
            if pos in BARRIER_CELLS:
                ax.text(c, r, "###", ha="center", va="center",
                        fontsize=6, color="white")
            elif pos == goal_pos:
                ax.text(c, r, f"G", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            elif pos in ALL_PARKING_POSITIONS and pos not in walkable:
                sid = [k for k, v in PARKING_SPOTS.items() if v == pos][0]
                ax.text(c, r, f"[{SPOT_NAMES[sid]}]",
                        ha="center", va="center", fontsize=6, color="gray")
            elif pos in values:
                ax.text(c, r, f"{values[pos]:.1f}",
                        ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="State Value")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"State Values — Goal: {SPOT_NAMES[goal_id]}")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_multi_policy(policies: Dict[int, Dict[Tuple[int, int], int]],
                      agent_name: str = "",
                      show: bool = True,
                      save_path: Optional[str] = None):
    """Plot policies for multiple goals in a 2x4 subplot grid."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{agent_name} Learned Policies", fontsize=14)

    for gid in range(NUM_GOALS):
        row, col = gid // 4, gid % 4
        ax = axes[row, col]
        plot_parking_lot(goal_id=gid, policy=policies[gid],
                         title=f"Goal: {SPOT_NAMES[gid]}", ax=ax,
                         show=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_parameter_sensitivity(results: Dict[str, Dict],
                               param_name: str,
                               title: str = "Parameter Sensitivity",
                               show: bool = True,
                               save_path: Optional[str] = None):
    """Plot parameter sweep results.

    Args:
        results: {param_value_str: {"rewards": [...], "lengths": [...]}}
        param_name: Name of the parameter being swept
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, (param_val, data) in enumerate(results.items()):
        rewards = data["rewards"]
        window = min(100, len(rewards) // 5)
        if window > 0 and len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(np.arange(window, len(rewards) + 1), ma,
                     label=f"{param_name}={param_val}",
                     color=colors[i % 10], linewidth=1.5)

        lengths = data["lengths"]
        if window > 0 and len(lengths) >= window:
            ma = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax2.plot(np.arange(window, len(lengths) + 1), ma,
                     label=f"{param_name}={param_val}",
                     color=colors[i % 10], linewidth=1.5)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title(f"Reward vs {param_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title(f"Steps vs {param_name}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_q_overestimation(q_history: List[float],
                          dq_history: List[float],
                          window: int = 100,
                          title: str = "Q-Value Overestimation",
                          show: bool = True,
                          save_path: Optional[str] = None):
    """Compare Q-value estimates between Q-learning and Double Q-learning."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for data, label, color in [(q_history, "Q-Learning", "tab:red"),
                                (dq_history, "Double Q-Learning", "tab:blue")]:
        episodes = np.arange(1, len(data) + 1)
        ax.plot(episodes, data, alpha=0.15, color=color)
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(data) + 1), ma,
                    label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Max Q-Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
