"""
Generate Figures for HW1 Monte Carlo Paper

Saves publication-quality figures for the LaTeX paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Tuple, Optional

from maze import Maze, create_hw1_maze
from monte_carlo import MonteCarloAgent
from maze_generator import generate_random_maze


# Set matplotlib style for publication
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def plot_maze_with_policy(maze: Maze,
                          policy: Dict[Tuple[int, int], int],
                          values: Optional[Dict[Tuple[int, int], float]] = None,
                          title: str = "Learned Policy",
                          filename: str = None) -> None:
    """
    Plot maze with policy arrows and optionally value colors.

    Args:
        maze: The maze environment
        policy: Policy mapping states to actions
        values: Optional state values for coloring
        title: Plot title
        filename: If provided, save to this file
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Arrow directions for policy visualization
    arrow_dirs = {
        Maze.UP: (0, 0.3),
        Maze.DOWN: (0, -0.3),
        Maze.LEFT: (-0.3, 0),
        Maze.RIGHT: (0.3, 0)
    }

    # Normalize values if provided
    if values:
        v_min = min(values.values())
        v_max = max(values.values())
        if v_max > v_min:
            norm_values = {s: (v - v_min) / (v_max - v_min) for s, v in values.items()}
        else:
            norm_values = {s: 0.5 for s in values}

    # Draw grid
    for r in range(1, maze.rows + 2):
        ax.axhline(y=maze.rows + 1 - r, color='black', linewidth=1)
    for c in range(1, maze.cols + 2):
        ax.axvline(x=c - 1, color='black', linewidth=1)

    # Draw cells
    for r in range(1, maze.rows + 1):
        for c in range(1, maze.cols + 1):
            pos = (r, c)
            px = c - 0.5
            py = maze.rows - r + 0.5

            if pos in maze.barriers:
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='dimgray', edgecolor='black')
                ax.add_patch(rect)
            elif pos == maze.goal:
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                ax.text(px, py, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
            elif pos == maze.start:
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='lightblue', edgecolor='black')
                ax.add_patch(rect)
                ax.text(px, py, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
            elif values and pos in values:
                v = norm_values.get(pos, 0.5)
                color = plt.cm.RdYlGn(v)
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor=color, edgecolor='black')
                ax.add_patch(rect)

            # Draw policy arrow
            if policy and pos in policy and pos != maze.goal and pos not in maze.barriers:
                action = policy[pos]
                dx, dy = arrow_dirs[action]
                ax.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1,
                        fc='blue', ec='blue')

    ax.set_xlim(0, maze.cols)
    ax.set_ylim(0, maze.rows)
    ax.set_xticks([c - 0.5 for c in range(1, maze.cols + 1)])
    ax.set_xticklabels([str(c) for c in range(1, maze.cols + 1)])
    ax.set_yticks([maze.rows - r + 0.5 for r in range(1, maze.rows + 1)])
    ax.set_yticklabels([str(r) for r in range(1, maze.rows + 1)])
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()


def plot_learning_curves(episode_lengths: list,
                         episode_returns: list,
                         window: int = 100,
                         filename: str = None) -> None:
    """
    Plot learning curves showing episode length and returns.

    Args:
        episode_lengths: List of episode lengths
        episode_returns: List of episode returns
        window: Moving average window
        filename: If provided, save to this file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    episodes = np.arange(1, len(episode_lengths) + 1)

    def moving_avg(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    # Episode lengths
    ax1.plot(episodes, episode_lengths, alpha=0.2, color='blue', linewidth=0.5)
    ma_lengths = moving_avg(episode_lengths, window)
    ax1.plot(episodes[window-1:], ma_lengths, color='blue', linewidth=1.5,
             label=f'Moving Avg (n={window})')
    ax1.set_ylabel('Episode Length')
    ax1.set_title('Learning Curves')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Episode returns
    ax2.plot(episodes, episode_returns, alpha=0.2, color='green', linewidth=0.5)
    ma_returns = moving_avg(episode_returns, window)
    ax2.plot(episodes[window-1:], ma_returns, color='green', linewidth=1.5,
             label=f'Moving Avg (n={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()


def plot_value_heatmap(maze: Maze,
                       values: Dict[Tuple[int, int], float],
                       title: str = "State Values V(s)",
                       filename: str = None) -> None:
    """
    Plot state values as a heatmap.

    Args:
        maze: The maze environment
        values: Dictionary mapping states to values
        title: Plot title
        filename: If provided, save to this file
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create value matrix
    V_matrix = np.full((maze.rows, maze.cols), np.nan)

    for (r, c), v in values.items():
        V_matrix[r - 1, c - 1] = v

    for (r, c) in maze.barriers:
        V_matrix[r - 1, c - 1] = np.nan

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='dimgray')

    im = ax.imshow(V_matrix, cmap=cmap, origin='upper')

    # Add value annotations
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not np.isnan(V_matrix[r, c]):
                ax.text(c, r, f'{V_matrix[r, c]:.2f}',
                       ha='center', va='center', fontsize=9)

    ax.set_xticks(range(maze.cols))
    ax.set_xticklabels(range(1, maze.cols + 1))
    ax.set_yticks(range(maze.rows))
    ax.set_yticklabels(range(1, maze.rows + 1))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Value', shrink=0.8)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()


def plot_random_mazes_comparison(filename: str = None) -> None:
    """
    Plot learned policies for random mazes of different sizes.

    Args:
        filename: If provided, save to this file
    """
    import random

    sizes = [(5, 5), (7, 7), (10, 10)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (rows, cols) in enumerate(sizes):
        ax = axes[idx]

        # Generate and train
        random.seed(42 + idx)
        np.random.seed(42 + idx)

        maze = generate_random_maze(rows, cols, barrier_density=0.2, seed=42 + idx)
        agent = MonteCarloAgent(maze, gamma=0.99)

        n_episodes = 5000 * (rows * cols) // 25
        agent.train(n_episodes=n_episodes, verbose=False)

        # Draw maze with policy
        arrow_dirs = {
            Maze.UP: (0, 0.25),
            Maze.DOWN: (0, -0.25),
            Maze.LEFT: (-0.25, 0),
            Maze.RIGHT: (0.25, 0)
        }

        for r in range(1, maze.rows + 2):
            ax.axhline(y=maze.rows + 1 - r, color='black', linewidth=0.5)
        for c in range(1, maze.cols + 2):
            ax.axvline(x=c - 1, color='black', linewidth=0.5)

        for r in range(1, maze.rows + 1):
            for c in range(1, maze.cols + 1):
                pos = (r, c)
                px = c - 0.5
                py = maze.rows - r + 0.5

                if pos in maze.barriers:
                    rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                              facecolor='dimgray', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                elif pos == maze.goal:
                    rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                              facecolor='lightgreen', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                    ax.text(px, py, 'G', ha='center', va='center', fontsize=8, fontweight='bold')
                elif pos == maze.start:
                    rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                              facecolor='lightblue', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                    ax.text(px, py, 'S', ha='center', va='center', fontsize=8, fontweight='bold')

                if agent.policy and pos in agent.policy and pos != maze.goal and pos not in maze.barriers:
                    action = agent.policy[pos]
                    dx, dy = arrow_dirs[action]
                    ax.arrow(px, py, dx, dy, head_width=0.12, head_length=0.08,
                            fc='blue', ec='blue', linewidth=0.5)

        ax.set_xlim(0, maze.cols)
        ax.set_ylim(0, maze.rows)
        ax.set_aspect('equal')
        ax.set_title(f'{rows}×{cols} Maze ({maze.n_states} states)')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()


def main():
    """Generate all figures for the paper."""
    import random

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Output directory
    output_dir = "../Source/figures"
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Training agent on HW1 maze...")
    maze = create_hw1_maze()
    agent = MonteCarloAgent(maze, gamma=0.99)
    agent.train(n_episodes=5000, verbose=True, print_interval=1000)

    # Figure 1: Learned policy
    print("\nGenerating Figure 1: Learned Policy...")
    plot_maze_with_policy(maze, agent.policy,
                          title="Learned Policy (5×5 HW1 Maze)",
                          filename=f"{output_dir}/learned_policy.pdf")

    # Figure 2: Learning curves
    print("Generating Figure 2: Learning Curves...")
    plot_learning_curves(agent.episode_lengths, agent.episode_returns,
                         window=100,
                         filename=f"{output_dir}/learning_curves.pdf")

    # Figure 3: Value heatmap
    print("Generating Figure 3: Value Heatmap...")
    V = agent.get_state_values()
    plot_value_heatmap(maze, V,
                       title="State Values V(s)",
                       filename=f"{output_dir}/value_heatmap.pdf")

    # Figure 4: Policy with values
    print("Generating Figure 4: Policy with Values...")
    plot_maze_with_policy(maze, agent.policy, values=V,
                          title="Policy with State Values",
                          filename=f"{output_dir}/policy_with_values.pdf")

    # Figure 5: Random mazes comparison
    print("Generating Figure 5: Random Mazes Comparison...")
    plot_random_mazes_comparison(filename=f"{output_dir}/random_mazes.pdf")

    print("\nAll figures generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
