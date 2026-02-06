"""
Visualization for Monte Carlo Maze Solver

Provides matplotlib-based visualization for:
- Maze rendering with policy arrows
- Learning curves
- Value function heatmaps
- Episode replay animation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Tuple, Optional

from maze import Maze


def plot_maze(maze: Maze,
              policy: Optional[Dict[Tuple[int, int], int]] = None,
              values: Optional[Dict[Tuple[int, int], float]] = None,
              path: Optional[List[Tuple[int, int]]] = None,
              title: str = "Maze",
              ax: Optional[plt.Axes] = None,
              show: bool = True) -> plt.Axes:
    """
    Plot the maze with optional policy arrows, values, and path.

    Args:
        maze: The maze environment
        policy: Policy mapping states to actions (shows arrows)
        values: State values (shows as heatmap)
        path: List of states forming a path (highlights cells)
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show: Whether to call plt.show()

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Arrow directions for policy visualization (dx, dy in plot coordinates)
    arrow_dirs = {
        Maze.UP: (0, 0.3),
        Maze.DOWN: (0, -0.3),
        Maze.LEFT: (-0.3, 0),
        Maze.RIGHT: (0.3, 0)
    }

    # Create value colormap if values provided
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
            # Plot coordinates (x, y) where y is inverted
            px = c - 0.5
            py = maze.rows - r + 0.5

            if pos in maze.barriers:
                # Barrier - dark gray
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='dimgray', edgecolor='black')
                ax.add_patch(rect)
            elif pos == maze.goal:
                # Goal - green
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                ax.text(px, py, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
            elif pos == maze.start:
                # Start - light blue
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='lightblue', edgecolor='black')
                ax.add_patch(rect)
                ax.text(px, py, 'S', ha='center', va='center', fontsize=14, fontweight='bold')
            elif values and pos in values:
                # Value heatmap coloring
                v = norm_values.get(pos, 0.5)
                color = plt.cm.RdYlGn(v)  # Red (low) to Green (high)
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(px, py - 0.25, f'{values[pos]:.2f}',
                       ha='center', va='center', fontsize=8)
            elif path and pos in path:
                # Path highlight - yellow
                rect = mpatches.Rectangle((c - 1, maze.rows - r), 1, 1,
                                          facecolor='yellow', edgecolor='black', alpha=0.5)
                ax.add_patch(rect)

            # Draw policy arrow if provided and not terminal/barrier
            if policy and pos in policy and pos != maze.goal and pos not in maze.barriers:
                action = policy[pos]
                dx, dy = arrow_dirs[action]
                ax.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1,
                        fc='blue', ec='blue')

    # Labels
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

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_learning_curves(episode_lengths: List[int],
                        episode_returns: List[float],
                        window: int = 100,
                        title: str = "Learning Curves") -> None:
    """
    Plot learning curves showing episode length and returns over training.

    Args:
        episode_lengths: List of episode lengths
        episode_returns: List of episode returns
        window: Window size for moving average
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    episodes = np.arange(1, len(episode_lengths) + 1)

    # Moving average
    def moving_avg(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    # Episode lengths
    ax1.plot(episodes, episode_lengths, alpha=0.3, color='blue', label='Raw')
    ma_lengths = moving_avg(episode_lengths, window)
    ax1.plot(episodes[window-1:], ma_lengths, color='blue', linewidth=2,
             label=f'Moving Avg ({window})')
    ax1.set_ylabel('Episode Length')
    ax1.set_title(f'{title} - Episode Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode returns
    ax2.plot(episodes, episode_returns, alpha=0.3, color='green', label='Raw')
    ma_returns = moving_avg(episode_returns, window)
    ax2.plot(episodes[window-1:], ma_returns, color='green', linewidth=2,
             label=f'Moving Avg ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    ax2.set_title(f'{title} - Episode Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_value_heatmap(maze: Maze,
                       values: Dict[Tuple[int, int], float],
                       title: str = "State Values V(s)") -> None:
    """
    Plot state values as a heatmap.

    Args:
        maze: The maze environment
        values: Dictionary mapping states to values
        title: Plot title
    """
    # Create value matrix
    V_matrix = np.full((maze.rows, maze.cols), np.nan)

    for (r, c), v in values.items():
        V_matrix[r - 1, c - 1] = v

    # Mark barriers with a distinct value
    for (r, c) in maze.barriers:
        V_matrix[r - 1, c - 1] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))

    # Custom colormap
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='dimgray')  # Barriers

    im = ax.imshow(V_matrix, cmap=cmap, origin='upper')

    # Add value annotations
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not np.isnan(V_matrix[r, c]):
                ax.text(c, r, f'{V_matrix[r, c]:.2f}',
                       ha='center', va='center', fontsize=10)

    # Labels
    ax.set_xticks(range(maze.cols))
    ax.set_xticklabels(range(1, maze.cols + 1))
    ax.set_yticks(range(maze.rows))
    ax.set_yticklabels(range(1, maze.rows + 1))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Value')

    plt.tight_layout()
    plt.show()


def plot_policy_comparison(maze: Maze,
                          policies: List[Dict[Tuple[int, int], int]],
                          titles: List[str]) -> None:
    """
    Plot multiple policies side by side for comparison.

    Args:
        maze: The maze environment
        policies: List of policy dictionaries
        titles: List of titles for each policy
    """
    n = len(policies)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, policy, title in zip(axes, policies, titles):
        plot_maze(maze, policy=policy, title=title, ax=ax, show=False)

    plt.tight_layout()
    plt.show()


def animate_episode(maze: Maze,
                   episode: List[Tuple[Tuple[int, int], int, float]],
                   delay: float = 0.5) -> None:
    """
    Animate an episode step by step.

    Args:
        maze: The maze environment
        episode: List of (state, action, reward) tuples
        delay: Delay between frames in seconds
    """
    import time

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    path = []
    for i, (state, action, reward) in enumerate(episode):
        path.append(state)
        ax.clear()
        plot_maze(maze, path=path, title=f'Step {i+1}: Action={Maze.ACTION_NAMES[action]}, R={reward:.2f}',
                 ax=ax, show=False)
        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    from maze import create_hw1_maze
    from monte_carlo import MonteCarloAgent

    # Create and train agent
    maze = create_hw1_maze()
    agent = MonteCarloAgent(maze, gamma=0.99)

    print("Training agent...")
    agent.train(n_episodes=5000, verbose=True, print_interval=1000)

    # Plot maze with learned policy
    print("\nPlotting learned policy...")
    plot_maze(maze, policy=agent.policy, title="Learned Policy")

    # Plot with values
    V = agent.get_state_values()
    plot_maze(maze, policy=agent.policy, values=V, title="Policy with Values")

    # Plot learning curves
    print("Plotting learning curves...")
    plot_learning_curves(agent.episode_lengths, agent.episode_returns,
                        window=100, title="Monte Carlo ES Learning")

    # Plot value heatmap
    print("Plotting value heatmap...")
    plot_value_heatmap(maze, V, title="State Values V(s)")
