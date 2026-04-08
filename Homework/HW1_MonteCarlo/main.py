"""
Main Experiment Runner for HW1 Monte Carlo RL

Runs experiments for all homework questions:
1. Monte Carlo ES on HW1 maze
2. Parameter sensitivity analysis
3. Policy consistency across runs
4. Random mazes of different sizes

Usage:
    python main.py              # Run with matplotlib plots
    python main.py --no-plots   # Run text-only (no matplotlib needed)

Setup on any machine:
    pip install -r requirements.txt
    python main.py
"""

import numpy as np
import random
from typing import Dict, List, Tuple

from maze import Maze, create_hw1_maze
from monte_carlo import MonteCarloAgent
from maze_generator import generate_random_maze

# Defer matplotlib import -- only needed when show_plots=True
plot_maze = None
plot_learning_curves = None
plot_value_heatmap = None


def _load_visualize():
    """Load visualization functions on demand (requires matplotlib)."""
    global plot_maze, plot_learning_curves, plot_value_heatmap
    from visualize import plot_maze, plot_learning_curves, plot_value_heatmap


def run_single_experiment(maze: Maze,
                          gamma: float = 0.99,
                          n_episodes: int = 5000,
                          seed: int = None,
                          first_visit: bool = True,
                          verbose: bool = False) -> Tuple[MonteCarloAgent, Dict]:
    """
    Run a single Monte Carlo ES experiment.

    Args:
        maze: The maze environment
        gamma: Discount factor
        n_episodes: Number of training episodes
        seed: Random seed for reproducibility
        first_visit: Use first-visit MC (True) or every-visit MC (False)
        verbose: Print training progress

    Returns:
        Tuple of (trained agent, results dict)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    agent = MonteCarloAgent(maze, gamma=gamma)
    agent.train(n_episodes=n_episodes, first_visit=first_visit, verbose=verbose,
                print_interval=n_episodes // 5 if verbose else n_episodes)

    # Evaluate final policy
    success_rate, avg_steps, avg_return = agent.evaluate_policy(n_episodes=100)

    results = {
        'gamma': gamma,
        'n_episodes': n_episodes,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_return': avg_return,
        'final_episode_lengths': agent.episode_lengths[-100:],
        'final_episode_returns': agent.episode_returns[-100:]
    }

    return agent, results


def experiment_hw1_maze(show_plots: bool = True) -> None:
    """
    Q1-Q3: Train and analyze Monte Carlo ES on the HW1 maze.
    """
    print("=" * 60)
    print("EXPERIMENT 1: HW1 Maze (5x5)")
    print("=" * 60)

    maze = create_hw1_maze()
    print("\nMaze Layout:")
    print(maze.render())

    # Train agent
    print("\nTraining Monte Carlo ES agent...")
    agent, results = run_single_experiment(maze, gamma=0.99, n_episodes=5000,
                                           seed=42, verbose=True)

    # Show results
    print("\n--- Final Policy ---")
    agent.print_policy()
    agent.print_values()

    print(f"\n--- Evaluation Results ---")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Avg Steps to Goal: {results['avg_steps']:.1f}")
    print(f"Avg Return: {results['avg_return']:.4f}")

    if show_plots:
        _load_visualize()
        plot_maze(maze, policy=agent.policy, title="Learned Policy (HW1 Maze)")
        plot_learning_curves(agent.episode_lengths, agent.episode_returns,
                            window=100, title="Monte Carlo ES Learning")
        V = agent.get_state_values()
        plot_value_heatmap(maze, V, title="State Values V(s)")

    return agent


def experiment_policy_consistency(n_runs: int = 5, show_plots: bool = True) -> None:
    """
    Q4: Analyze policy consistency across multiple training runs.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Policy Consistency Analysis")
    print("=" * 60)

    maze = create_hw1_maze()
    policies = []
    results_list = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}...")
        agent, results = run_single_experiment(maze, gamma=0.99, n_episodes=5000,
                                               seed=run * 100, verbose=False)
        policies.append(agent.policy.copy())
        results_list.append(results)
        print(f"  Success Rate: {results['success_rate']*100:.0f}%, "
              f"Avg Steps: {results['avg_steps']:.1f}")

    # Analyze policy agreement
    print("\n--- Policy Agreement Analysis ---")
    states = [s for s in maze.states if not maze.is_terminal(s)]

    for state in states:
        actions = [p.get(state) for p in policies]
        action_names = [maze.ACTION_NAMES.get(a, '?') for a in actions]
        unique = len(set(actions))
        agreement = (n_runs - unique + 1) / n_runs * 100
        print(f"State {state}: {action_names} - {agreement:.0f}% agreement")

    # Summary statistics
    success_rates = [r['success_rate'] for r in results_list]
    avg_steps_list = [r['avg_steps'] for r in results_list]

    print(f"\n--- Summary Across {n_runs} Runs ---")
    print(f"Success Rate: {np.mean(success_rates)*100:.1f}% "
          f"(std: {np.std(success_rates)*100:.1f}%)")
    print(f"Avg Steps: {np.mean(avg_steps_list):.1f} "
          f"(std: {np.std(avg_steps_list):.1f})")


def experiment_parameter_sensitivity(show_plots: bool = True) -> None:
    """
    Parameter sensitivity analysis for gamma and episode count.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Parameter Sensitivity")
    print("=" * 60)

    maze = create_hw1_maze()

    # Test different gamma values
    gammas = [0.9, 0.95, 0.99, 0.999]
    print("\n--- Gamma Sensitivity ---")

    for gamma in gammas:
        agent, results = run_single_experiment(maze, gamma=gamma, n_episodes=5000,
                                               seed=42, verbose=False)
        print(f"gamma={gamma}: Success={results['success_rate']*100:.0f}%, "
              f"Steps={results['avg_steps']:.1f}, Return={results['avg_return']:.4f}")

    # Test different episode counts
    episode_counts = [1000, 2500, 5000, 10000]
    print("\n--- Episode Count Sensitivity ---")

    for n_ep in episode_counts:
        agent, results = run_single_experiment(maze, gamma=0.99, n_episodes=n_ep,
                                               seed=42, verbose=False)
        print(f"episodes={n_ep}: Success={results['success_rate']*100:.0f}%, "
              f"Steps={results['avg_steps']:.1f}")


def experiment_random_mazes(show_plots: bool = True) -> None:
    """
    Q5: Test on randomly generated mazes of different sizes.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Random Mazes of Different Sizes")
    print("=" * 60)

    sizes = [(5, 5), (7, 7), (10, 10)]

    for rows, cols in sizes:
        print(f"\n--- {rows}x{cols} Random Maze ---")

        # Generate random maze
        maze = generate_random_maze(rows, cols, barrier_density=0.2, seed=42)
        print(maze.render())
        print(f"States: {maze.n_states}, Barriers: {len(maze.barriers)}")

        # Scale episodes with maze size
        n_episodes = 5000 * (rows * cols) // 25

        print(f"\nTraining with {n_episodes} episodes...")
        agent, results = run_single_experiment(maze, gamma=0.99,
                                               n_episodes=n_episodes,
                                               seed=42, verbose=True)

        print(f"\nResults:")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Avg Steps: {results['avg_steps']:.1f}")

        if show_plots:
            _load_visualize()
            plot_maze(maze, policy=agent.policy,
                     title=f"Learned Policy ({rows}x{cols} Random Maze)")


def experiment_first_vs_every_visit(show_plots: bool = True) -> None:
    """
    Extra Credit: Compare first-visit MC vs every-visit MC.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: First-Visit vs Every-Visit MC")
    print("=" * 60)

    maze = create_hw1_maze()

    for label, fv in [("First-Visit", True), ("Every-Visit", False)]:
        print(f"\n--- {label} MC-ES ---")
        agent, results = run_single_experiment(maze, gamma=0.99, n_episodes=5000,
                                               seed=42, first_visit=fv, verbose=True)
        agent.print_policy()
        print(f"\nSuccess Rate: {results['success_rate']*100:.0f}%, "
              f"Avg Steps: {results['avg_steps']:.1f}, "
              f"Avg Return: {results['avg_return']:.4f}")


def main():
    """Run all experiments."""
    import sys

    show_plots = "--no-plots" not in sys.argv

    if not show_plots:
        print("(Running without plots -- text output only)")
        print()

    print("Monte Carlo RL Experiments - HW1")
    print("=" * 60)

    # Core HW1 maze experiments
    experiment_hw1_maze(show_plots=show_plots)

    # Policy consistency
    experiment_policy_consistency(n_runs=5, show_plots=show_plots)

    # Parameter sensitivity
    experiment_parameter_sensitivity(show_plots=show_plots)

    # Random mazes of different sizes
    experiment_random_mazes(show_plots=show_plots)

    # Extra credit: first-visit vs every-visit comparison
    experiment_first_vs_every_visit(show_plots=show_plots)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
