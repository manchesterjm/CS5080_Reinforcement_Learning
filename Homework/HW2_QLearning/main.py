"""
Main Experiment Runner for HW2 Q-Learning / DQN

Runs all experiments for the homework:
1. Tabular Q-learning baseline
2. Double Q-learning comparison
3. Tabular parameter sensitivity
4. DQN baseline
5. DQN parameter sensitivity
6. DQN vs Tabular comparison
7. DQN parameter sensitivity
8. DQN vs Tabular comparison
9. Enhancement: scaling ramp-up across larger parking lots

Usage:
    python main.py              # Run with matplotlib plots
    python main.py --no-plots   # Text-only output
    python main.py --quick      # Reduced episodes for testing
"""

import argparse
import sys
import numpy as np
from typing import Dict

from parking_lot import ParkingLot, print_bfs_summary
from q_learning import (QLearningAgent, DoubleQLearningAgent,
                         print_evaluation)
from dqn import DQNAgent

# Defer matplotlib imports
plot_parking_lot = None
plot_learning_curves = None
plot_multi_policy = None
plot_parameter_sensitivity = None
plot_q_overestimation = None


def _load_viz():
    """Load visualization on demand."""
    global plot_parking_lot, plot_learning_curves, plot_multi_policy
    global plot_parameter_sensitivity, plot_q_overestimation
    from visualize import (plot_parking_lot, plot_learning_curves,
                           plot_multi_policy, plot_parameter_sensitivity,
                           plot_q_overestimation)


def experiment_q_learning_baseline(env, n_episodes=5000, show_plots=True):
    """Experiment 1: Tabular Q-learning on all 8 goals."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: TABULAR Q-LEARNING BASELINE")
    print("=" * 60)

    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    results = agent.train(n_episodes=n_episodes, verbose=True)
    summary = agent.evaluate()
    print_evaluation(summary, "Q-Learning", env)

    # Show policies for representative goals
    for gid in [0, 3, 4, 7]:  # P1, P4, P5, P8
        policy = agent.get_policy(gid)
        env.render(policy=policy, goal_id=gid)

    if show_plots:
        _load_viz()
        policies = {gid: agent.get_policy(gid) for gid in range(env.num_goals)}
        plot_multi_policy(policies, agent_name="Q-Learning")

    return agent, results


def experiment_double_q_learning(env, n_episodes=5000, show_plots=True):
    """Experiment 2: Double Q-learning comparison."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: DOUBLE Q-LEARNING")
    print("=" * 60)

    agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    results = agent.train(n_episodes=n_episodes, verbose=True)
    summary = agent.evaluate()
    print_evaluation(summary, "Double Q-Learning", env)

    for gid in [0, 3, 4, 7]:
        policy = agent.get_policy(gid)
        env.render(policy=policy, goal_id=gid)

    return agent, results


def experiment_q_vs_double_q(env, n_episodes=5000, show_plots=True):
    """Experiment 3: Q-learning vs Double Q-learning comparison."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Q-LEARNING vs DOUBLE Q-LEARNING")
    print("=" * 60)

    # Track Q-value estimates over training
    q_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    dq_agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)

    q_estimates = []
    dq_estimates = []
    check_interval = 50

    # Train Q-learning with periodic Q-value tracking
    print("  Training Q-Learning...")
    for ep in range(n_episodes):
        goal_id = np.random.randint(env.num_goals)
        state = env.reset(goal_id)
        for step in range(env.max_steps):
            action = q_agent.select_action(state, goal_id)
            next_state, reward, done = env.step(action)
            q_agent.update(state, action, reward, next_state, done, goal_id)
            state = next_state
            if done:
                break
        q_agent.episode_rewards.append(reward)
        q_agent.epsilon = max(q_agent.epsilon_min,
                              q_agent.epsilon * q_agent.epsilon_decay)
        if (ep + 1) % check_interval == 0:
            estimates = q_agent.get_max_q_estimates()
            q_estimates.append(np.mean(list(estimates.values())))

    print("  Training Double Q-Learning...")
    for ep in range(n_episodes):
        goal_id = np.random.randint(env.num_goals)
        state = env.reset(goal_id)
        for step in range(env.max_steps):
            action = dq_agent.select_action(state, goal_id)
            next_state, reward, done = env.step(action)
            dq_agent.update(state, action, reward, next_state, done, goal_id)
            state = next_state
            if done:
                break
        dq_agent.episode_rewards.append(reward)
        dq_agent.epsilon = max(dq_agent.epsilon_min,
                               dq_agent.epsilon * dq_agent.epsilon_decay)
        if (ep + 1) % check_interval == 0:
            estimates = dq_agent.get_max_q_estimates()
            dq_estimates.append(np.mean(list(estimates.values())))

    # Print final Q-value comparison
    q_final = q_agent.get_max_q_estimates()
    dq_final = dq_agent.get_max_q_estimates()
    print(f"\n  Final Mean Max Q-Values:")
    print(f"  {'Spot':<6} {'Q-Learning':>12} {'Double Q':>12}")
    print(f"  {'-'*30}")
    for gid in range(env.num_goals):
        print(f"  {env.spot_names[gid]:<6} {q_final[gid]:>12.2f} "
              f"{dq_final[gid]:>12.2f}")
    print(f"  {'Mean':<6} {np.mean(list(q_final.values())):>12.2f} "
          f"{np.mean(list(dq_final.values())):>12.2f}")

    if show_plots:
        _load_viz()
        plot_q_overestimation(q_estimates, dq_estimates, window=10,
                              title="Q-Value Estimates Over Training")

    return q_estimates, dq_estimates


def experiment_tabular_param_sensitivity(env, n_episodes=5000,
                                         show_plots=True):
    """Experiment 4: Parameter sensitivity for tabular Q-learning."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: TABULAR PARAMETER SENSITIVITY")
    print("=" * 60)

    # Alpha sweep
    print("\n  --- Alpha Sweep ---")
    alpha_results = {}
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        agent = QLearningAgent(env, alpha=alpha, gamma=0.99, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        avg_steps = np.mean([s["avg_steps"] for s in summary.values()])
        print(f"  alpha={alpha:.2f}: success={success:.0%}, "
              f"avg_steps={avg_steps:.1f}")
        alpha_results[str(alpha)] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    # Gamma sweep
    print("\n  --- Gamma Sweep ---")
    gamma_results = {}
    for gamma in [0.9, 0.95, 0.99, 0.999]:
        agent = QLearningAgent(env, alpha=0.1, gamma=gamma, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        avg_steps = np.mean([s["avg_steps"] for s in summary.values()])
        print(f"  gamma={gamma:.3f}: success={success:.0%}, "
              f"avg_steps={avg_steps:.1f}")
        gamma_results[str(gamma)] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    # Epsilon decay sweep
    print("\n  --- Epsilon Decay Sweep ---")
    eps_results = {}
    for eps_decay in [0.99, 0.995, 0.999]:
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99,
                               epsilon_decay=eps_decay, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        avg_steps = np.mean([s["avg_steps"] for s in summary.values()])
        print(f"  eps_decay={eps_decay:.3f}: success={success:.0%}, "
              f"avg_steps={avg_steps:.1f}")
        eps_results[str(eps_decay)] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    if show_plots:
        _load_viz()
        plot_parameter_sensitivity(alpha_results, "α",
                                   "Q-Learning: Learning Rate Sensitivity")
        plot_parameter_sensitivity(gamma_results, "γ",
                                   "Q-Learning: Discount Factor Sensitivity")
        plot_parameter_sensitivity(eps_results, "ε_decay",
                                   "Q-Learning: Exploration Decay Sensitivity")

    return alpha_results, gamma_results, eps_results


def experiment_dqn_baseline(env, n_episodes=3000, show_plots=True):
    """Experiment 5: DQN baseline."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: DQN BASELINE")
    print("=" * 60)

    agent = DQNAgent(env, seed=42)
    results = agent.train(n_episodes=n_episodes, verbose=True,
                          print_interval=500)
    summary = agent.evaluate()
    print_evaluation(summary, "DQN", env)

    for gid in [0, 3, 4, 7]:
        policy = agent.get_policy(gid)
        env.render(policy=policy, goal_id=gid)

    if show_plots:
        _load_viz()
        policies = {gid: agent.get_policy(gid) for gid in range(env.num_goals)}
        plot_multi_policy(policies, agent_name="DQN")

    return agent, results


def experiment_dqn_multi_run(env, n_episodes=3000, n_runs=5,
                              show_plots=True):
    """Experiment 6: DQN reliability across multiple runs."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 6: DQN MULTI-RUN ({n_runs} runs)")
    print("=" * 60)

    all_rewards = []
    all_lengths = []
    all_success = []

    for run in range(n_runs):
        print(f"\n  --- Run {run+1}/{n_runs} ---")
        agent = DQNAgent(env, seed=run * 100)
        res = agent.train(n_episodes=n_episodes, verbose=True,
                          print_interval=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        avg_steps = np.mean([s["avg_steps"] for s in summary.values()])
        all_rewards.append(res["episode_rewards"])
        all_lengths.append(res["episode_lengths"])
        all_success.append(success)
        print(f"  Run {run+1}: success={success:.0%}, "
              f"avg_steps={avg_steps:.1f}")

    # Summary
    print(f"\n  DQN Multi-Run Summary ({n_runs} runs):")
    print(f"  Success Rate: {np.mean(all_success):.1%} "
          f"± {np.std(all_success):.1%}")

    if show_plots:
        _load_viz()
        # Plot mean ± std of reward curves
        min_len = min(len(r) for r in all_rewards)
        rewards_arr = np.array([r[:min_len] for r in all_rewards])
        mean_r = rewards_arr.mean(axis=0)
        std_r = rewards_arr.std(axis=0)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        episodes = np.arange(1, min_len + 1)
        window = 100
        if min_len >= window:
            mean_ma = np.convolve(mean_r, np.ones(window) / window,
                                  mode="valid")
            std_ma = np.convolve(std_r, np.ones(window) / window,
                                 mode="valid")
            x = np.arange(window, min_len + 1)
            ax.plot(x, mean_ma, color="tab:blue", linewidth=2,
                    label="Mean Reward")
            ax.fill_between(x, mean_ma - std_ma, mean_ma + std_ma,
                            alpha=0.2, color="tab:blue", label="±1 Std")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"DQN Learning Curves ({n_runs} runs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return all_rewards, all_success


def experiment_dqn_param_sensitivity(env, n_episodes=2000,
                                      show_plots=True):
    """Experiment 7: DQN parameter sensitivity."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: DQN PARAMETER SENSITIVITY")
    print("=" * 60)

    # Learning rate sweep
    print("\n  --- Learning Rate Sweep ---")
    lr_results = {}
    for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
        agent = DQNAgent(env, lr=lr, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        print(f"  lr={lr:.0e}: success={success:.0%}")
        lr_results[f"{lr:.0e}"] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    # Memory size sweep
    print("\n  --- Memory Size Sweep ---")
    mem_results = {}
    for mem_size in [1000, 5000, 10000, 50000]:
        agent = DQNAgent(env, memory_size=mem_size, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        print(f"  memory={mem_size}: success={success:.0%}")
        mem_results[str(mem_size)] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    # Target update frequency sweep
    print("\n  --- Target Update Frequency Sweep ---")
    tu_results = {}
    for tu in [50, 100, 200, 500]:
        agent = DQNAgent(env, target_update=tu, seed=42)
        res = agent.train(n_episodes=n_episodes)
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        print(f"  target_update={tu}: success={success:.0%}")
        tu_results[str(tu)] = {
            "rewards": res["episode_rewards"],
            "lengths": res["episode_lengths"],
        }

    if show_plots:
        _load_viz()
        plot_parameter_sensitivity(lr_results, "lr",
                                   "DQN: Learning Rate Sensitivity")
        plot_parameter_sensitivity(mem_results, "memory",
                                   "DQN: Replay Memory Size")
        plot_parameter_sensitivity(tu_results, "target_update",
                                   "DQN: Target Network Update Frequency")

    return lr_results, mem_results, tu_results


def experiment_dqn_vs_tabular(env, n_episodes=5000, show_plots=True):
    """Experiment 8: DQN vs Tabular Q-learning convergence comparison."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: DQN vs TABULAR COMPARISON")
    print("=" * 60)

    print("  Training Q-Learning...")
    q_agent = QLearningAgent(env, seed=42)
    q_res = q_agent.train(n_episodes=n_episodes, verbose=True,
                          print_interval=2500)

    print("\n  Training DQN...")
    dqn_agent = DQNAgent(env, seed=42)
    dqn_res = dqn_agent.train(n_episodes=n_episodes, verbose=True,
                              print_interval=2500)

    q_summary = q_agent.evaluate()
    dqn_summary = dqn_agent.evaluate()

    print_evaluation(q_summary, "Q-Learning", env)
    print_evaluation(dqn_summary, "DQN", env)

    if show_plots:
        _load_viz()
        plot_learning_curves(
            rewards_dict={
                "Q-Learning": q_res["episode_rewards"],
                "DQN": dqn_res["episode_rewards"],
            },
            lengths_dict={
                "Q-Learning": q_res["episode_lengths"],
                "DQN": dqn_res["episode_lengths"],
            },
            title="Q-Learning vs DQN Convergence",
        )

    return q_res, dqn_res


def experiment_scaling_rampup(base_episodes=5000, quick=False,
                              show_plots=True):
    """Experiment 9: Scaling ramp-up across progressively larger lots.

    Tests Q-learning, Double Q-learning, and DQN on increasing grid
    sizes to demonstrate how complexity affects convergence.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: SCALING RAMP-UP (ENHANCEMENT)")
    print("=" * 60)

    # Grid configurations: (rows, cols, spots_per_row, label)
    if quick:
        configs = [
            (6, 7, 4, "6x7"),
            (10, 12, 8, "10x12"),
            (14, 16, 10, "14x16"),
        ]
    else:
        configs = [
            (6, 7, 4, "6x7"),
            (10, 12, 8, "10x12"),
            (14, 16, 10, "14x16"),
            (18, 20, 12, "18x20"),
        ]

    # Scale episodes with grid size
    episode_scale = 0.2 if quick else 1.0
    results_table = []

    for rows, cols, spr, label in configs:
        print(f"\n{'-' * 60}")
        print(f"  Grid: {label} ({rows}x{cols})")
        print(f"{'-' * 60}")

        if rows == 6 and cols == 7:
            env = ParkingLot()
        else:
            env = ParkingLot(rows=rows, cols=cols, spots_per_row=spr)

        n_spots = env.num_goals
        avg_bfs = np.mean([len(p) - 1 for p in env.bfs_paths.values()])
        print(f"  Spots: {n_spots}, Avg BFS path: {avg_bfs:.1f} steps")

        # Scale episodes with complexity
        complexity_factor = (rows * cols) / (6 * 7)
        n_episodes_tabular = int(base_episodes * complexity_factor
                                 * episode_scale)
        n_episodes_dqn = int(3000 * complexity_factor * episode_scale)

        print(f"  Episodes: tabular={n_episodes_tabular}, "
              f"dqn={n_episodes_dqn}")

        row_result = {
            "grid": label,
            "spots": n_spots,
            "avg_bfs": avg_bfs,
            "state_space": rows * cols,
        }

        # Q-Learning
        print(f"\n  Training Q-Learning on {label}...")
        q_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
        q_res = q_agent.train(n_episodes=n_episodes_tabular, verbose=True,
                              print_interval=max(1, n_episodes_tabular // 3))
        q_summary = q_agent.evaluate(n_episodes=50)
        q_success = np.mean([s["success_rate"]
                             for s in q_summary.values()])
        q_avg_steps = np.mean([s["avg_steps"]
                               for s in q_summary.values()])
        row_result["q_success"] = q_success
        row_result["q_avg_steps"] = q_avg_steps

        # Double Q-Learning
        print(f"\n  Training Double Q-Learning on {label}...")
        dq_agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
        dq_res = dq_agent.train(n_episodes=n_episodes_tabular, verbose=True,
                                print_interval=max(1,
                                                   n_episodes_tabular // 3))
        dq_summary = dq_agent.evaluate(n_episodes=50)
        dq_success = np.mean([s["success_rate"]
                              for s in dq_summary.values()])
        dq_avg_steps = np.mean([s["avg_steps"]
                                for s in dq_summary.values()])
        row_result["dq_success"] = dq_success
        row_result["dq_avg_steps"] = dq_avg_steps

        # DQN
        print(f"\n  Training DQN on {label}...")
        dqn_agent = DQNAgent(env, hidden_size=64, seed=42)
        dqn_res = dqn_agent.train(n_episodes=n_episodes_dqn, verbose=True,
                                  print_interval=max(1,
                                                     n_episodes_dqn // 3))
        dqn_summary = dqn_agent.evaluate(n_episodes=50)
        dqn_success = np.mean([s["success_rate"]
                               for s in dqn_summary.values()])
        dqn_avg_steps = np.mean([s["avg_steps"]
                                 for s in dqn_summary.values()])
        row_result["dqn_success"] = dqn_success
        row_result["dqn_avg_steps"] = dqn_avg_steps

        results_table.append(row_result)

    # Print summary table
    print("\n" + "=" * 60)
    print("SCALING RAMP-UP SUMMARY")
    print("=" * 60)
    print(f"\n  {'Grid':<8} {'Spots':>5} {'States':>6} {'BFS':>5} "
          f"| {'Q-Learn':>9} {'DblQ':>9} {'DQN':>9}")
    print(f"  {'':<8} {'':>5} {'':>6} {'avg':>5} "
          f"| {'success':>9} {'success':>9} {'success':>9}")
    print(f"  {'-' * 62}")

    for r in results_table:
        print(f"  {r['grid']:<8} {r['spots']:>5} {r['state_space']:>6} "
              f"{r['avg_bfs']:>5.1f} "
              f"| {r['q_success']:>8.0%} {r['dq_success']:>8.0%} "
              f"{r['dqn_success']:>8.0%}")

    print(f"\n  {'Grid':<8} "
          f"| {'Q-Learn':>9} {'DblQ':>9} {'DQN':>9}")
    print(f"  {'':<8} "
          f"| {'avg_steps':>9} {'avg_steps':>9} {'avg_steps':>9}")
    print(f"  {'-' * 40}")

    for r in results_table:
        print(f"  {r['grid']:<8} "
              f"| {r['q_avg_steps']:>9.1f} {r['dq_avg_steps']:>9.1f} "
              f"{r['dqn_avg_steps']:>9.1f}")

    return results_table


def main():
    parser = argparse.ArgumentParser(description="HW2 Q-Learning Experiments")
    parser.add_argument("--no-plots", action="store_true",
                        help="Run without matplotlib plots")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced episodes for quick testing")
    parser.add_argument("--experiment", type=int, default=0,
                        help="Run specific experiment (1-9), 0 for all")
    args = parser.parse_args()

    show_plots = not args.no_plots
    scale = 0.2 if args.quick else 1.0

    env = ParkingLot()
    print_bfs_summary(env)

    experiments = {
        1: ("Q-Learning Baseline",
            lambda: experiment_q_learning_baseline(
                env, n_episodes=int(5000 * scale), show_plots=show_plots)),
        2: ("Double Q-Learning",
            lambda: experiment_double_q_learning(
                env, n_episodes=int(5000 * scale), show_plots=show_plots)),
        3: ("Q vs Double Q Comparison",
            lambda: experiment_q_vs_double_q(
                env, n_episodes=int(5000 * scale), show_plots=show_plots)),
        4: ("Tabular Parameter Sensitivity",
            lambda: experiment_tabular_param_sensitivity(
                env, n_episodes=int(5000 * scale), show_plots=show_plots)),
        5: ("DQN Baseline",
            lambda: experiment_dqn_baseline(
                env, n_episodes=int(3000 * scale), show_plots=show_plots)),
        6: ("DQN Multi-Run",
            lambda: experiment_dqn_multi_run(
                env, n_episodes=int(3000 * scale),
                n_runs=3 if args.quick else 5, show_plots=show_plots)),
        7: ("DQN Parameter Sensitivity",
            lambda: experiment_dqn_param_sensitivity(
                env, n_episodes=int(2000 * scale), show_plots=show_plots)),
        8: ("DQN vs Tabular",
            lambda: experiment_dqn_vs_tabular(
                env, n_episodes=int(5000 * scale), show_plots=show_plots)),
        9: ("Scaling Ramp-Up (Enhancement)",
            lambda: experiment_scaling_rampup(
                base_episodes=5000, quick=args.quick,
                show_plots=show_plots)),
    }

    if args.experiment > 0:
        name, func = experiments[args.experiment]
        print(f"\nRunning Experiment {args.experiment}: {name}")
        func()
    else:
        for num, (name, func) in experiments.items():
            print(f"\n{'#' * 60}")
            print(f"# Running Experiment {num}: {name}")
            print(f"{'#' * 60}")
            func()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
