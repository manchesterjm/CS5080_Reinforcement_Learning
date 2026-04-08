"""
Generate publication-quality figures for HW2 paper.

Saves all figures as PDFs to figures/ directory at 300 DPI.

Usage:
    python generate_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

from parking_lot import ParkingLot, NUM_GOALS, SPOT_NAMES, print_bfs_summary
from q_learning import QLearningAgent, DoubleQLearningAgent
from dqn import DQNAgent
from visualize import (plot_parking_lot, plot_learning_curves,
                       plot_multi_policy, plot_parameter_sensitivity,
                       plot_q_overestimation, plot_value_heatmap)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(filename):
    """Save current figure to figures/ directory."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig_environment_layout():
    """Figure 1: Environment layout with BFS paths for 4 representative goals."""
    print("\n  Generating: environment_layout.pdf")
    env = ParkingLot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Parking Lot Environment with BFS Shortest Paths",
                 fontsize=14)

    for idx, gid in enumerate([0, 3, 4, 7]):
        ax = axes[idx // 2, idx % 2]
        path = env.bfs_paths[gid]
        plot_parking_lot(goal_id=gid, path=path,
                         title=f"Goal: {SPOT_NAMES[gid]} "
                               f"({len(path)-1} steps)",
                         ax=ax, show=False)

    plt.tight_layout()
    save("environment_layout.pdf")


def fig_q_learning_policies():
    """Figure 2: Q-learning learned policies for all 8 goals."""
    print("\n  Generating: q_learning_policies.pdf")
    env = ParkingLot()
    agent = QLearningAgent(env, seed=42)
    agent.train(n_episodes=5000)
    policies = {gid: agent.get_policy(gid) for gid in range(NUM_GOALS)}
    plot_multi_policy(policies, agent_name="Q-Learning",
                      show=False,
                      save_path=os.path.join(FIGURES_DIR,
                                             "q_learning_policies.pdf"))
    print(f"  Saved: {FIGURES_DIR}/q_learning_policies.pdf")


def fig_q_vs_double_q_curves():
    """Figure 3: Q-learning vs Double Q-learning learning curves."""
    print("\n  Generating: q_vs_double_q_curves.pdf")
    env = ParkingLot()

    q_agent = QLearningAgent(env, seed=42)
    q_res = q_agent.train(n_episodes=5000)

    dq_agent = DoubleQLearningAgent(env, seed=42)
    dq_res = dq_agent.train(n_episodes=5000)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_learning_curves(
        rewards_dict={
            "Q-Learning": q_res["episode_rewards"],
            "Double Q-Learning": dq_res["episode_rewards"],
        },
        lengths_dict={
            "Q-Learning": q_res["episode_lengths"],
            "Double Q-Learning": dq_res["episode_lengths"],
        },
        title="Q-Learning vs Double Q-Learning",
        show=False,
    )
    save("q_vs_double_q_curves.pdf")


def fig_q_overestimation():
    """Figure 4: Q-value overestimation comparison."""
    print("\n  Generating: q_overestimation.pdf")
    env = ParkingLot()
    n_episodes = 5000
    check_interval = 50

    q_estimates = []
    dq_estimates = []

    q_agent = QLearningAgent(env, seed=42)
    for ep in range(n_episodes):
        goal_id = np.random.randint(NUM_GOALS)
        state = env.reset(goal_id)
        for _ in range(env.max_steps):
            action = q_agent.select_action(state, goal_id)
            ns, r, done = env.step(action)
            q_agent.update(state, action, r, ns, done, goal_id)
            state = ns
            if done:
                break
        q_agent.epsilon = max(q_agent.epsilon_min,
                              q_agent.epsilon * q_agent.epsilon_decay)
        if (ep + 1) % check_interval == 0:
            est = q_agent.get_max_q_estimates()
            q_estimates.append(np.mean(list(est.values())))

    dq_agent = DoubleQLearningAgent(env, seed=42)
    for ep in range(n_episodes):
        goal_id = np.random.randint(NUM_GOALS)
        state = env.reset(goal_id)
        for _ in range(env.max_steps):
            action = dq_agent.select_action(state, goal_id)
            ns, r, done = env.step(action)
            dq_agent.update(state, action, r, ns, done, goal_id)
            state = ns
            if done:
                break
        dq_agent.epsilon = max(dq_agent.epsilon_min,
                               dq_agent.epsilon * dq_agent.epsilon_decay)
        if (ep + 1) % check_interval == 0:
            est = dq_agent.get_max_q_estimates()
            dq_estimates.append(np.mean(list(est.values())))

    plot_q_overestimation(q_estimates, dq_estimates, window=10,
                          title="Q-Value Estimates: Q-Learning vs Double Q",
                          show=False,
                          save_path=os.path.join(FIGURES_DIR,
                                                 "q_overestimation.pdf"))
    print(f"  Saved: {FIGURES_DIR}/q_overestimation.pdf")


def fig_value_heatmaps():
    """Figure 5: Value heatmaps for representative goals."""
    print("\n  Generating: value_heatmaps.pdf")
    env = ParkingLot()
    agent = QLearningAgent(env, seed=42)
    agent.train(n_episodes=5000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Learning State Values", fontsize=14)

    for idx, gid in enumerate([0, 3, 4, 7]):
        ax = axes[idx // 2, idx % 2]
        values = agent.get_values(gid)
        plot_value_heatmap(values, gid,
                           title=f"Goal: {SPOT_NAMES[gid]}",
                           ax=ax, show=False)

    plt.tight_layout()
    save("value_heatmaps.pdf")


def fig_dqn_policies():
    """Figure 6: DQN learned policies for all 8 goals."""
    print("\n  Generating: dqn_policies.pdf")
    env = ParkingLot()
    agent = DQNAgent(env, seed=42)
    agent.train(n_episodes=3000)
    policies = {gid: agent.get_policy(gid) for gid in range(NUM_GOALS)}
    plot_multi_policy(policies, agent_name="DQN",
                      show=False,
                      save_path=os.path.join(FIGURES_DIR,
                                             "dqn_policies.pdf"))
    print(f"  Saved: {FIGURES_DIR}/dqn_policies.pdf")


def fig_dqn_multi_run():
    """Figure 7: DQN learning curves across multiple runs."""
    print("\n  Generating: dqn_multi_run.pdf")
    env = ParkingLot()
    n_runs = 5
    n_episodes = 3000
    all_rewards = []

    for run in range(n_runs):
        print(f"    Run {run+1}/{n_runs}...")
        agent = DQNAgent(env, seed=run * 100)
        res = agent.train(n_episodes=n_episodes)
        all_rewards.append(res["episode_rewards"])

    min_len = min(len(r) for r in all_rewards)
    rewards_arr = np.array([r[:min_len] for r in all_rewards])
    mean_r = rewards_arr.mean(axis=0)
    std_r = rewards_arr.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
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
    save("dqn_multi_run.pdf")


def fig_tabular_param_sensitivity():
    """Figure 8: Tabular Q-learning parameter sensitivity."""
    print("\n  Generating: tabular_param_sensitivity.pdf")
    env = ParkingLot()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Learning Parameter Sensitivity", fontsize=14)
    colors = plt.cm.tab10.colors

    # Alpha sweep
    for i, alpha in enumerate([0.01, 0.05, 0.1, 0.3, 0.5]):
        agent = QLearningAgent(env, alpha=alpha, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        window = 100
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0].plot(np.arange(window, len(rewards) + 1), ma,
                         label=f"α={alpha}", color=colors[i], linewidth=1.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Learning Rate (α)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Gamma sweep
    for i, gamma in enumerate([0.9, 0.95, 0.99, 0.999]):
        agent = QLearningAgent(env, gamma=gamma, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[1].plot(np.arange(window, len(rewards) + 1), ma,
                         label=f"γ={gamma}", color=colors[i], linewidth=1.5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Discount Factor (γ)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Epsilon decay sweep
    for i, eps_decay in enumerate([0.99, 0.995, 0.999]):
        agent = QLearningAgent(env, epsilon_decay=eps_decay, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[2].plot(np.arange(window, len(rewards) + 1), ma,
                         label=f"ε_decay={eps_decay}",
                         color=colors[i], linewidth=1.5)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Episode Reward")
    axes[2].set_title("Exploration Decay (ε)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save("tabular_param_sensitivity.pdf")


def fig_dqn_vs_tabular():
    """Figure 9: DQN vs Tabular convergence comparison."""
    print("\n  Generating: dqn_vs_tabular.pdf")
    env = ParkingLot()

    q_agent = QLearningAgent(env, seed=42)
    q_res = q_agent.train(n_episodes=5000)

    dqn_agent = DQNAgent(env, seed=42)
    dqn_res = dqn_agent.train(n_episodes=5000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    window = 100
    colors = {"Q-Learning": "tab:blue", "DQN": "tab:orange"}

    for name, rewards in [("Q-Learning", q_res["episode_rewards"]),
                           ("DQN", dqn_res["episode_rewards"])]:
        episodes = np.arange(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, alpha=0.1, color=colors[name])
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(np.arange(window, len(rewards) + 1), ma,
                     label=name, color=colors[name], linewidth=2)
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Q-Learning vs DQN Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, lengths in [("Q-Learning", q_res["episode_lengths"]),
                           ("DQN", dqn_res["episode_lengths"])]:
        episodes = np.arange(1, len(lengths) + 1)
        ax2.plot(episodes, lengths, alpha=0.1, color=colors[name])
        if len(lengths) >= window:
            ma = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax2.plot(np.arange(window, len(lengths) + 1), ma,
                     label=name, color=colors[name], linewidth=2)
    ax2.set_ylabel("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save("dqn_vs_tabular.pdf")


def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    fig_environment_layout()
    fig_q_learning_policies()
    fig_q_vs_double_q_curves()
    fig_q_overestimation()
    fig_value_heatmaps()
    fig_dqn_policies()
    fig_dqn_multi_run()
    fig_tabular_param_sensitivity()
    fig_dqn_vs_tabular()

    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
