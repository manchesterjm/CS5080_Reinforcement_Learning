"""Regenerate all figures from corrected 7x6 layout."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from parking_lot import ParkingLot, ACTION_DELTAS, NUM_ACTIONS
from q_learning import QLearningAgent, DoubleQLearningAgent
from dqn import DQNAgent

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Arrow directions for policy display
ARROW_DX = {0: 0, 1: 0, 2: -0.3, 3: 0.3}
ARROW_DY = {0: 0.3, 1: -0.3, 2: 0, 3: 0}


def draw_parking_lot(ax, env, goal_id, policy=None, values=None,
                     path=None, title=None):
    """Draw parking lot on given axes."""
    walkable = env._get_walkable_cells(goal_id)
    goal_pos = env.parking_spots[goal_id]
    path_set = set(path) if path else set()

    if values:
        vmin = min(values.values())
        vmax = max(values.values())
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlGn

    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            x, y = c, env.rows - 1 - r

            if pos in env.barrier_cells:
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#404040", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y, "###", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
            elif pos == goal_pos:
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#4CAF50", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y, f"G\n{env.spot_names[goal_id]}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            elif pos == env.entrance:
                color = "#E3F2FD"
                if values and pos in values:
                    color = cmap(norm(values[pos]))
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y + 0.25, "E", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")
                if policy and pos in policy:
                    action = policy[pos]
                    ax.annotate("", xy=(x + ARROW_DX[action],
                                        y + ARROW_DY[action]),
                                xytext=(x, y - 0.1),
                                arrowprops=dict(arrowstyle="->",
                                                color="blue", lw=1.5))
            elif pos in env.all_parking_positions and pos not in walkable:
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#FFCDD2", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                sid = [k for k, v in env.parking_spots.items()
                       if v == pos][0]
                ax.text(x, y, f"[{env.spot_names[sid]}]",
                        ha="center", va="center",
                        fontsize=7, color="#B71C1C")
            elif pos in walkable:
                color = "white"
                if values and pos in values:
                    color = cmap(norm(values[pos]))
                elif pos in path_set:
                    color = "#FFF9C4"
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                if policy and pos in policy:
                    action = policy[pos]
                    ax.annotate("", xy=(x + ARROW_DX[action],
                                        y + ARROW_DY[action]),
                                xytext=(x, y),
                                arrowprops=dict(arrowstyle="->",
                                                color="blue", lw=1.5))
            else:
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor="#E0E0E0", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)

    if path and len(path) > 1:
        px = [p[1] for p in path]
        py = [env.rows - 1 - p[0] for p in path]
        ax.plot(px, py, "o-", color="orange", markersize=4,
                linewidth=2, alpha=0.7, zorder=5)

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_yticklabels([str(env.rows - 1 - i) for i in range(env.rows)])
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    if title:
        ax.set_title(title, fontsize=10)


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def gen_environment_layout(env):
    """Figure: environment layout with BFS paths for 4 representative spots."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 14))
    for idx, gid in enumerate([0, 3, 4, 7]):
        ax = axes[idx // 2, idx % 2]
        bfs = env.bfs_paths[gid]
        draw_parking_lot(ax, env, gid, path=bfs,
                         title=f"{env.spot_names[gid]} — "
                               f"BFS: {len(bfs)-1} steps")
    fig.suptitle("Parking Lot Environment with BFS Shortest Paths",
                 fontsize=13)
    plt.tight_layout()
    save(fig, "environment_layout.pdf")


def gen_q_policies(env, agent, name="Q-Learning"):
    """Figure: learned policies for all 8 goals."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{name} Learned Policies", fontsize=14)
    for gid in range(env.num_goals):
        ax = axes[gid // 4, gid % 4]
        policy = agent.get_policy(gid)
        draw_parking_lot(ax, env, gid, policy=policy,
                         title=f"Goal: {env.spot_names[gid]}")
    plt.tight_layout()
    save(fig, "q_learning_policies.pdf")


def gen_dqn_policies(env, agent):
    """Figure: DQN learned policies."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("DQN Learned Policies", fontsize=14)
    for gid in range(env.num_goals):
        ax = axes[gid // 4, gid % 4]
        policy = agent.get_policy(gid)
        draw_parking_lot(ax, env, gid, policy=policy,
                         title=f"Goal: {env.spot_names[gid]}")
    plt.tight_layout()
    save(fig, "dqn_policies.pdf")


def gen_value_heatmaps(env, agent):
    """Figure: value heatmaps for representative goals."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Q-Learning State Values (V = max_a Q(s,a))", fontsize=14)
    for gid in range(env.num_goals):
        ax = axes[gid // 4, gid % 4]
        values = agent.get_values(gid)
        draw_parking_lot(ax, env, gid, values=values,
                         title=f"{env.spot_names[gid]}")
    plt.tight_layout()
    save(fig, "value_heatmaps.pdf")


def gen_q_overestimation(env):
    """Figure: Q vs Double Q value estimation comparison."""
    print("  Training Q-Learning for overestimation comparison...")
    q_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    dq_agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)

    q_estimates, dq_estimates = [], []
    check_interval = 50
    n_episodes = 5000

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
            est = q_agent.get_max_q_estimates()
            q_estimates.append(np.mean(list(est.values())))

    np.random.seed(42)
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
            est = dq_agent.get_max_q_estimates()
            dq_estimates.append(np.mean(list(est.values())))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    window = 10
    for data, label, color in [(q_estimates, "Q-Learning", "tab:red"),
                                (dq_estimates, "Double Q-Learning", "tab:blue")]:
        eps = np.arange(1, len(data) + 1) * check_interval
        ax.plot(eps, data, alpha=0.3, color=color)
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window) / window, mode="valid")
            ax.plot(eps[window-1:], ma, label=label, color=color, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Max Q-Value")
    ax.set_title("Q-Value Estimates Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "q_overestimation.pdf")

    return q_agent, dq_agent


def gen_q_vs_dq_curves(env):
    """Figure: Q-learning vs Double Q-learning reward curves."""
    print("  Training for Q vs Double Q curves...")
    q_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    dq_agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)

    q_res = q_agent.train(n_episodes=5000)
    dq_res = dq_agent.train(n_episodes=5000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    window = 100
    colors = ["tab:blue", "tab:orange"]

    for i, (name, rewards) in enumerate([("Q-Learning", q_res["episode_rewards"]),
                                          ("Double Q", dq_res["episode_rewards"])]):
        eps = np.arange(1, len(rewards) + 1)
        ax1.plot(eps, rewards, alpha=0.15, color=colors[i])
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(np.arange(window, len(rewards) + 1), ma,
                 label=name, color=colors[i], linewidth=2)
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Q-Learning vs Double Q-Learning")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, (name, lengths) in enumerate([("Q-Learning", q_res["episode_lengths"]),
                                          ("Double Q", dq_res["episode_lengths"])]):
        eps = np.arange(1, len(lengths) + 1)
        ax2.plot(eps, lengths, alpha=0.15, color=colors[i])
        ma = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax2.plot(np.arange(window, len(lengths) + 1), ma,
                 label=name, color=colors[i], linewidth=2)
    ax2.set_ylabel("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, "q_vs_double_q_curves.pdf")


def gen_tabular_param_sensitivity(env):
    """Figure: tabular parameter sensitivity (3-panel)."""
    print("  Running tabular parameter sweeps...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    window = 100
    colors = plt.cm.tab10.colors

    # Alpha sweep
    for i, alpha in enumerate([0.01, 0.05, 0.1, 0.3, 0.5]):
        agent = QLearningAgent(env, alpha=alpha, gamma=0.99, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax1.plot(np.arange(window, len(rewards)+1), ma,
                 label=f"a={alpha}", color=colors[i], linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Learning Rate (alpha)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Gamma sweep
    for i, gamma in enumerate([0.9, 0.95, 0.99, 0.999]):
        agent = QLearningAgent(env, alpha=0.1, gamma=gamma, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax2.plot(np.arange(window, len(rewards)+1), ma,
                 label=f"g={gamma}", color=colors[i], linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_title("Discount Factor (gamma)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Epsilon decay sweep
    for i, eps_decay in enumerate([0.99, 0.995, 0.999]):
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99,
                               epsilon_decay=eps_decay, seed=42)
        res = agent.train(n_episodes=5000)
        rewards = res["episode_rewards"]
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax3.plot(np.arange(window, len(rewards)+1), ma,
                 label=f"decay={eps_decay}", color=colors[i], linewidth=1.5)
    ax3.set_xlabel("Episode")
    ax3.set_title("Exploration Decay")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Q-Learning Parameter Sensitivity", fontsize=13)
    plt.tight_layout()
    save(fig, "tabular_param_sensitivity.pdf")


def gen_dqn_vs_tabular(env):
    """Figure: DQN vs tabular convergence comparison."""
    print("  Training for DQN vs Tabular comparison...")
    q_agent = QLearningAgent(env, seed=42)
    q_res = q_agent.train(n_episodes=5000)

    dqn_agent = DQNAgent(env, hidden_size=128, lr=5e-4, seed=42,
                         epsilon_decay=0.998, memory_size=20000,
                         target_update=50)
    dqn_res = dqn_agent.train(n_episodes=5000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    window = 100

    for i, (name, rewards) in enumerate([("Q-Learning", q_res["episode_rewards"]),
                                          ("DQN", dqn_res["episode_rewards"])]):
        eps = np.arange(1, len(rewards) + 1)
        ax1.plot(eps, rewards, alpha=0.15, color=plt.cm.tab10.colors[i])
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax1.plot(np.arange(window, len(rewards)+1), ma,
                 label=name, color=plt.cm.tab10.colors[i], linewidth=2)
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Q-Learning vs DQN Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, (name, lengths) in enumerate([("Q-Learning", q_res["episode_lengths"]),
                                          ("DQN", dqn_res["episode_lengths"])]):
        eps = np.arange(1, len(lengths) + 1)
        ax2.plot(eps, lengths, alpha=0.15, color=plt.cm.tab10.colors[i])
        ma = np.convolve(lengths, np.ones(window)/window, mode="valid")
        ax2.plot(np.arange(window, len(lengths)+1), ma,
                 label=name, color=plt.cm.tab10.colors[i], linewidth=2)
    ax2.set_ylabel("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, "dqn_vs_tabular.pdf")

    return dqn_agent


def gen_dqn_multi_run(env):
    """Figure: DQN multi-run reliability."""
    print("  Running DQN multi-run (5 runs)...")
    n_runs = 5
    n_episodes = 5000
    all_rewards = []

    for run in range(n_runs):
        agent = DQNAgent(env, hidden_size=128, lr=5e-4, seed=run * 100,
                         epsilon_decay=0.998, memory_size=20000,
                         target_update=50)
        res = agent.train(n_episodes=n_episodes)
        all_rewards.append(res["episode_rewards"])
        summary = agent.evaluate(n_episodes=50)
        success = np.mean([s["success_rate"] for s in summary.values()])
        print(f"    Run {run+1}: {success:.0%}")

    min_len = min(len(r) for r in all_rewards)
    rewards_arr = np.array([r[:min_len] for r in all_rewards])
    mean_r = rewards_arr.mean(axis=0)
    std_r = rewards_arr.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    window = 100
    mean_ma = np.convolve(mean_r, np.ones(window)/window, mode="valid")
    std_ma = np.convolve(std_r, np.ones(window)/window, mode="valid")
    x = np.arange(window, min_len + 1)
    ax.plot(x, mean_ma, color="tab:blue", linewidth=2, label="Mean Reward")
    ax.fill_between(x, mean_ma - std_ma, mean_ma + std_ma,
                    alpha=0.2, color="tab:blue", label="+/- 1 Std")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"DQN Learning Curves ({n_runs} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "dqn_multi_run.pdf")


if __name__ == "__main__":
    env = ParkingLot()
    print(f"Grid: {env.rows}x{env.cols}, Spots: {env.num_goals}")
    print(f"Entrance: {env.entrance}")
    print(f"Barriers: {sorted(env.barrier_cells)}")

    # 1. Environment layout with BFS paths
    print("\n[1/8] Environment layout...")
    gen_environment_layout(env)

    # 2. Q vs Double Q overestimation (also gives us trained agents)
    print("\n[2/8] Q-value overestimation comparison...")
    q_agent, dq_agent = gen_q_overestimation(env)

    # 3. Q-learning policies (train fresh for clean results)
    print("\n[3/8] Q-Learning policies...")
    q_agent2 = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
    q_agent2.train(n_episodes=5000)
    gen_q_policies(env, q_agent2)

    # 4. Value heatmaps
    print("\n[4/8] Value heatmaps...")
    gen_value_heatmaps(env, q_agent2)

    # 5. Q vs Double Q learning curves
    print("\n[5/8] Q vs Double Q curves...")
    gen_q_vs_dq_curves(env)

    # 6. Tabular parameter sensitivity
    print("\n[6/8] Tabular parameter sensitivity...")
    gen_tabular_param_sensitivity(env)

    # 7. DQN vs Tabular comparison
    print("\n[7/8] DQN vs Tabular comparison...")
    dqn_agent = gen_dqn_vs_tabular(env)

    # 8. DQN policies + multi-run
    print("\n[8/8] DQN policies and multi-run...")
    gen_dqn_policies(env, dqn_agent)
    gen_dqn_multi_run(env)

    print("\nAll figures regenerated!")
