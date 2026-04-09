"""
Interactive Parking Lot Demo (7x6 grid)

Loads pre-trained agents and animates the car navigating to a
chosen parking spot. Run train_and_save.py first.

Usage:
    python demo.py
"""

import os
import sys
import pickle
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

import torch

from parking_lot import ParkingLot
from q_learning import QLearningAgent, DoubleQLearningAgent
from dqn import DQNAgent

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models")
GRID_LABEL = "7x6"

# Colors
COLOR_LANE = "#E8F5E9"
COLOR_BARRIER = "#404040"
COLOR_SPOT_OPEN = "#BBDEFB"
COLOR_SPOT_BLOCKED = "#FFCDD2"
COLOR_GOAL = "#4CAF50"
COLOR_ENTRANCE = "#FFF9C4"
COLOR_CAR = "#F44336"


def load_q_agent(env):
    """Load a trained Q-learning agent."""
    path = os.path.join(MODELS_DIR, f"q_{GRID_LABEL}.pkl")
    if not os.path.exists(path):
        return None

    agent = QLearningAgent(env, seed=0)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent.Q.update(data["Q"])
    agent.epsilon = agent.epsilon_min
    return agent


def load_dq_agent(env):
    """Load a trained Double Q-learning agent."""
    path = os.path.join(MODELS_DIR, f"dq_{GRID_LABEL}.pkl")
    if not os.path.exists(path):
        return None

    agent = DoubleQLearningAgent(env, seed=0)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent.Q1.update(data["Q1"])
    agent.Q2.update(data["Q2"])
    agent.epsilon = agent.epsilon_min
    return agent


def load_dqn_agent(env):
    """Load a trained DQN agent."""
    path = os.path.join(MODELS_DIR, f"dqn_{GRID_LABEL}.pkl")
    if not os.path.exists(path):
        return None

    data = torch.load(path, weights_only=True)
    hidden_size = data["q_net_state"]["net.0.weight"].shape[0]
    agent = DQNAgent(env, hidden_size=hidden_size, seed=0)
    agent.q_net.load_state_dict(data["q_net_state"])
    agent.target_net.load_state_dict(data["target_net_state"])
    agent.q_net.eval()
    agent.epsilon = agent.epsilon_min
    return agent


def generate_path(agent, env, goal_id):
    """Run the agent greedily and record the path taken."""
    state = env.reset(goal_id)
    path = [state]

    for _ in range(env.max_steps):
        action = agent.greedy_action(state, goal_id)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


def draw_grid(ax, env, goal_id):
    """Draw the static parking lot grid."""
    ax.clear()
    walkable = env._get_walkable_cells(goal_id)
    goal_pos = env.parking_spots[goal_id]

    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            x, y = c, env.rows - 1 - r

            if pos in env.barrier_cells:
                color = COLOR_BARRIER
                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.45), 0.9, 0.9,
                    facecolor=color, edgecolor="black",
                    linewidth=0.5, boxstyle="round,pad=0.02")
                ax.add_patch(rect)

            elif pos == goal_pos:
                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.45), 0.9, 0.9,
                    facecolor=COLOR_GOAL, edgecolor="black",
                    linewidth=0.5, boxstyle="round,pad=0.02")
                ax.add_patch(rect)
                ax.text(x, y, env.spot_names[goal_id],
                        ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")

            elif pos == env.entrance:
                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.45), 0.9, 0.9,
                    facecolor=COLOR_ENTRANCE, edgecolor="black",
                    linewidth=0.5, boxstyle="round,pad=0.02")
                ax.add_patch(rect)
                ax.text(x, y, "E", ha="center", va="center",
                        fontsize=10, fontweight="bold")

            elif pos in env.all_parking_positions:
                sid = [k for k, v in env.parking_spots.items()
                       if v == pos][0]
                if pos not in walkable:
                    color = COLOR_SPOT_BLOCKED
                    rect = mpatches.FancyBboxPatch(
                        (x - 0.45, y - 0.45), 0.9, 0.9,
                        facecolor=color, edgecolor="black",
                        linewidth=0.5, boxstyle="round,pad=0.02")
                    ax.add_patch(rect)
                    ax.text(x, y, env.spot_names[sid],
                            ha="center", va="center",
                            fontsize=8, color="#B71C1C")
                else:
                    rect = mpatches.FancyBboxPatch(
                        (x - 0.45, y - 0.45), 0.9, 0.9,
                        facecolor=COLOR_SPOT_OPEN, edgecolor="black",
                        linewidth=0.5, boxstyle="round,pad=0.02")
                    ax.add_patch(rect)
                    ax.text(x, y, env.spot_names[sid],
                            ha="center", va="center",
                            fontsize=8, color="#1565C0")
            else:
                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.45), 0.9, 0.9,
                    facecolor=COLOR_LANE, edgecolor="#BDBDBD",
                    linewidth=0.3, boxstyle="round,pad=0.02")
                ax.add_patch(rect)

    ax.set_xlim(-0.6, env.cols - 0.4)
    ax.set_ylim(-0.6, env.rows - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


def animate_path(env, path, goal_id, agent_name):
    """Animate the car moving along the path."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor("white")

    goal_name = env.spot_names[goal_id]
    steps = len(path) - 1
    bfs_len = len(env.bfs_paths[goal_id]) - 1

    draw_grid(ax, env, goal_id)

    r0, c0 = path[0]
    car_circle = mpatches.Circle(
        (c0, env.rows - 1 - r0), 0.3,
        facecolor=COLOR_CAR, edgecolor="darkred", linewidth=2, zorder=10)
    ax.add_patch(car_circle)

    trail_dots = []

    title_text = ax.set_title(
        f"{agent_name} | Target: {goal_name}\n"
        f"Step 0/{steps}  (BFS optimal: {bfs_len})",
        fontsize=12, fontweight="bold")

    def update(frame):
        r, c = path[frame]
        x, y = c, env.rows - 1 - r

        if frame > 0:
            pr, pc = path[frame - 1]
            dot = mpatches.Circle(
                (pc, env.rows - 1 - pr), 0.08,
                facecolor="#EF9A9A", edgecolor="none", zorder=5)
            ax.add_patch(dot)
            trail_dots.append(dot)

        car_circle.center = (x, y)

        status = "PARKED!" if frame == len(path) - 1 else f"Step {frame}/{steps}"
        title_text.set_text(
            f"{agent_name} | Target: {goal_name}\n"
            f"{status}  (BFS optimal: {bfs_len})")

        return [car_circle, title_text] + trail_dots

    anim = FuncAnimation(fig, update, frames=len(path),
                         interval=400, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()

    return len(path) - 1


def main():
    print("=" * 50)
    print("  PARKING LOT NAVIGATION DEMO (7x6)")
    print("  Q-Learning / Double Q / DQN")
    print("=" * 50)

    # Check models exist
    env_path = os.path.join(MODELS_DIR, f"env_{GRID_LABEL}.pkl")
    if not os.path.exists(env_path):
        print(f"\nError: {env_path} not found. Run train_and_save.py first.")
        sys.exit(1)

    env = ParkingLot()

    # Check which algorithms are available
    available = []
    if os.path.exists(os.path.join(MODELS_DIR, f"q_{GRID_LABEL}.pkl")):
        available.append(("Q-Learning", "q"))
    if os.path.exists(os.path.join(MODELS_DIR, f"dq_{GRID_LABEL}.pkl")):
        available.append(("Double Q-Learning", "dq"))
    if os.path.exists(os.path.join(MODELS_DIR, f"dqn_{GRID_LABEL}.pkl")):
        available.append(("DQN", "dqn"))

    if not available:
        print("\nNo trained models found. Run train_and_save.py first.")
        sys.exit(1)

    print(f"\nAvailable algorithms: {', '.join(name for name, _ in available)}")
    print(f"Parking spots: P1-P{env.num_goals}")

    while True:
        # Select algorithm
        print(f"\nAlgorithms:")
        for i, (name, _) in enumerate(available, 1):
            print(f"  [{i}] {name}")
        algo_input = input("\nPick algorithm (or 'q' to quit): ").strip()
        if algo_input.lower() == "q":
            break

        try:
            algo_idx = int(algo_input) - 1
            if not (0 <= algo_idx < len(available)):
                print(f"  Invalid. Pick 1-{len(available)}.")
                continue
        except ValueError:
            print(f"  Invalid. Pick 1-{len(available)}.")
            continue

        agent_name, algo_key = available[algo_idx]

        # Load agent
        if algo_key == "q":
            agent = load_q_agent(env)
        elif algo_key == "dq":
            agent = load_dq_agent(env)
        else:
            agent = load_dqn_agent(env)

        if agent is None:
            print(f"  Failed to load {agent_name} model.")
            continue

        # Select parking spot
        spot_input = input(f"Pick a spot (1-{env.num_goals}): ").strip()

        if spot_input.upper().startswith("P"):
            spot_input = spot_input[1:]
        try:
            spot_num = int(spot_input) - 1
        except ValueError:
            print("  Invalid spot.")
            continue

        if spot_num < 0 or spot_num >= env.num_goals:
            print(f"  Invalid. Pick 1-{env.num_goals}.")
            continue

        # Generate path (retry up to 20 times)
        max_attempts = 20
        path = None
        for attempt in range(1, max_attempts + 1):
            candidate = generate_path(agent, env, spot_num)
            if candidate[-1] == env.parking_spots[spot_num]:
                path = candidate
                break

        print(f"\n  {agent_name} -> {env.spot_names[spot_num]}")

        if path is None:
            print(f"  FAILED TO REACH target after {max_attempts} attempts")
            continue

        if attempt > 1:
            print(f"  Succeeded on attempt {attempt}/{max_attempts}")
        print(f"  Path: {len(path)-1} steps "
              f"(BFS optimal: {len(env.bfs_paths[spot_num])-1})")

        # Animate
        animate_path(env, path, spot_num, agent_name)


if __name__ == "__main__":
    main()
