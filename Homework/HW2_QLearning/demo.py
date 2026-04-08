"""
Interactive Parking Lot Demo

Loads pre-trained agents and animates the car navigating to a
chosen parking spot. Run train_and_save.py first.

Usage:
    python demo.py              # Interactive mode
    python demo.py --grid 10x12 # Start with a specific grid
"""

import os
import sys
import pickle
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

import torch

from parking_lot import ParkingLot, NUM_ACTIONS, ACTION_DELTAS
from q_learning import QLearningAgent, DoubleQLearningAgent
from dqn import DQNAgent, QNetwork

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models")

# Colors
COLOR_LANE = "#E8F5E9"
COLOR_BARRIER = "#404040"
COLOR_SPOT_OPEN = "#BBDEFB"
COLOR_SPOT_BLOCKED = "#FFCDD2"
COLOR_GOAL = "#4CAF50"
COLOR_ENTRANCE = "#FFF9C4"
COLOR_CAR = "#F44336"
COLOR_PATH = "#FFE082"


def load_env(label):
    """Load environment config and create ParkingLot."""
    env_path = os.path.join(MODELS_DIR, f"env_{label}.pkl")
    if not os.path.exists(env_path):
        print(f"  Error: {env_path} not found. Run train_and_save.py first.")
        return None

    with open(env_path, "rb") as f:
        env_data = pickle.load(f)

    rows = env_data["rows"]
    cols = env_data["cols"]
    spr = env_data["spots_per_row"]

    if rows == 7 and cols == 6:
        return ParkingLot()
    return ParkingLot(rows=rows, cols=cols, spots_per_row=spr)


def load_q_agent(env, label):
    """Load a trained Q-learning agent."""
    path = os.path.join(MODELS_DIR, f"q_{label}.pkl")
    if not os.path.exists(path):
        return None

    agent = QLearningAgent(env, seed=0)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent.Q.update(data["Q"])
    agent.epsilon = agent.epsilon_min
    return agent


def load_dq_agent(env, label):
    """Load a trained Double Q-learning agent."""
    path = os.path.join(MODELS_DIR, f"dq_{label}.pkl")
    if not os.path.exists(path):
        return None

    agent = DoubleQLearningAgent(env, seed=0)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent.Q1.update(data["Q1"])
    agent.Q2.update(data["Q2"])
    agent.epsilon = agent.epsilon_min
    return agent


def load_dqn_agent(env, label):
    """Load a trained DQN agent."""
    path = os.path.join(MODELS_DIR, f"dqn_{label}.pkl")
    if not os.path.exists(path):
        return None

    data = torch.load(path, weights_only=True)
    # Infer hidden size from saved weights
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


def animate_path(env, path, goal_id, agent_name, grid_label):
    """Animate the car moving along the path."""
    fig, ax = plt.subplots(1, 1, figsize=(max(8, env.cols * 0.8),
                                          max(6, env.rows * 0.8)))
    fig.patch.set_facecolor("white")

    goal_name = env.spot_names[goal_id]
    steps = len(path) - 1
    bfs_len = len(env.bfs_paths[goal_id]) - 1

    # Draw static grid once
    draw_grid(ax, env, goal_id)

    # Car marker (will be updated each frame)
    r0, c0 = path[0]
    car_circle = mpatches.Circle(
        (c0, env.rows - 1 - r0), 0.3,
        facecolor=COLOR_CAR, edgecolor="darkred", linewidth=2, zorder=10)
    ax.add_patch(car_circle)

    # Trail dots
    trail_dots = []

    title_text = ax.set_title(
        f"{grid_label} | {agent_name} | Target: {goal_name}\n"
        f"Step 0/{steps}  (BFS optimal: {bfs_len})",
        fontsize=12, fontweight="bold")

    def update(frame):
        r, c = path[frame]
        x, y = c, env.rows - 1 - r

        # Add trail dot at previous position
        if frame > 0:
            pr, pc = path[frame - 1]
            dot = mpatches.Circle(
                (pc, env.rows - 1 - pr), 0.08,
                facecolor="#EF9A9A", edgecolor="none", zorder=5)
            ax.add_patch(dot)
            trail_dots.append(dot)

        # Move car
        car_circle.center = (x, y)

        # Update title
        status = "PARKED!" if frame == len(path) - 1 else f"Step {frame}/{steps}"
        title_text.set_text(
            f"{grid_label} | {agent_name} | Target: {goal_name}\n"
            f"{status}  (BFS optimal: {bfs_len})")

        return [car_circle, title_text] + trail_dots

    anim = FuncAnimation(fig, update, frames=len(path),
                         interval=400, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()

    return len(path) - 1


def list_available_models():
    """Show what's available in models/ directory."""
    if not os.path.exists(MODELS_DIR):
        print("No models/ directory. Run train_and_save.py first.")
        return []

    grids = []
    for f in os.listdir(MODELS_DIR):
        if f.startswith("env_") and f.endswith(".pkl"):
            label = f[4:-4]
            grids.append(label)
    # Sort by grid area (rows * cols)
    grids.sort(key=lambda g: int(g.split("x")[0]) * int(g.split("x")[1]))

    if not grids:
        print("No saved models found. Run train_and_save.py first.")
        return []

    print("\nAvailable grids:")
    for label in grids:
        agents = []
        if os.path.exists(os.path.join(MODELS_DIR, f"q_{label}.pkl")):
            agents.append("Q-Learning")
        if os.path.exists(os.path.join(MODELS_DIR, f"dq_{label}.pkl")):
            agents.append("Double Q")
        if os.path.exists(os.path.join(MODELS_DIR, f"dqn_{label}.pkl")):
            agents.append("DQN")

        with open(os.path.join(MODELS_DIR, f"env_{label}.pkl"), "rb") as f:
            env_data = pickle.load(f)
        n_spots = env_data["num_goals"]
        print(f"  {label}: {n_spots} spots | {', '.join(agents)}")

    return grids


def main():
    parser = argparse.ArgumentParser(description="Parking Lot Demo")
    parser.add_argument("--grid", type=str, default=None,
                        help="Grid size (e.g., 6x7, 10x12)")
    args = parser.parse_args()

    print("=" * 50)
    print("  PARKING LOT NAVIGATION DEMO")
    print("  Q-Learning / Double Q / DQN")
    print("=" * 50)

    grids = list_available_models()
    if not grids:
        sys.exit(1)

    while True:
        # Select grid
        print(f"\nGrids:")
        for i, g in enumerate(grids, 1):
            print(f"  [{i}] {g}")
        grid_input = input("\nPick a grid (or 'q' to quit): ").strip()
        if grid_input.lower() == "q":
            break
        try:
            grid_idx = int(grid_input) - 1
            if 0 <= grid_idx < len(grids):
                grid_label = grids[grid_idx]
            else:
                print(f"  Invalid. Pick 1-{len(grids)}.")
                continue
        except ValueError:
            if grid_input in grids:
                grid_label = grid_input
            else:
                print(f"  Invalid. Pick 1-{len(grids)}.")
                continue

        # Load environment
        env = load_env(grid_label)
        if env is None:
            continue

        # Select algorithm
        print(f"\nAlgorithms:")
        print(f"  [1] Q-Learning")
        print(f"  [2] Double Q-Learning")
        print(f"  [3] DQN")
        algo_input = input("Pick algorithm (1/2/3): ").strip()
        algo_map = {"1": "q", "2": "dq", "3": "dqn"}
        algo_names = {"1": "Q-Learning", "2": "Double Q-Learning",
                      "3": "DQN"}
        if algo_input not in algo_map:
            print("  Invalid. Pick 1, 2, or 3.")
            continue

        algo_key = algo_map[algo_input]
        agent_name = algo_names[algo_input]

        # Load agent
        if algo_key == "q":
            agent = load_q_agent(env, grid_label)
        elif algo_key == "dq":
            agent = load_dq_agent(env, grid_label)
        else:
            agent = load_dqn_agent(env, grid_label)

        if agent is None:
            print(f"  No saved {agent_name} model for {grid_label}.")
            continue

        # Select parking spot
        print(f"\nSpots: 1-{env.num_goals}")
        spot_input = input(f"Pick a spot (1-{env.num_goals}): ").strip()

        # Parse spot number
        if spot_input.upper().startswith("P"):
            spot_input = spot_input[1:]
        try:
            spot_num = int(spot_input) - 1  # Convert to 0-indexed
        except ValueError:
            print("  Invalid spot.")
            continue

        if spot_num < 0 or spot_num >= env.num_goals:
            print(f"  Invalid. Pick 1-{env.num_goals}.")
            continue

        # Generate path using trained agent (retry up to 20 times)
        max_attempts = 20
        path = None
        for attempt in range(1, max_attempts + 1):
            candidate = generate_path(agent, env, spot_num)
            if candidate[-1] == env.parking_spots[spot_num]:
                path = candidate
                break

        print(f"\n  {agent_name} on {grid_label} -> "
              f"{env.spot_names[spot_num]}")

        if path is None:
            print(f"  FAILED TO REACH target after {max_attempts} attempts")
            continue

        if attempt > 1:
            print(f"  Succeeded on attempt {attempt}/{max_attempts}")
        print(f"  Path: {len(path)-1} steps "
              f"(BFS optimal: {len(env.bfs_paths[spot_num])-1})")

        # Animate
        animate_path(env, path, spot_num, agent_name, grid_label)


if __name__ == "__main__":
    main()
