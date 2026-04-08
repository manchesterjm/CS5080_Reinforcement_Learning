"""
Train all agents across all grid sizes and save to disk.

Run once before the demo. Saves to models/ directory.
Models: Q-learning, Double Q-learning, DQN
Grids: 6x7, 10x12, 14x16, 18x20

Usage:
    python train_and_save.py           # Full training
    python train_and_save.py --quick   # Reduced episodes for testing
"""

import os
import pickle
import argparse
import numpy as np
import torch

from parking_lot import ParkingLot
from q_learning import QLearningAgent, DoubleQLearningAgent
from dqn import DQNAgent

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Grid configurations: (rows, cols, spots_per_row, label)
CONFIGS = [
    (7, 6, 4, "7x6"),
    (10, 12, 8, "10x12"),
    (14, 16, 10, "14x16"),
    (18, 20, 12, "18x20"),
]


def make_env(rows, cols, spr):
    """Create environment, using default for 7x6."""
    if rows == 7 and cols == 6:
        return ParkingLot()
    return ParkingLot(rows=rows, cols=cols, spots_per_row=spr)


def save_tabular_agent(agent, path):
    """Save a tabular agent's Q-table and metadata."""
    data = {
        "Q": dict(agent.Q) if hasattr(agent, "Q") else None,
        "Q1": dict(agent.Q1) if hasattr(agent, "Q1") else None,
        "Q2": dict(agent.Q2) if hasattr(agent, "Q2") else None,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon_min": agent.epsilon_min,
        "episode_rewards": agent.episode_rewards,
        "episode_lengths": agent.episode_lengths,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_dqn_agent(agent, path):
    """Save a DQN agent's network and metadata."""
    data = {
        "q_net_state": agent.q_net.state_dict(),
        "target_net_state": agent.target_net.state_dict(),
        "row_norm": agent.row_norm,
        "col_norm": agent.col_norm,
        "gamma": agent.gamma,
        "episode_rewards": agent.episode_rewards,
        "episode_lengths": agent.episode_lengths,
        "losses": agent.losses,
    }
    torch.save(data, path)


def train_all(quick=False):
    """Train and save all agents for all grid sizes."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    scale = 0.2 if quick else 1.0

    for rows, cols, spr, label in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"GRID: {label} ({rows}x{cols})")
        print(f"{'=' * 60}")

        env = make_env(rows, cols, spr)
        print(f"  Spots: {env.num_goals}, "
              f"Avg BFS: {np.mean([len(p)-1 for p in env.bfs_paths.values()]):.1f}")

        # Scale episodes with complexity
        complexity = (rows * cols) / (6 * 7)
        n_tabular = int(5000 * complexity * scale)
        n_dqn = int(3000 * complexity * scale)

        # Save environment config
        env_path = os.path.join(MODELS_DIR, f"env_{label}.pkl")
        env_data = {
            "rows": rows, "cols": cols, "spots_per_row": spr,
            "num_goals": env.num_goals,
            "parking_spots": env.parking_spots,
            "spot_names": env.spot_names,
            "barrier_cells": env.barrier_cells,
            "entrance": env.entrance,
            "bfs_paths": env.bfs_paths,
        }
        with open(env_path, "wb") as f:
            pickle.dump(env_data, f)

        # Q-Learning
        print(f"\n  Training Q-Learning ({n_tabular} episodes)...")
        q_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
        q_agent.train(n_episodes=n_tabular, verbose=True,
                      print_interval=max(1, n_tabular // 4))
        q_summary = q_agent.evaluate(n_episodes=50)
        q_success = np.mean([s["success_rate"]
                             for s in q_summary.values()])
        print(f"  Q-Learning success: {q_success:.0%}")
        save_tabular_agent(q_agent,
                           os.path.join(MODELS_DIR, f"q_{label}.pkl"))

        # Double Q-Learning
        print(f"\n  Training Double Q-Learning ({n_tabular} episodes)...")
        dq_agent = DoubleQLearningAgent(env, alpha=0.1, gamma=0.99, seed=42)
        dq_agent.train(n_episodes=n_tabular, verbose=True,
                       print_interval=max(1, n_tabular // 4))
        dq_summary = dq_agent.evaluate(n_episodes=50)
        dq_success = np.mean([s["success_rate"]
                              for s in dq_summary.values()])
        print(f"  Double Q-Learning success: {dq_success:.0%}")
        save_tabular_agent(dq_agent,
                           os.path.join(MODELS_DIR, f"dq_{label}.pkl"))

        # DQN
        print(f"\n  Training DQN ({n_dqn} episodes)...")
        dqn_agent = DQNAgent(env, hidden_size=128, lr=5e-4, seed=42,
                             epsilon_decay=0.998, memory_size=20000,
                             target_update=50)
        dqn_agent.train(n_episodes=n_dqn, verbose=True,
                        print_interval=max(1, n_dqn // 4))
        dqn_summary = dqn_agent.evaluate(n_episodes=50)
        dqn_success = np.mean([s["success_rate"]
                               for s in dqn_summary.values()])
        print(f"  DQN success: {dqn_success:.0%}")
        save_dqn_agent(dqn_agent,
                       os.path.join(MODELS_DIR, f"dqn_{label}.pkl"))

        print(f"\n  Saved all agents for {label}")

    print(f"\n{'=' * 60}")
    print(f"ALL MODELS SAVED to {MODELS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced episodes for testing")
    args = parser.parse_args()
    train_all(quick=args.quick)
