"""
Tabular Q-Learning and Double Q-Learning Agents

Implements both algorithms from scratch using only NumPy.
No RL libraries used.
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from parking_lot import ParkingLot, NUM_ACTIONS, ACTION_NAMES, ACTION_ARROWS


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Q-learning update (off-policy TD control):
        Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    Reference: Sutton & Barto (2018), Section 6.5, p.131
    """

    def __init__(self, env: ParkingLot,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 seed: Optional[int] = None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if seed is not None:
            np.random.seed(seed)

        # Q-table: (row, col, goal_id, action) -> value
        self.Q = defaultdict(float)

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []

    def _state_key(self, state: Tuple[int, int],
                   goal_id: int) -> Tuple[int, int, int]:
        """Create hashable state key including goal."""
        return (state[0], state[1], goal_id)

    def _get_q_values(self, state: Tuple[int, int],
                      goal_id: int) -> np.ndarray:
        """Get Q-values for all actions at state."""
        sk = self._state_key(state, goal_id)
        return np.array([self.Q[(sk, a)] for a in range(NUM_ACTIONS)])

    def select_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            q_vals = self._get_q_values(state, goal_id)
            return int(np.argmax(q_vals))

    def greedy_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Pure greedy action (no exploration)."""
        q_vals = self._get_q_values(state, goal_id)
        return int(np.argmax(q_vals))

    def update(self, state: Tuple[int, int], action: int,
               reward: float, next_state: Tuple[int, int],
               done: bool, goal_id: int):
        """Single Q-learning update step."""
        sk = self._state_key(state, goal_id)
        nsk = self._state_key(next_state, goal_id)

        if done:
            td_target = reward
        else:
            max_next_q = max(self.Q[(nsk, a)] for a in range(NUM_ACTIONS))
            td_target = reward + self.gamma * max_next_q

        td_error = td_target - self.Q[(sk, action)]
        self.Q[(sk, action)] += self.alpha * td_error

    def train(self, n_episodes: int = 5000,
              verbose: bool = False,
              print_interval: int = 1000) -> Dict:
        """Train the agent across all goals.

        Each episode randomly selects a goal and trains to reach it.
        """
        for ep in range(n_episodes):
            goal_id = np.random.randint(self.env.num_goals)
            state = self.env.reset(goal_id)
            total_reward = 0.0
            steps = 0

            for step in range(self.env.max_steps):
                action = self.select_action(state, goal_id)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done, goal_id)

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % print_interval == 0:
                avg_r = np.mean(self.episode_rewards[-print_interval:])
                avg_l = np.mean(self.episode_lengths[-print_interval:])
                print(f"  Episode {ep+1:5d} | "
                      f"Avg Reward: {avg_r:7.2f} | "
                      f"Avg Length: {avg_l:5.1f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """Evaluate learned policy (greedy, no exploration).

        Returns per-goal and aggregate statistics.
        """
        results = {gid: {"successes": 0, "total_steps": [],
                         "total_rewards": []}
                   for gid in range(self.env.num_goals)}

        for _ in range(n_episodes):
            for goal_id in range(self.env.num_goals):
                state = self.env.reset(goal_id)
                total_reward = 0.0

                for step in range(self.env.max_steps):
                    action = self.greedy_action(state, goal_id)
                    next_state, reward, done = self.env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        results[goal_id]["successes"] += 1
                        break

                results[goal_id]["total_steps"].append(step + 1)
                results[goal_id]["total_rewards"].append(total_reward)

        # Aggregate
        summary = {}
        for gid in range(self.env.num_goals):
            r = results[gid]
            summary[gid] = {
                "success_rate": r["successes"] / n_episodes,
                "avg_steps": np.mean(r["total_steps"]),
                "avg_reward": np.mean(r["total_rewards"]),
            }
        return summary

    def get_policy(self, goal_id: int) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy for a specific goal."""
        policy = {}
        for state in self.env.get_all_states(goal_id):
            if state != self.env.parking_spots[goal_id]:
                policy[state] = self.greedy_action(state, goal_id)
        return policy

    def get_values(self, goal_id: int) -> Dict[Tuple[int, int], float]:
        """Extract state values V(s) = max_a Q(s,a) for a goal."""
        values = {}
        for state in self.env.get_all_states(goal_id):
            q_vals = self._get_q_values(state, goal_id)
            values[state] = float(np.max(q_vals))
        return values

    def get_max_q_estimates(self) -> Dict[int, float]:
        """Get mean max Q-value per goal (for overestimation analysis)."""
        estimates = {}
        for goal_id in range(self.env.num_goals):
            max_qs = []
            for state in self.env.get_all_states(goal_id):
                q_vals = self._get_q_values(state, goal_id)
                max_qs.append(float(np.max(q_vals)))
            estimates[goal_id] = np.mean(max_qs)
        return estimates


class DoubleQLearningAgent:
    """Double Q-learning to reduce maximization bias.

    Maintains two Q-tables. On each update, randomly picks which
    table to update, using the other for evaluation.

    Update (when updating Q1):
        Q1(s,a) += alpha * (r + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))

    Reference: Sutton & Barto (2018), Section 6.7, p.136
    """

    def __init__(self, env: ParkingLot,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 seed: Optional[int] = None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if seed is not None:
            np.random.seed(seed)

        # Two Q-tables
        self.Q1 = defaultdict(float)
        self.Q2 = defaultdict(float)

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []

    def _state_key(self, state: Tuple[int, int],
                   goal_id: int) -> Tuple[int, int, int]:
        return (state[0], state[1], goal_id)

    def _get_q_values(self, state: Tuple[int, int],
                      goal_id: int) -> np.ndarray:
        """Get combined Q-values (Q1 + Q2) / 2 for action selection."""
        sk = self._state_key(state, goal_id)
        return np.array([
            (self.Q1[(sk, a)] + self.Q2[(sk, a)]) / 2.0
            for a in range(NUM_ACTIONS)
        ])

    def select_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Epsilon-greedy using combined Q-values."""
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            q_vals = self._get_q_values(state, goal_id)
            return int(np.argmax(q_vals))

    def greedy_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Pure greedy using combined Q-values."""
        q_vals = self._get_q_values(state, goal_id)
        return int(np.argmax(q_vals))

    def update(self, state: Tuple[int, int], action: int,
               reward: float, next_state: Tuple[int, int],
               done: bool, goal_id: int):
        """Double Q-learning update: randomly update Q1 or Q2."""
        sk = self._state_key(state, goal_id)
        nsk = self._state_key(next_state, goal_id)

        if np.random.random() < 0.5:
            # Update Q1 using Q2 for evaluation
            if done:
                td_target = reward
            else:
                best_a = int(np.argmax([self.Q1[(nsk, a)]
                                        for a in range(NUM_ACTIONS)]))
                td_target = reward + self.gamma * self.Q2[(nsk, best_a)]
            td_error = td_target - self.Q1[(sk, action)]
            self.Q1[(sk, action)] += self.alpha * td_error
        else:
            # Update Q2 using Q1 for evaluation
            if done:
                td_target = reward
            else:
                best_a = int(np.argmax([self.Q2[(nsk, a)]
                                        for a in range(NUM_ACTIONS)]))
                td_target = reward + self.gamma * self.Q1[(nsk, best_a)]
            td_error = td_target - self.Q2[(sk, action)]
            self.Q2[(sk, action)] += self.alpha * td_error

    def train(self, n_episodes: int = 5000,
              verbose: bool = False,
              print_interval: int = 1000) -> Dict:
        """Train the agent across all goals."""
        for ep in range(n_episodes):
            goal_id = np.random.randint(self.env.num_goals)
            state = self.env.reset(goal_id)
            total_reward = 0.0
            steps = 0

            for step in range(self.env.max_steps):
                action = self.select_action(state, goal_id)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done, goal_id)

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % print_interval == 0:
                avg_r = np.mean(self.episode_rewards[-print_interval:])
                avg_l = np.mean(self.episode_lengths[-print_interval:])
                print(f"  Episode {ep+1:5d} | "
                      f"Avg Reward: {avg_r:7.2f} | "
                      f"Avg Length: {avg_l:5.1f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """Evaluate learned policy (greedy, no exploration)."""
        results = {gid: {"successes": 0, "total_steps": [],
                         "total_rewards": []}
                   for gid in range(self.env.num_goals)}

        for _ in range(n_episodes):
            for goal_id in range(self.env.num_goals):
                state = self.env.reset(goal_id)
                total_reward = 0.0

                for step in range(self.env.max_steps):
                    action = self.greedy_action(state, goal_id)
                    next_state, reward, done = self.env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        results[goal_id]["successes"] += 1
                        break

                results[goal_id]["total_steps"].append(step + 1)
                results[goal_id]["total_rewards"].append(total_reward)

        summary = {}
        for gid in range(self.env.num_goals):
            r = results[gid]
            summary[gid] = {
                "success_rate": r["successes"] / n_episodes,
                "avg_steps": np.mean(r["total_steps"]),
                "avg_reward": np.mean(r["total_rewards"]),
            }
        return summary

    def get_policy(self, goal_id: int) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy for a specific goal."""
        policy = {}
        for state in self.env.get_all_states(goal_id):
            if state != self.env.parking_spots[goal_id]:
                policy[state] = self.greedy_action(state, goal_id)
        return policy

    def get_values(self, goal_id: int) -> Dict[Tuple[int, int], float]:
        """Extract state values using combined Q-tables."""
        values = {}
        for state in self.env.get_all_states(goal_id):
            q_vals = self._get_q_values(state, goal_id)
            values[state] = float(np.max(q_vals))
        return values

    def get_max_q_estimates(self) -> Dict[int, float]:
        """Get mean max Q-value per goal (for overestimation analysis)."""
        estimates = {}
        for goal_id in range(self.env.num_goals):
            max_qs = []
            for state in self.env.get_all_states(goal_id):
                q_vals = self._get_q_values(state, goal_id)
                max_qs.append(float(np.max(q_vals)))
            estimates[goal_id] = np.mean(max_qs)
        return estimates


def print_evaluation(summary: Dict, agent_name: str, env=None):
    """Print evaluation results table."""
    if env is None:
        env = ParkingLot()

    print(f"\n  {agent_name} Evaluation Results:")
    print(f"  {'Spot':<6} {'Success':>8} {'Avg Steps':>10} "
          f"{'Avg Reward':>11} {'BFS Optimal':>12}")
    print(f"  {'-'*47}")

    for gid in range(env.num_goals):
        s = summary[gid]
        bfs_len = len(env.bfs_paths[gid]) - 1
        print(f"  {env.spot_names[gid]:<6} {s['success_rate']:>7.0%} "
              f"{s['avg_steps']:>10.1f} {s['avg_reward']:>11.2f} "
              f"{bfs_len:>12d}")


if __name__ == "__main__":
    env = ParkingLot()

    print("=" * 60)
    print("TABULAR Q-LEARNING")
    print("=" * 60)
    agent = QLearningAgent(env, seed=42)
    agent.train(n_episodes=5000, verbose=True)
    summary = agent.evaluate()
    print_evaluation(summary, "Q-Learning")

    # Show policy for P1
    policy = agent.get_policy(goal_id=0)
    env.render(policy=policy, goal_id=0)

    print("\n" + "=" * 60)
    print("DOUBLE Q-LEARNING")
    print("=" * 60)
    dagent = DoubleQLearningAgent(env, seed=42)
    dagent.train(n_episodes=5000, verbose=True)
    dsummary = dagent.evaluate()
    print_evaluation(dsummary, "Double Q-Learning")

    # Show policy for P1
    dpolicy = dagent.get_policy(goal_id=0)
    env.render(policy=dpolicy, goal_id=0)
